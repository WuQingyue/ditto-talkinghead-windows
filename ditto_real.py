###############################################################################
#  Integrate Ditto StreamSDK pipeline with WebRTC (aiortc) through BaseReal
#
#  This adapter reuses the existing stream_pipeline_online.py workers, but
#  replaces the file writer with a WebRTC writer and pushes audio in 20ms chunks
#  to the aiortc audio track.
###############################################################################

import asyncio
import time
import threading
import numpy as np
import soundfile as sf
import librosa

from av import VideoFrame, AudioFrame

from basereal import BaseReal
from stream_pipeline_online import StreamSDK


class _TTSEmpty:
    def render(self, quit_event):
        # No-op TTS for this adapter (audio comes from file)
        while not quit_event.is_set():
            time.sleep(0.1)
    def put_msg_txt(self, *args, **kwargs):
        pass
    def flush_talk(self):
        pass


class _ASREmpty:
    def put_audio_frame(self, *args, **kwargs):
        pass
    def flush_talk(self):
        pass


class WebRTCVideoWriter:
    def __init__(self, loop, video_track, expected_frames: int | None = None):
        self.loop = loop
        self.video_track = video_track
        self.closed = False
        self.expected = expected_frames
        self.count = 0
        self._last_frame = None
    def __call__(self, img, fmt="rgb"):
        if self.closed:
            return
        if fmt == "rgb":
            frame = VideoFrame.from_ndarray(img, format="rgb24")
        else:  # assume bgr
            frame = VideoFrame.from_ndarray(img, format="bgr24")
        # eventpoint is None for normal frames
        asyncio.run_coroutine_threadsafe(self.video_track._queue.put((frame, None)), self.loop)
        self.count += 1
        # keep last frame for potential tail padding
        self._last_frame = frame
    def close(self):
        self.closed = True
        # Pad tail with last frame if we wrote fewer than expected frames
        try:
            if self.expected is not None and self._last_frame is not None and self.count < self.expected:
                to_pad = self.expected - self.count
                for _ in range(to_pad):
                    asyncio.run_coroutine_threadsafe(self.video_track._queue.put((self._last_frame, None)), self.loop)
                self.count += to_pad
        except Exception:
            pass
        try:
            asyncio.run_coroutine_threadsafe(self.video_track._queue.put((None, 'eof_video')), self.loop)
        except Exception:
            pass


class AudioPusher(threading.Thread):
    def __init__(self, loop, audio_track, wav_path=None, sample_rate=16000, chunk_samples=320, data: np.ndarray | None = None):
        super().__init__(daemon=True)
        self.loop = loop
        self.audio_track = audio_track
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.chunk = chunk_samples
        self._quit = threading.Event()
        self._data = data  # if provided, use this preloaded mono float32 16kHz audio
        # buffering for multiple uploads
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._buffer_list: list[np.ndarray] = []  # list of float32 mono 16k arrays

    def stop(self):
        self._quit.set()

    def run(self):
        # If initial data provided, queue it
        if self._data is not None and isinstance(self._data, np.ndarray) and self._data.size > 0:
            with self._cond:
                self._buffer_list.append(self._data.astype(np.float32))
                self._data = None
                self._cond.notify_all()

        ptime = self.chunk / float(self.sample_rate)  # 20ms pacing
        start = time.time()
        sent_chunks = 0
        cur: np.ndarray | None = None
        idx = 0
        while not self._quit.is_set():
            if cur is None or idx >= len(cur):
                # fetch next buffer or wait
                with self._cond:
                    while not self._quit.is_set() and len(self._buffer_list) == 0:
                        self._cond.wait(timeout=0.1)
                    if self._quit.is_set():
                        break
                    cur = self._buffer_list.pop(0)
                    idx = 0
                    start = time.time()
                    sent_chunks = 0
            # send one chunk (with tail padding inside the buffer)
            end = min(idx + self.chunk, len(cur))
            chunk = cur[idx:end]
            if len(chunk) < self.chunk:
                pad = np.zeros((self.chunk - len(chunk),), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], 0)
            s16 = (chunk * 32767.0).astype(np.int16)
            af = AudioFrame(format='s16', layout='mono', samples=self.chunk)
            af.planes[0].update(s16.tobytes())
            af.sample_rate = self.sample_rate
            asyncio.run_coroutine_threadsafe(self.audio_track._queue.put((af, None)), self.loop)
            idx += self.chunk
            sent_chunks += 1
            target_time = start + sent_chunks * ptime
            delay = target_time - time.time()
            if delay > 0:
                time.sleep(delay)
        # On quit, signal EOS once
        try:
            asyncio.run_coroutine_threadsafe(self.audio_track._queue.put((None, 'eof_audio')), self.loop)
        except Exception:
            pass

    def append_audio(self, data: np.ndarray, sr: int):
        # append new audio segment (any time). Converts to mono float32 16k
        x = np.asarray(data)
        if x.ndim > 1:
            x = x[:, 0]
        if sr != self.sample_rate and x.size > 0:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)
        x = x.astype(np.float32)
        with self._cond:
            self._buffer_list.append(x)
            self._cond.notify_all()


class DittoReal(BaseReal):
    def __init__(self, opt, model=None, avatar=None):
        # We reuse BaseReal for consistent webrtc plumbing, but disable its TTS/ASR
        super().__init__(opt)
        self.tts = _TTSEmpty()
        self.asr = _ASREmpty()
        # placeholders for sdk and threads
        self._sdk = None
        self._audio_pusher = None
        self._source_path = None  # uploaded source path per session
        self._audio_path = None   # uploaded audio path per session

    def paste_back_frame(self, pred_frame, idx: int):
        # Not used in this adapter; frames are composed inside StreamSDK
        return pred_frame

    # allow runtime setting of source path (image/video) per session
    def set_source_path(self, path: str):
        self._source_path = path

    # allow runtime setting of audio path per session
    def set_audio_path(self, path: str):
        self._audio_path = path
        # If SDK is already running, push this newly uploaded audio into scheduler immediately
        try:
            sdk = getattr(self, '_sdk', None)
            if sdk is not None and hasattr(sdk, 'push_audio_file_bytes'):
                with open(path, 'rb') as f:
                    file_bytes = f.read()
                sdk.push_audio_file_bytes(file_bytes)
            # Also append to webrtc audio pusher for audible playback
            if self._audio_pusher is not None:
                try:
                    data, sr = sf.read(path, dtype='float32')
                    self._audio_pusher.append_audio(data, sr)
                except Exception:
                    pass
        except Exception:
            # non-fatal; initial render loop will still consume _audio_path if needed
            pass

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        # 1) Build StreamSDK from opt
        # validate required options
        required = ['ditto_cfg_pkl', 'ditto_data_root', 'ditto_audio']
        for k in required:
            if not hasattr(self.opt, k):
                raise RuntimeError(f"Missing required option: {k}. Please start app.py with --{k}.")
        if not self.opt.ditto_cfg_pkl or not self.opt.ditto_data_root:
            raise RuntimeError("Ditto cfg/data_root not set. Provide --ditto_cfg_pkl and --ditto_data_root.")
        # decide source path: uploaded first, else CLI; if missing, wait for frontend upload
        source_path = self._source_path or getattr(self.opt, 'ditto_source', '')
        if not source_path:
            t0 = time.time()
            while not quit_event.is_set() and (time.time() - t0) < 60.0 and not self._source_path:
                time.sleep(0.2)
            source_path = self._source_path or getattr(self.opt, 'ditto_source', '')
            if not source_path:
                raise RuntimeError("Source is empty. Upload a source file or start with --ditto_source.")
        # decide audio path: uploaded first, else CLI; if missing, wait for frontend upload
        audio_path = self._audio_path or getattr(self.opt, 'ditto_audio', '')
        if not audio_path:
            t0 = time.time()
            while not quit_event.is_set() and (time.time() - t0) < 60.0 and not self._audio_path:
                time.sleep(0.2)
            audio_path = self._audio_path or getattr(self.opt, 'ditto_audio', '')
            if not audio_path:
                raise RuntimeError("Audio is empty. Upload an audio file or start with --ditto_audio.")
        sdk = StreamSDK(self.opt.ditto_cfg_pkl, self.opt.ditto_data_root)
        self._sdk = sdk

        # 2) Determine N_d from audio length at 25fps (use the same audio array for both SDK and pusher)
        audio, sr = librosa.core.load(audio_path, sr=16000)
        # ensure mono float32 ndarray at 16k
        if audio.ndim > 1:
            audio = audio[0, :]
        # Zero-pad to a whole number of 20ms chunks to align features with playout tail
        rem = len(audio) % 320
        if rem != 0:
            audio = np.concatenate([audio, np.zeros(320 - rem, dtype=audio.dtype)], axis=0)
        # Compute N_d based on actual 20ms audio chunks to cover the same playout duration
        n_chunks = int(len(audio) / 320)  # exact integer after padding
        n_frames = int(np.ceil(n_chunks / 2.0))      # 40ms per video frame (~25fps)

        # 3) Setup SDK (always offline pipeline). Inject WebRTC writer here.
        # Do not enable tail padding: expected_frames=None disables close-time padding
        webrtc_writer = WebRTCVideoWriter(loop, video_track, expected_frames=None)
        sdk.setup(source_path, "unused_output_path_from_webrtc", writer=webrtc_writer)
        sdk.setup_Nd(N_d=n_frames, fade_in=-1, fade_out=-1, ctrl_info={})

        # 4) Start audio pusher to webrtc, queue initial audio; later uploads will be appended
        self._audio_pusher = AudioPusher(loop, audio_track, audio_path, data=audio)
        self._audio_pusher.start()

        # 5) Feed audio to StreamSDK scheduler as bytes; do NOT signal EOF.
        # This allows continuous video with silence after audio ends, and supports multiple uploads.
        try:
            with open(audio_path, 'rb') as f:
                file_bytes = f.read()
            sdk.push_audio_file_bytes(file_bytes)
        except Exception:
            # Fallback: push from in-memory ndarray if file reading fails
            sdk.push_audio_chunk(audio, sr=16000)

        # 6) Wait until quit_event is set, then close everything
        try:
            while not quit_event.is_set():
                time.sleep(0.1)
        finally:
            # signal pipeline to flush
            try:
                sdk.close()
            except Exception:
                pass
            # stop audio pusher
            if self._audio_pusher:
                self._audio_pusher.stop()
                self._audio_pusher.join()

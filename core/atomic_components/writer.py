import imageio
import os
import subprocess
import numpy as np
import threading


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        video_format = kwargs.get("format", "mp4")  # default is mp4 format
        codec = kwargs.get("vcodec", "libx264")  # default is libx264 encoding
        quality = kwargs.get("quality")  # video quality
        pixelformat = kwargs.get("pixelformat", "yuv420p")  # video pixel format
        macro_block_size = kwargs.get("macro_block_size", 2)
        ffmpeg_params = ["-crf", str(kwargs.get("crf", 18))]

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            format=video_format,
            codec=codec,
            quality=quality,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixelformat,
            macro_block_size=macro_block_size,
        )
        self.writer = writer

    def __call__(self, img, fmt="bgr"):
        if fmt == "bgr":
            frame = img[..., ::-1]
        else:
            frame = img
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


class RTMPStreamWriter:
    """
    RTMP 推流器类，使用 ffmpeg 将视频帧和音频推送到 RTMP 服务器
    通过 subprocess 调用 ffmpeg，并通过 stdin 推送帧数据
    支持同时推送视频和音频流
    """
    def __init__(self, rtmp_url, fps=25, width=None, height=None, 
                 audio_path=None, audio_sr=16000, use_silent_audio=False, **kwargs):
        """
        初始化 RTMP 推流器
        
        参数:
            rtmp_url: RTMP 服务器地址，例如 "rtmp://localhost:1935/live/stream"
            fps: 视频帧率，默认 25
            width: 视频宽度（会在第一帧时自动检测）
            height: 视频高度（会在第一帧时自动检测）
            audio_path: 音频文件路径（可选，如果提供则从文件读取音频并推流）
            audio_sr: 音频采样率，默认 16000
            use_silent_audio: 如果为 True，使用 ffmpeg 的 anullsrc 生成静音音频流（无限长度）
            **kwargs: 其他参数（预留）
        """
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.audio_path = audio_path
        self.audio_sr = audio_sr
        self.use_silent_audio = use_silent_audio  # 是否使用静音音频流
        self.ffmpeg_process = None
        self.lock = threading.Lock()
        self.is_started = False
        
    def _start_ffmpeg(self, width, height):
        """
        启动 ffmpeg 进程用于 RTMP 推流（支持音视频混合推流）
        
        参数:
            width: 视频宽度
            height: 视频高度
        """
        # 构建 ffmpeg 命令
        # 根据是否有音频文件或使用静音音频流选择不同的命令
        if self.use_silent_audio:
            # 方案C：使用 ffmpeg 的 anullsrc 生成静音音频流（无限长度，适合循环推流）
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                # 视频输入：从 stdin 读取原始视频
                '-f', 'rawvideo',  # 输入格式为原始视频
                '-vcodec', 'rawvideo',  # 输入编码器
                '-s', f'{width}x{height}',  # 视频尺寸
                '-pix_fmt', 'rgb24',  # 像素格式（RGB，每通道8位）
                '-r', str(self.fps),  # 输入帧率
                '-i', '-',  # 从 stdin 读取视频
                # 音频输入：使用 anullsrc 生成静音音频流
                '-f', 'lavfi',  # 使用 libavfilter 输入
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # 生成静音音频流
                # 输出设置
                '-c:v', 'libx264',  # 视频编码器
                '-pix_fmt', 'yuv420p',  # 输出像素格式
                '-preset', 'ultrafast',  # 编码预设（最快，适合实时推流）
                '-tune', 'zerolatency',  # 零延迟调优
                '-c:a', 'aac',  # 音频编码器
                '-b:a', '128k',  # 音频比特率
                '-ar', '44100',  # 音频采样率
                '-ac', '2',  # 音频声道数（立体声）
                # 流映射
                '-map', '0:v:0',  # 映射视频流
                '-map', '1:a:0',  # 映射音频流
                # 音视频同步设置
                '-async', '1',  # 音频同步模式
                '-vsync', 'cfr',  # 视频同步模式（cfr=恒定帧率）
                # 不使用 -shortest，让视频决定时长（静音音频是无限的）
                '-f', 'flv',  # 输出格式为 FLV（RTMP 常用格式）
                self.rtmp_url  # RTMP 服务器地址
            ]
            audio_info = "（静音音频流）"
        elif self.audio_path and os.path.exists(self.audio_path):
            # 方案A：有音频文件时，同时推送视频和音频
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                # 视频输入：从 stdin 读取原始视频
                '-f', 'rawvideo',  # 输入格式为原始视频
                '-vcodec', 'rawvideo',  # 输入编码器
                '-s', f'{width}x{height}',  # 视频尺寸
                '-pix_fmt', 'rgb24',  # 像素格式（RGB，每通道8位）
                '-r', str(self.fps),  # 输入帧率
                '-i', '-',  # 从 stdin 读取视频
                # 音频输入：从文件读取（不循环）
                '-stream_loop', '0',  # 不循环音频文件（0=不循环，防止音频重复）
                '-i', self.audio_path,  # 音频文件路径
                # 输出设置
                '-c:v', 'libx264',  # 视频编码器
                '-pix_fmt', 'yuv420p',  # 输出像素格式
                '-preset', 'ultrafast',  # 编码预设（最快，适合实时推流）
                '-tune', 'zerolatency',  # 零延迟调优
                # 音频编码设置
                '-c:a', 'aac',  # 音频编码器
                '-b:a', '128k',  # 音频比特率
                '-ar', '44100',  # 音频采样率（FLV/RTMP标准采样率，ffmpeg会自动转换）
                '-ac', '2',  # 音频声道数（立体声）
                # 流映射：明确指定使用哪个视频流和音频流
                '-map', '0:v:0',  # 映射第一个输入（视频）的视频流
                '-map', '1:a:0',  # 映射第二个输入（音频）的音频流
                # 音视频同步设置
                # 不使用 -async，避免音频被拉伸或重复
                '-vsync', 'cfr',  # 视频同步模式（cfr=恒定帧率）
                # 使用 -shortest 确保音视频同步结束
                # 如果音频比视频短，ffmpeg会在音频结束后立即停止（防止音频重复）
                # 这是关键参数，确保音频不会循环播放
                '-shortest',  # 以最短的流为准（确保音视频同步结束）
                # 额外保护：使用 -avoid_negative_ts make_zero 确保时间戳正确
                '-avoid_negative_ts', 'make_zero',  # 避免负时间戳，确保音视频同步
                '-f', 'flv',  # 输出格式为 FLV（RTMP 常用格式）
                self.rtmp_url  # RTMP 服务器地址
            ]
            audio_info = f", 音频: {os.path.basename(self.audio_path)}"
            # 打印完整的 ffmpeg 命令用于调试
            print(f"[调试] FFmpeg 命令: {' '.join(ffmpeg_cmd)}")
            print(f"[调试] 音频文件路径: {self.audio_path}, 存在: {os.path.exists(self.audio_path)}")
        else:
            # 方案B：无音频文件时，仅推送视频（保持原有功能）
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',  # 输入格式为原始视频
                '-vcodec', 'rawvideo',  # 输入编码器
                '-s', f'{width}x{height}',  # 视频尺寸
                '-pix_fmt', 'rgb24',  # 像素格式（RGB，每通道8位）
                '-r', str(self.fps),  # 输入帧率
                '-i', '-',  # 从 stdin 读取
                '-c:v', 'libx264',  # 输出编码器
                '-pix_fmt', 'yuv420p',  # 输出像素格式
                '-preset', 'ultrafast',  # 编码预设（最快，适合实时推流）
                '-tune', 'zerolatency',  # 零延迟调优
                '-f', 'flv',  # 输出格式为 FLV（RTMP 常用格式）
                self.rtmp_url  # RTMP 服务器地址
            ]
            audio_info = "（仅视频）"
        
        # 启动 ffmpeg 进程
        # stdin=subprocess.PIPE 用于向进程写入帧数据
        # stderr=subprocess.PIPE 用于捕获错误信息
        # 注意：stderr 包含 ffmpeg 的日志输出，包括音频处理信息
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # 启动一个线程来监控 ffmpeg 的 stderr 输出（用于调试）
        def monitor_ffmpeg_stderr():
            """监控 ffmpeg 的 stderr 输出，用于调试"""
            try:
                while self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    line = self.ffmpeg_process.stderr.readline()
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        # 只打印重要的信息（包含 audio 或 error 的行）
                        if 'audio' in line_str.lower() or 'error' in line_str.lower() or 'stream' in line_str.lower():
                            print(f"[FFmpeg] {line_str}")
            except Exception as e:
                pass  # 忽略监控线程的错误
        
        stderr_monitor = threading.Thread(target=monitor_ffmpeg_stderr, daemon=True)
        stderr_monitor.start()
        
        self.width = width
        self.height = height
        self.is_started = True
        print(f"RTMP 推流已启动: {self.rtmp_url}, 分辨率: {width}x{height}, 帧率: {self.fps}{audio_info}")
    
    def __call__(self, img, fmt="rgb"):
        """
        推送一帧到 RTMP 流
        
        参数:
            img: 图像数据（numpy array），格式为 RGB 或 BGR
            fmt: 图像格式，"rgb" 或 "bgr"，默认为 "rgb"
        """
        with self.lock:
            # 确保图像是 numpy array 类型
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # 转换为 RGB 格式（如果输入是 BGR）
            if fmt == "bgr":
                frame = img[..., ::-1]  # BGR 转 RGB
            else:
                frame = img  # 已经是 RGB
            
            # 获取图像尺寸
            height, width = frame.shape[:2]
            
            # 如果还没有启动 ffmpeg 进程，现在启动
            if not self.is_started:
                self._start_ffmpeg(width, height)
            else:
                # 确保图像尺寸匹配（仅在已启动后检查）
                if width != self.width or height != self.height:
                    raise ValueError(
                        f"图像尺寸不匹配: 期望 {self.width}x{self.height}, "
                        f"实际 {width}x{height}"
                    )
            
            # 确保图像数据类型为 uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            # 确保图像是连续的（连续的数组写入更快）
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # 将帧写入 ffmpeg 的 stdin
            # 首先检查 ffmpeg 进程是否还在运行
            if self.ffmpeg_process is None:
                # ffmpeg 进程未初始化，直接返回
                return
            
            # 检查进程是否已经结束
            poll_result = self.ffmpeg_process.poll()
            if poll_result is not None:
                # ffmpeg 进程已经结束（可能是因为 -shortest 参数，音频结束后自动关闭）
                # 这是正常情况，不需要抛出异常
                # 标记为已停止，避免后续写入尝试
                self.is_started = False
                return
            
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # 如果管道断开或写入失败，说明 ffmpeg 进程可能已经结束
                # 这可能是因为使用了 -shortest 参数，音频流结束后 ffmpeg 自动关闭
                # 再次检查进程状态
                poll_result = self.ffmpeg_process.poll()
                if poll_result is not None:
                    # 进程已经结束（无论返回码是什么，都可能是正常结束）
                    # 使用 -shortest 参数时，ffmpeg 会在较短的流结束后正常退出
                    # 这是预期行为，不需要抛出异常
                    self.is_started = False
                    return
                else:
                    # 进程还在运行但写入失败，这可能是真正的错误
                    # 但为了稳定性，我们也静默处理，避免程序崩溃
                    # 可以记录日志但不抛出异常
                    print(f"[警告] 写入 FFmpeg 失败，但进程仍在运行: {e}")
                    self.is_started = False
                    return
    
    def close(self):
        """
        关闭 RTMP 推流器
        关闭 ffmpeg 进程的 stdin，等待进程结束
        """
        with self.lock:
            if self.ffmpeg_process is not None:
                try:
                    # 检查进程是否还在运行
                    poll_result = self.ffmpeg_process.poll()
                    if poll_result is None:
                        # 进程还在运行，关闭 stdin，ffmpeg 会处理完剩余数据后退出
                        try:
                            if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                                self.ffmpeg_process.stdin.close()
                        except (OSError, BrokenPipeError, AttributeError):
                            # stdin 可能已经关闭或不存在，忽略错误
                            pass
                        # 等待进程结束（最多等待 5 秒）
                        try:
                            self.ffmpeg_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # 如果超时，强制终止进程
                            try:
                                self.ffmpeg_process.kill()
                                self.ffmpeg_process.wait()
                            except Exception:
                                pass
                    else:
                        # 进程已经结束（可能是因为 -shortest 参数）
                        # 这是正常情况，不需要打印警告
                        pass
                except Exception as e:
                    # 关闭过程中的错误不应该影响程序继续运行
                    print(f"[警告] 关闭 RTMP 推流器时出错: {e}")
                finally:
                    self.ffmpeg_process = None
                    self.is_started = False
                    print("RTMP 推流已关闭")

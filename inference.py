import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_online import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        # 在线模式：使用 RTMP 推流，不需要音频和视频合成
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        # 离线模式：写入文件，后续需要音频和视频合成
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # 只有在离线模式下才进行音频和视频合成
    if not online_mode:
        # 离线模式：将视频文件和音频文件合成
        if SDK.tmp_output_path and os.path.exists(SDK.tmp_output_path):
            cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
            print(cmd)
            os.system(cmd)
            print(f"视频已保存到: {output_path}")
        else:
            print(f"警告: 临时视频文件不存在: {SDK.tmp_output_path}")
    else:
        # 在线模式：直接推流，不需要合成
        print("在线模式：视频已推送到 RTMP 流，无需合成文件")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    
    # 在线模式相关参数
    parser.add_argument("--online_mode", action="store_true", help="启用在线模式（RTMP 推流）")
    parser.add_argument("--rtmp_url", type=str, default="rtmp://localhost:1935/live/stream", help="RTMP 服务器地址（仅在在线模式下使用）")
    
    # 配置文件参数（可选）
    parser.add_argument("--more_kwargs", type=str, default=None, help="配置文件路径（pkl格式），包含 setup_kwargs 和 run_kwargs")
    
    args = parser.parse_args()

    # init sdk
    data_root = args.data_root   # model dir
    cfg_pkl = args.cfg_pkl     # cfg pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # 准备 more_kwargs
    if args.more_kwargs:
        # 如果提供了配置文件，加载它
        more_kwargs = load_pkl(args.more_kwargs)
    else:
        # 如果没有配置文件，使用默认值或命令行参数
        more_kwargs = {}
    
    # 设置 setup_kwargs（如果命令行参数提供了 online_mode 或 rtmp_url，会覆盖配置文件中的设置）
    if "setup_kwargs" not in more_kwargs:
        more_kwargs["setup_kwargs"] = {}
    
    # 命令行参数优先级高于配置文件
    if args.online_mode:
        more_kwargs["setup_kwargs"]["online_mode"] = True
        more_kwargs["setup_kwargs"]["rtmp_url"] = args.rtmp_url
    elif "online_mode" not in more_kwargs["setup_kwargs"]:
        # 如果命令行没有指定，且配置文件中也没有，则默认为 False（离线模式）
        more_kwargs["setup_kwargs"]["online_mode"] = False
    
    # 如果没有设置 run_kwargs，使用空字典
    if "run_kwargs" not in more_kwargs:
        more_kwargs["run_kwargs"] = {}

    # run
    # seed_everything(1024)
    run(SDK, audio_path, source_path, output_path, more_kwargs)

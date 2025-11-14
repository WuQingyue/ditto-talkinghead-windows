
import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import signal
import sys
import tempfile
import soundfile as sf

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


# 全局变量，用于控制循环推流
_streaming_should_stop = False

def _signal_handler(signum, frame):
    """处理停止信号（Ctrl+C）"""
    global _streaming_should_stop
    print("\n[信息] 收到停止信号，正在优雅关闭...")
    _streaming_should_stop = True


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, 
        more_kwargs: str | dict = {}, silence_path: str = None):

    global _streaming_should_stop
    _streaming_should_stop = False

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    online_mode = setup_kwargs.get("online_mode", False)
    
    # 如果是在线模式且指定了 silence_path，注册信号处理器（用于优雅停止）
    if online_mode and silence_path:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    # 先加载音频，以便在在线模式下创建包含静音前缀和填充静音的临时文件
    audio, sr = librosa.core.load(audio_path, sr=16000)
    
    # 临时音频文件路径（用于在线模式，包含静音前缀和最后一块的填充静音）
    temp_audio_path = None
    
    # 精确计算视频帧数：考虑静音前缀和最后一块的填充静音（在线模式）
    if online_mode:
        # 在线模式：先获取 chunksize，计算静音前缀长度
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        silence_prefix_len = chunksize[0] * 640  # 静音前缀的采样点数
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        
        # 计算最后一块的填充长度
        # 添加静音前缀后的音频长度
        audio_with_prefix = np.concatenate([
            np.zeros((silence_prefix_len,), dtype=np.float32), 
            audio
        ], axis=0)
        
        # 计算最后一块的实际长度和填充长度
        chunk_step = chunksize[1] * 640  # 每次循环处理的采样点数（步长）
        # 计算循环中最后一个 i 值
        last_i = ((len(audio_with_prefix) - 1) // chunk_step) * chunk_step
        # 计算最后一块的实际长度（从 last_i 开始到音频末尾）
        last_chunk_actual_len = len(audio_with_prefix) - last_i
        
        # 如果最后一块不足 split_len，需要填充
        # 在循环中，我们取 audio_with_prefix[i:i + split_len]
        # 如果 last_i + split_len > len(audio_with_prefix)，需要填充
        last_chunk_padding_len = 0
        if last_i + split_len > len(audio_with_prefix):
            last_chunk_padding_len = (last_i + split_len) - len(audio_with_prefix)
        
        # 计算包含静音前缀和填充静音的总音频长度
        total_audio_len = len(audio_with_prefix) + last_chunk_padding_len
        
        # 创建包含静音前缀和填充静音的临时音频文件，用于 RTMP 推流
        audio_for_streaming = np.concatenate([
            audio_with_prefix,
            np.zeros((last_chunk_padding_len,), dtype=np.float32)  # 最后一块的填充静音
        ], axis=0)
        
        # 创建临时音频文件
        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav', prefix='ditto_audio_')
        os.close(temp_audio_fd)  # 关闭文件描述符，soundfile 会重新打开
        sf.write(temp_audio_path, audio_for_streaming, 16000)
        
        # 更新 setup_kwargs 中的音频路径为包含静音前缀和填充静音的临时文件
        setup_kwargs["audio_path"] = temp_audio_path
        setup_kwargs["audio_sr"] = 16000  # 传递音频采样率
        
        # 基于总长度计算帧数（25fps）
        num_f = math.ceil(total_audio_len / 16000 * 25)
        print(f"[调试] 原始音频长度: {len(audio)} 采样点 ({len(audio)/16000:.2f}秒)")
        print(f"[调试] 静音前缀长度: {silence_prefix_len} 采样点 ({silence_prefix_len/16000:.2f}秒)")
        print(f"[调试] 最后一块填充长度: {last_chunk_padding_len} 采样点 ({last_chunk_padding_len/16000:.2f}秒)")
        print(f"[调试] 总音频长度（包含前缀和填充）: {total_audio_len} 采样点 ({total_audio_len/16000:.2f}秒)")
        print(f"[调试] 计算的视频帧数: {num_f} 帧 ({num_f/25:.2f}秒)")
        print(f"[调试] 临时音频文件已创建: {temp_audio_path}")
    else:
        # 离线模式：直接基于原始音频长度计算帧数
        num_f = math.ceil(len(audio) / 16000 * 25)
    
    # 现在调用 SDK.setup，此时 setup_kwargs 中的 audio_path 已经是正确的（在线模式是临时文件，离线模式不需要）
    SDK.setup(source_path, output_path, **setup_kwargs)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    if online_mode:
        # 在线模式：使用 RTMP 推流，不需要音频和视频合成
        # 注意：audio 变量此时仍然是原始音频（不包含前缀）
        # 我们需要使用包含前缀的音频进行处理
        # audio_with_prefix 已经在上面计算过了，但为了代码清晰，我们重新创建
        audio_with_prefix = np.concatenate([
            np.zeros((chunksize[0] * 640,), dtype=np.float32), 
            audio
        ], axis=0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        
        # 阶段1：处理 audio.wav（使用包含静音前缀的音频）
        print("[阶段1] 开始处理 audio.wav...")
        for i in range(0, len(audio_with_prefix), chunksize[1] * 640):
            if _streaming_should_stop:
                break
            audio_chunk = audio_with_prefix[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
        
        print("[阶段1] audio.wav 处理完成")
        
        # 阶段2：如果指定了 silence.wav，则处理一次 silence.wav
        if silence_path and os.path.exists(silence_path) and not _streaming_should_stop:
            # 关闭阶段1的推流
            SDK.close()
            
            print("[阶段2] 切换到静音音频流，开始处理 silence.wav...")
            # 重新设置，使用静音音频流
            setup_kwargs["audio_path"] = None  # 不使用音频文件
            setup_kwargs["use_silent_audio"] = True  # 使用静音音频流
            
            # 重新初始化 SDK（阶段2）
            SDK.setup(source_path, output_path, **setup_kwargs)
            
            # 加载静音音频
            silence_audio, sr = librosa.core.load(silence_path, sr=16000)
            silence_audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), silence_audio], 0)
            num_f_silence = math.ceil(len(silence_audio) / 16000 * 25)
            
            # 设置帧数
            SDK.setup_Nd(N_d=num_f_silence, fade_in=-1, fade_out=-1, ctrl_info={})
            
            # 处理 silence.wav 的所有音频块（只处理一次，不循环）
            for i in range(0, len(silence_audio), chunksize[1] * 640):
                if _streaming_should_stop:
                    break
                audio_chunk = silence_audio[i:i + split_len]
                if len(audio_chunk) < split_len:
                    audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
                SDK.run_chunk(audio_chunk, chunksize)
            
            print("[阶段2] silence.wav 处理完成")
    else:
        # 离线模式：写入文件，后续需要音频和视频合成
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    SDK.close()

    # 清理临时音频文件（如果存在）
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            print(f"[调试] 临时音频文件已删除: {temp_audio_path}")
        except Exception as e:
            print(f"[警告] 删除临时音频文件失败: {e}")

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
    parser.add_argument("--silence_path", type=str, default=None, help="静音音频文件路径（silence.wav），audio.wav 处理完后会处理一次 silence.wav")
    
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

    # 检查 silence_path 参数
    if args.silence_path:
        if not args.online_mode:
            print("[错误] --silence_path 只能在 --online_mode 下使用")
            sys.exit(1)
        if not os.path.exists(args.silence_path):
            print(f"[错误] 静音音频文件不存在: {args.silence_path}")
            sys.exit(1)
        print("=" * 60)
        print("两阶段推流模式已启用")
        print("=" * 60)
        print(f"阶段1音频: {audio_path}")
        print(f"阶段2音频: {args.silence_path}")
        print(f"RTMP 地址: {more_kwargs['setup_kwargs'].get('rtmp_url', args.rtmp_url)}")
        print("=" * 60)
    
    # run
    # seed_everything(1024)
    try:
        run(SDK, audio_path, source_path, output_path, more_kwargs, 
            silence_path=args.silence_path)
    except KeyboardInterrupt:
        print("\n[信息] 收到键盘中断，正在停止...")
        _streaming_should_stop = True
        SDK.close()
    except Exception as e:
        print(f"[错误] 发生异常: {e}")
        import traceback
        traceback.print_exc()
        SDK.close()
        raise

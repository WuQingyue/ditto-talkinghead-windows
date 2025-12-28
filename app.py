import os as _os
_os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
#from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
# --- aiortc compatibility patch: guard missing private __encoder on RTCRtpSender ---
try:
    _orig_handle_rtcp_packet = RTCRtpSender._handle_rtcp_packet
    async def _safe_handle_rtcp_packet(self, packet):
        if not hasattr(self, "_RTCRtpSender__encoder"):
            # initialize missing private attribute to avoid AttributeError in older aiortc
            setattr(self, "_RTCRtpSender__encoder", None)
        return await _orig_handle_rtcp_packet(self, packet)
    RTCRtpSender._handle_rtcp_packet = _safe_handle_rtcp_packet
except Exception:
    pass
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
import gc
import os

app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    from ditto import DittoReal
    nerfreal = DittoReal(opt, None, opt.avatar_id)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # if len(nerfreals) >= opt.max_session:
    #     logger.info('reach max session')
    #     return web.Response(
    #         content_type="application/json",
    #         text=json.dumps(
    #             {"code": -1, "msg": "reach max session"}
    #         ),
    #     )
    sessionid = randN(6) #len(nerfreals)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            # cleanup nerfreals and uploaded files for this session
            try:
                if sessionid in nerfreals:
                    del nerfreals[sessionid]
                upload_dir = os.path.join('data', 'uploads', str(sessionid))
                if os.path.isdir(upload_dir):
                    shutil.rmtree(upload_dir, ignore_errors=True)
            except Exception:
                logger.exception('failed to cleanup session resources')
        if pc.connectionState == "closed":
            pcs.discard(pc)
            # cleanup nerfreals and uploaded files for this session
            try:
                if sessionid in nerfreals:
                    del nerfreals[sessionid]
                upload_dir = os.path.join('data', 'uploads', str(sessionid))
                if os.path.isdir(upload_dir):
                    shutil.rmtree(upload_dir, ignore_errors=True)
            except Exception:
                logger.exception('failed to cleanup session resources')
            # gc.collect()

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    # Workaround: some aiortc versions access private __encoder before it exists.
    # Pre-initialize to avoid AttributeError in older aiortc
    try:
        if not hasattr(audio_sender, "_RTCRtpSender__encoder"):
            setattr(audio_sender, "_RTCRtpSender__encoder", None)
        if not hasattr(video_sender, "_RTCRtpSender__encoder"):
            setattr(video_sender, "_RTCRtpSender__encoder", None)
    except Exception:
        pass

    await pc.setRemoteDescription(offer)

    # Create answer then munge SDP to enforce higher target bitrate for video
    answer = await pc.createAnswer()

    def _munge_video_bitrate(sdp: str, max_kbps: int = 3000, min_kbps: int = 1800) -> str:
        lines = sdp.split("\r\n")
        out = []
        in_video = False
        video_payloads = set()
        # First pass: collect payload types for video codecs (H264/VP8), and insert b=AS
        for i, ln in enumerate(lines):
            if ln.startswith("m="):
                in_video = ln.startswith("m=video")
            if in_video and ln.startswith("m=video"):
                # ensure a bandwidth line b=AS:max_kbps after media header if not present later
                out.append(ln)
                # We will add a b=AS line immediately after if not already added in previous media lines
                # But defer decision until we see next non-empty line; add here for simplicity
                out.append(f"b=AS:{max_kbps}")
                continue
            # Collect rtpmap payloads mapping to H264/VP8
            if in_video and ln.startswith("a=rtpmap:"):
                try:
                    pt, rest = ln[len("a=rtpmap:"):].split(" ", 1)
                    codec = rest.split("/")[0].upper()
                    if codec in ("H264", "VP8"):
                        video_payloads.add(pt)
                except Exception:
                    pass
            out.append(ln)

        # Second pass: add/patch fmtp for those payloads with x-google-*
        sdp2 = []
        for ln in out:
            if ln.startswith("a=fmtp:"):
                try:
                    head, params = ln.split(" ", 1)
                    pt = head[len("a=fmtp:"):]
                    if pt in video_payloads:
                        kv = params
                        # append or update x-google-max-bitrate/min-bitrate (in kbps)
                        # Normalize separators
                        if kv and not kv.endswith(";"):
                            kv = kv + ";"
                        kv = kv + f"x-google-max-bitrate={max_kbps};x-google-min-bitrate={min_kbps}"
                        ln = f"a=fmtp:{pt} {kv}"
                except Exception:
                    pass
            sdp2.append(ln)
        return "\r\n".join(sdp2)

    munged = _munge_video_bitrate(answer.sdp, max_kbps=3000, min_kbps=1800)
    answer = RTCSessionDescription(sdp=munged, type=answer.type)
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type']=='chat':
            asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])                         
            #nerfreals[sessionid].put_msg_txt(res)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def humanaudio(request):
    try:
        logger.info("[humanaudio] 收到请求: method=%s, content_type=%s", request.method, request.content_type)
        logger.info("[humanaudio] 请求头部部分信息: %s", dict(list(request.headers.items())[:10]))

        form = await request.post()
        logger.info("[humanaudio] 解析到的表单字段 keys: %s", list(form.keys()))

        raw_sessionid = form.get('sessionid', 0)
        try:
            sessionid = int(raw_sessionid)
        except (TypeError, ValueError):
            logger.info("[humanaudio] sessionid 无法转换为 int, 原始值=%r", raw_sessionid)
            sessionid = 0

        logger.info("[humanaudio] 解析到的 sessionid=%s, 是否在 nerfreals 中: %s", sessionid, sessionid in nerfreals)

        upload_path = None
        upload_dir = None
        if 'file' in form:
            fileobj = form["file"]
            filename = getattr(fileobj, 'filename', None)
            file_stream = getattr(fileobj, 'file', None)
            reported_size = getattr(fileobj, 'size', None)
            logger.info("[humanaudio] 收到文件字段, filename=%s, fileobj_type=%s, file_stream_type=%s, reported_size=%s",
                        filename, type(fileobj), type(file_stream), reported_size)

            # 将上传的音频文件落盘到 uploads/<sessionid>/ 目录
            if file_stream is not None and sessionid in nerfreals:
                try:
                    upload_dir = os.path.join('uploads', str(sessionid))
                    os.makedirs(upload_dir, exist_ok=True)
                    safe_name = filename or 'upload.wav'
                    upload_path = os.path.join(upload_dir, safe_name)
                    with open(upload_path, 'wb') as f:
                        while True:
                            chunk = file_stream.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                    logger.info("[humanaudio] 音频已保存到: %s", upload_path)
                except Exception:
                    logger.exception("[humanaudio] 保存上传音频到磁盘时异常")

        interrupt_value = form.get('interrupt')
        logger.info("[humanaudio] 收到 interrupt 字段原始值=%r", interrupt_value)

        # Check if the value is the boolean True OR the string 'true' (case-insensitive)
        if interrupt_value is True or str(interrupt_value).lower() == 'true':
            logger.info("[humanaudio] interrupt 标志为真, 尝试打断当前说话 (sessionid=%s)", sessionid)
            if sessionid in nerfreals:
                try:
                    nerfreals[sessionid].flush_talk()
                    logger.info("[humanaudio] 已调用 nerfreals[%s].flush_talk()", sessionid)
                except Exception:
                    logger.exception("[humanaudio] 调用 flush_talk 时异常")
            else:
                logger.info("[humanaudio] sessionid=%s 不在 nerfreals 中, 无法 flush_talk", sessionid)

            # 中断模式下，删除该会话目录下除本次最新音频外的其它文件
            if upload_dir and upload_path:
                try:
                    for name in os.listdir(upload_dir):
                        full_path = os.path.join(upload_dir, name)
                        if full_path != upload_path and os.path.isfile(full_path):
                            try:
                                os.remove(full_path)
                            except Exception:
                                logger.exception("[humanaudio] 删除旧音频文件失败: %s", full_path)
                except Exception:
                    logger.exception("[humanaudio] 清理旧音频文件时异常, upload_dir=%s", upload_dir)

        # 在可能的 flush_talk() 之后再把最新音频推入 Ditto 模型，避免被清空
        if upload_path and sessionid in nerfreals:
            try:
                nerfreals[sessionid].set_audio_path(upload_path)
                logger.info("[humanaudio] 最终调用 nerfreals[%s].set_audio_path, path=%s", sessionid, upload_path)
            except Exception:
                logger.exception("[humanaudio] 调用 set_audio_path 时异常")

        logger.info("[humanaudio] 处理完成, 即将返回成功响应")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)    
        nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='ditto') #ditto wav2lip ultralight musetalk

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    # Ditto pipeline inputs for the 'ditto' model adapter
    parser.add_argument('--ditto_data_root', type=str, default='./checkpoints/ditto_trt_custom', help='Ditto data root (models)')
    parser.add_argument('--ditto_cfg_pkl', type=str, default='./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl', help='Ditto cfg pickle path')
    parser.add_argument('--ditto_source', type=str, default='', help='Ditto source image or video path')
    parser.add_argument('--ditto_audio', type=str, default='', help='Ditto audio wav path (16k preferred)')

    opt = parser.parse_args()

    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # Enforce Ditto-only model
    opt.model = 'ditto'
    # Ditto pipeline (StreamSDK) is created inside DittoReal on demand; no preload here.
    logger.info('Using Ditto (StreamSDK) pipeline; models will be initialized in DittoReal at runtime')

    # Optional: start a render thread only for virtual camera transport
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'

    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/webrtc-embed.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
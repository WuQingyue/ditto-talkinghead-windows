var pc = null;

// URL参数检测和嵌入模式支持
const urlParams = new URLSearchParams(window.location.search);
const IS_EMBED_MODE = urlParams.get('mode') === 'embed';
const parentOrigin = urlParams.get('parentOrigin');

console.log(`[iFrame] Running in ${IS_EMBED_MODE ? 'Embed Mode' : 'Standalone Mode'}.`);

if (IS_EMBED_MODE && !parentOrigin) {
    console.error('[iFrame] Embed mode requires a "parentOrigin" URL parameter.');
}

// Loading 显示/隐藏函数
function showLoading(message) {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    if (overlay) {
        if (text && message) {
            text.textContent = message;
        }
        overlay.style.display = 'flex'; // 使用 flex 来显示，以保证内容居中
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// 向父窗口发送消息
function sendMessageToParent(type, payload) {
    if (!IS_EMBED_MODE) {
        return;
    }
    if (!parentOrigin) {
        console.error("[iFrame] Cannot send message: parentOrigin is not set.");
        return;
    }
    console.log(`[iFrame] Sending message to parent:`, { type, payload });
    window.parent.postMessage({
        source: 'webrtc-iframe',
        type: type,
        payload: payload
    }, parentOrigin);
}

// 处理连接失败
function handleConnectionFailure(errorMessage) {
    console.error('CONNECTION FAILED:', errorMessage);
    // 在UI上显示错误信息
    showLoading(`连接失败: ${errorMessage}`);
    // 通知父页面
    sendMessageToParent('error', { message: errorMessage });
    // 5秒后隐藏错误信息，并重置UI状态
    setTimeout(() => {
        hideLoading();
        const stopBtn = document.getElementById('stop');
        const startBtn = document.getElementById('start');
        if (stopBtn) stopBtn.style.display = 'none';
        if (startBtn) startBtn.style.display = 'inline-block';
    }, 5000);
    // 关闭可能存在的peer connection
    if (pc) {
        pc.close();
        pc = null;
    }
}

// 监听父窗口消息（仅在嵌入模式下）
if (IS_EMBED_MODE) {
    window.addEventListener('message', (event) => {
        // 安全检查：确保消息来自预期的父窗口源
        if (event.origin !== parentOrigin) {
            console.warn(`[iFrame] Ignored message from untrusted origin: ${event.origin}`);
            return;
        }
        const command = event.data;
        if (!command || typeof command.type !== 'string') return;
        console.log(`[iFrame] Received command from parent:`, command);
        switch (command.type) {
            case 'START_WEBRTC':
                start();
                break;
            case 'DISCONNECT_WEBRTC':
                stop();
                break;
        }
    });
}

function negotiate() {
    showLoading('正在建立连接...');
    const negotiationTimeout = setTimeout(() => {
        handleConnectionFailure('连接超时，服务器未在规定时间内响应。');
    }, 30000);

    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        console.log('Sending offer to server...');
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        console.log('Received response from server:', response.status);
        if (!response.ok) { // 检查服务器响应状态
            throw new Error(`服务器错误: ${response.status} ${response.statusText}`);
        }
        return response.json();
    }).then((answer) => {
        console.log('Parsed response data:', answer);
        clearTimeout(negotiationTimeout);
        showLoading('正在接收视频流...');
        if (answer.sessionid) {
            document.getElementById('sessionid').value = answer.sessionid;
            sendMessageToParent('sessionidIsUpdated', { sessionId: answer.sessionid });
            // 使用我们在 webrtcapi.html 中定义的函数来更新 session ID 显示
            if (window.updateSessionDisplay) {
                console.log('Updating session display with ID:', answer.sessionid);
                window.updateSessionDisplay(answer.sessionid);
            }
        }
        return pc.setRemoteDescription(answer);
    }).then(() => {
        // wait for ICE connection to be established (connected or completed)
        return new Promise((resolve) => {
            const done = () => (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed');
            if (done()) {
                resolve();
                return;
            }
            const onState = () => {
                if (done()) {
                    pc.removeEventListener('iceconnectionstatechange', onState);
                    resolve();
                }
            };
            pc.addEventListener('iceconnectionstatechange', onState);
        });
    }).catch((e) => {
        clearTimeout(negotiationTimeout);
        sendMessageToParent('error', { message: e.message });
        handleConnectionFailure(e.message);
        throw e; // 重新抛出错误，让调用者知道失败
    });
}

function start() {
    showLoading('正在初始化...');
    sendMessageToParent('WEBRTC_NEGOTIATION_STARTED');

    var config = {
        sdpSemantics: 'unified-plan'
    };

    // Optional STUN: only apply if a toggle exists and is checked
    var stunToggle = document.getElementById('use-stun');
    if (stunToggle && stunToggle.checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    const videoElement = document.getElementById('video');
    const audioElement = document.getElementById('audio');
    
    // 确保 autoplay 策略满足浏览器要求
    if (videoElement) {
        videoElement.autoplay = true;
        videoElement.playsInline = true;
    }
    if (audioElement) {
        audioElement.autoplay = true;
    }

    // 视频播放时隐藏 loading
    const hideLoadingOnPlay = () => {
        hideLoading();
        videoElement.removeEventListener('playing', hideLoadingOnPlay);
    };
    videoElement.addEventListener('playing', hideLoadingOnPlay);

    // 处理音视频轨道
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            if (videoElement && evt.streams && evt.streams[0]) {
                videoElement.srcObject = evt.streams[0];
            }
        } else if (evt.track.kind == 'audio') {
            if (audioElement && evt.streams && evt.streams[0]) {
                audioElement.srcObject = evt.streams[0];
            }
        }
    });

    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    // upload controls
    const uploadAudioBtn = document.getElementById('btn_upload_audio');
    const uploadSourceBtn = document.getElementById('btn_upload_source');
    
    // Set Start to loading/disabled while negotiating and connecting
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        startBtn.style.display = 'none';
    }
    if (stopBtn) {
        stopBtn.style.display = 'inline-block';
    }
    // Disable upload buttons until WebRTC connected
    if (uploadAudioBtn) uploadAudioBtn.disabled = true;
    if (uploadSourceBtn) uploadSourceBtn.disabled = true;

    negotiate().then(() => {
        // Connected: hide Start, show Stop
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start';
            startBtn.style.display = 'none';
        }
        if (stopBtn) {
            stopBtn.style.display = 'inline-block';
        }
        // Re-enable upload buttons only after connection succeeds
        if (uploadAudioBtn) uploadAudioBtn.disabled = false;
        if (uploadSourceBtn) uploadSourceBtn.disabled = false;
        sendMessageToParent('WEBRTC_CONNECTED');
    }).catch((e) => {
        // Failure: restore Start button state, keep Stop hidden
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start';
            startBtn.style.display = 'inline-block';
        }
        if (stopBtn) {
            stopBtn.style.display = 'none';
        }
        // Keep upload buttons disabled until a successful connection per requirement
        if (uploadAudioBtn) uploadAudioBtn.disabled = true;
        if (uploadSourceBtn) uploadSourceBtn.disabled = true;
        // handleConnectionFailure 已经在 negotiate 中调用，这里不需要再次调用
    });
}

function stop() {
    const stopBtn = document.getElementById('stop');
    const startBtn = document.getElementById('start');
    if (stopBtn) stopBtn.style.display = 'none';
    if (startBtn) startBtn.style.display = 'inline-block';

    if (pc) {
        pc.close();
        pc = null;
    }
    
    sendMessageToParent('WEBRTC_DISCONNECTED');
    hideLoading();
}

// 处理页面关闭事件
window.addEventListener('DOMContentLoaded', () => {
    if (IS_EMBED_MODE) {
        console.log('[iFrame] DOM loaded. Reporting IFRAME_READY to parent.');
        sendMessageToParent('IFRAME_READY');
    }
});

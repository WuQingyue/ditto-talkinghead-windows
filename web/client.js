var pc = null;
var remoteStream = null;

function negotiate() {
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
        return response.json();
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid
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
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    // Optional STUN: only apply if a toggle exists and is checked
    var stunToggle = document.getElementById('use-stun');
    if (stunToggle && stunToggle.checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // Use a single MediaStream for both audio and video to keep A/V in sync
    remoteStream = new MediaStream();
    const videoEl = document.getElementById('video');
    // Ensure autoplay policies are met in most browsers
    if (videoEl) {
        videoEl.autoplay = true;
        videoEl.playsInline = true;
    }

    pc.addEventListener('track', (evt) => {
        // Some browsers may not populate evt.streams; always add the track explicitly
        remoteStream.addTrack(evt.track);
        if (videoEl && videoEl.srcObject !== remoteStream) {
            videoEl.srcObject = remoteStream;
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
    }
    if (stopBtn) {
        stopBtn.style.display = 'none';
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
    }).catch((e) => {
        // Failure: restore Start button state, keep Stop hidden
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start';
        }
        if (stopBtn) {
            stopBtn.style.display = 'none';
        }
        // Keep upload buttons disabled until a successful connection per requirement
        alert(e);
    });
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
        setTimeout(() => {
                pc.close();
            }, 500);
        e = e || window.event
        // 兼容IE8和Firefox 4之前的版本
        if (e) {
          e.returnValue = '关闭提示'
        }
        // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
        return '关闭提示'
      }
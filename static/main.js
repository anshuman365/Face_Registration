const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const statusBox  = document.getElementById('status');
const statusText = document.getElementById('status-text');
const overlay    = document.getElementById('scan-overlay');
const vault      = document.getElementById('vault');
const simWrap    = document.getElementById('sim-wrap');
const simBar     = document.getElementById('sim-bar');
const simValue   = document.getElementById('sim-value');

// Camera start
navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
  .then(stream => {
    video.srcObject = stream;
    setStatus('✅ Camera ready — Register karo ya Unlock karo.', 'ok');
  })
  .catch(err => {
    setStatus('❌ Camera error: ' + err.message, 'err');
  });

function setStatus(msg, type = 'ok') {
  statusText.textContent = msg;
  statusBox.className    = 'status-box';
  if (type === 'err')  statusBox.classList.add('err');
  if (type === 'info') statusBox.classList.add('info');
}

function setScanOverlay(state) {
  overlay.className = 'camera-overlay';
  if (state) overlay.classList.add(state);
}

function showSimilarity(val) {
  simWrap.style.display = 'block';
  const pct = Math.round(val * 100);
  simBar.style.width    = pct + '%';
  simBar.style.background = pct >= 90 ? '#a6e3a1' : pct >= 75 ? '#f9e2af' : '#f38ba8';
  simValue.textContent  = pct + '% match';
}

function captureFrame() {
  const ctx = canvas.getContext('2d');
  canvas.width  = video.videoWidth  || 320;
  canvas.height = video.videoHeight || 240;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
}

async function post(url, img) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ image: img })
  });
  return res.json();
}

async function registerFace() {
  vault.style.display  = 'none';
  simWrap.style.display = 'none';
  setStatus('📸 Scanning chehra...', 'info');
  setScanOverlay('scanning');

  try {
    const data = await post('/register', captureFrame());
    setScanOverlay(data.success ? 'success' : 'fail');
    setStatus(data.message, data.success ? 'ok' : 'err');
  } catch(e) {
    setScanOverlay('fail');
    setStatus('❌ Network error: ' + e, 'err');
  }
}

async function unlockVault() {
  vault.style.display   = 'none';
  simWrap.style.display = 'none';
  setStatus('🔍 Verify ho raha hai...', 'info');
  setScanOverlay('scanning');

  try {
    const data = await post('/unlock', captureFrame());
    setScanOverlay(data.success ? 'success' : 'fail');
    setStatus(data.message, data.success ? 'ok' : 'err');

    if (data.similarity !== undefined) {
      showSimilarity(data.similarity);
    }

    if (data.success) {
      const fl = document.getElementById('file-list');
      fl.innerHTML = '';
      (data.files || []).forEach(f => {
        const li = document.createElement('li');
        li.innerHTML = '<a href="/secret/' + encodeURIComponent(f) + '" target="_blank">📄 ' + f + '</a>';
        fl.appendChild(li);
      });
      vault.style.display = 'block';
    }
  } catch(e) {
    setScanOverlay('fail');
    setStatus('❌ Network error: ' + e, 'err');
  }
}

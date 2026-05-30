const video      = document.getElementById('video');
const canvas     = document.getElementById('canvas');
const statusBox  = document.getElementById('status');
const statusText = document.getElementById('status-text');
const overlay    = document.getElementById('scan-overlay');
const vault      = document.getElementById('vault');
const simWrap    = document.getElementById('sim-wrap');
const simBar     = document.getElementById('sim-bar');
const simValue   = document.getElementById('sim-value');
const camLabel   = document.getElementById('cam-label');

let currentStream    = null;
let currentFacing    = 'user';
let sessionToken     = null;

// ── Camera ────────────────────────────────────
async function startCamera(facing) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
  }
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing },
      audio: false
    });
    video.srcObject = currentStream;
    camLabel.textContent = facing === 'user' ? 'Front Camera' : 'Back Camera';
    setStatus('✅ Camera ready.', 'ok');
  } catch(e) {
    setStatus('❌ Camera error: ' + e.message, 'err');
  }
}

function switchCamera(facing) {
  currentFacing = facing;
  document.getElementById('btn-front').classList.toggle('active', facing === 'user');
  document.getElementById('btn-back').classList.toggle('active', facing === 'environment');
  startCamera(facing);
}

// Start with front camera
startCamera('user');

// ── Helpers ───────────────────────────────────
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
  simBar.style.width      = pct + '%';
  simBar.style.background = pct >= 70 ? '#a6e3a1' : pct >= 40 ? '#f9e2af' : '#f38ba8';
  simValue.textContent    = pct + '% match';
}

function captureFrame() {
  const ctx = canvas.getContext('2d');
  canvas.width  = video.videoWidth  || 320;
  canvas.height = video.videoHeight || 240;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
}

async function post(url, body) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body)
  });
  return res.json();
}

// ── Register ──────────────────────────────────
async function registerFace() {
  vault.style.display   = 'none';
  simWrap.style.display = 'none';
  sessionToken          = null;
  setStatus('📸 Scanning chehra...', 'info');
  setScanOverlay('scanning');

  try {
    const data = await post('/register', { image: captureFrame() });
    setScanOverlay(data.success ? 'success' : 'fail');
    setStatus(data.message, data.success ? 'ok' : 'err');
  } catch(e) {
    setScanOverlay('fail');
    setStatus('❌ Network error: ' + e, 'err');
  }
}

// ── Unlock ────────────────────────────────────
async function unlockVault() {
  vault.style.display   = 'none';
  simWrap.style.display = 'none';
  sessionToken          = null;
  setStatus('🔍 Verify ho raha hai...', 'info');
  setScanOverlay('scanning');

  try {
    const data = await post('/unlock', { image: captureFrame() });
    setScanOverlay(data.success ? 'success' : 'fail');
    setStatus(data.message, data.success ? 'ok' : 'err');

    if (data.similarity !== undefined) showSimilarity(data.similarity);

    if (data.success && data.token) {
      sessionToken = data.token;
      const fl = document.getElementById('file-list');
      fl.innerHTML = '';
      (data.files || []).forEach(f => {
        const li  = document.createElement('li');
        const url = `/secret/${encodeURIComponent(f)}?token=${sessionToken}`;
        li.innerHTML = `<a href="${url}" target="_blank">📄 ${f}</a>
                        <a href="${url}" download class="dl-btn">⬇</a>`;
        fl.appendChild(li);
      });
      vault.style.display = 'block';
    }
  } catch(e) {
    setScanOverlay('fail');
    setStatus('❌ Network error: ' + e, 'err');
  }
}
import os
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from PIL import Image

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, ".face_data")
SECRET_DIR = os.path.join(BASE_DIR, ".secret_vault")
KNOWN_FILE = os.path.join(DATA_DIR, "known_face.npy")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(SECRET_DIR, exist_ok=True)

# Sample secret file
sample = os.path.join(SECRET_DIR, "secret_note.txt")
if not os.path.exists(sample):
    with open(sample, "w") as f:
        f.write("Yeh tumhari secret file hai!\n")

# OpenCV haar cascade — sirf ye chahiye
try:
    import cv2
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    CV2_OK = True
except Exception:
    CV2_OK = False


def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def extract_face_vector(img_array):
    """
    Face detect karo, crop karo, 32x32 resize karo,
    flatten karke normalized vector return karo.
    Yahi hamara 'embedding' hai.
    """
    if not CV2_OK:
        return None, "OpenCV available nahi hai."

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, "Koi chehra detect nahi hua. Seedha camera ke saamne baitho, achhi roshni rakho."

    # Sabse bada face lo
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_crop = img_array[y:y+h, x:x+w]
    face_resized = np.array(Image.fromarray(face_crop).resize((32, 32)))
    vector = face_resized.flatten().astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-6)
    return vector, None


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


# ── HTML ──────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Face Vault</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background:#0f0f1a; color:#cdd6f4;
      font-family:'Courier New',monospace;
      display:flex; flex-direction:column;
      align-items:center; padding:20px; min-height:100vh;
    }
    h1 { color:#cba6f7; margin-bottom:4px; font-size:1.5rem; }
    p.sub { color:#6c7086; font-size:0.82rem; margin-bottom:14px; }
    video { border-radius:12px; border:2px solid #313244; width:300px; height:225px; object-fit:cover; }
    canvas { display:none; }
    .btn-row { display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; justify-content:center; }
    button {
      background:#cba6f7; color:#1e1e2e; border:none;
      padding:10px 18px; border-radius:8px; font-weight:bold;
      font-family:inherit; cursor:pointer; font-size:0.88rem;
    }
    button:active { background:#b4a0e5; }
    #status {
      background:#1e1e2e; border:1px solid #313244;
      border-radius:10px; padding:12px 16px;
      margin-top:10px; width:300px; min-height:48px;
      font-size:0.88rem; color:#a6e3a1; text-align:center;
      line-height:1.5;
    }
    #status.err { color:#f38ba8; }
    #vault {
      display:none; background:#1e1e2e;
      border:2px solid #a6e3a1; border-radius:12px;
      padding:14px; margin-top:14px; width:300px;
    }
    #vault h2 { color:#a6e3a1; margin-bottom:10px; font-size:1rem; }
    #vault ul { list-style:none; }
    #vault ul li {
      background:#313244; margin:5px 0;
      padding:8px 10px; border-radius:6px; font-size:0.83rem;
    }
    #vault ul li a { color:#89dceb; text-decoration:none; }
    .hint { color:#6c7086; font-size:0.75rem; margin-top:8px; }
  </style>
</head>
<body>
  <h1>🔒 Face Vault</h1>
  <p class="sub">Pehle register karo, phir vault unlock karo</p>

  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>

  <div class="btn-row">
    <button onclick="registerFace()">📸 Register</button>
    <button onclick="unlockVault()">🔓 Unlock</button>
  </div>
  <p class="hint">💡 Achhi roshni mein seedha camera ke saamne baitho</p>

  <div id="status">Camera start ho rahi hai...</div>

  <div id="vault">
    <h2>🗂 Secret Vault</h2>
    <ul id="file-list"></ul>
  </div>

  <script>
    const video  = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const status = document.getElementById('status');
    const vault  = document.getElementById('vault');
    const ctx    = canvas.getContext('2d');

    navigator.mediaDevices.getUserMedia({ video: { facingMode:'user' }, audio:false })
      .then(stream => {
        video.srcObject = stream;
        status.textContent = "✅ Camera ready — register karo ya unlock karo.";
        status.className = '';
      })
      .catch(err => {
        status.textContent = "❌ Camera error: " + err.message;
        status.className = 'err';
      });

    function captureFrame() {
      canvas.width  = video.videoWidth  || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
    }

    async function post(url, img) {
      const res  = await fetch(url, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ image: img })
      });
      return res.json();
    }

    async function registerFace() {
      status.textContent = "📸 Scanning...";
      status.className = '';
      vault.style.display = 'none';
      try {
        const data = await post('/register', captureFrame());
        status.textContent = data.message;
        status.className = data.success ? '' : 'err';
      } catch(e) {
        status.textContent = "❌ " + e;
        status.className = 'err';
      }
    }

    async function unlockVault() {
      status.textContent = "🔍 Chehra verify ho raha hai...";
      status.className = '';
      vault.style.display = 'none';
      try {
        const data = await post('/unlock', captureFrame());
        status.textContent = data.message;
        status.className = data.success ? '' : 'err';
        if (data.success) {
          const fl = document.getElementById('file-list');
          fl.innerHTML = '';
          (data.files || []).forEach(f => {
            const li = document.createElement('li');
            li.innerHTML = '<a href="/secret/' + f + '" target="_blank">📄 ' + f + '</a>';
            fl.appendChild(li);
          });
          vault.style.display = 'block';
        }
      } catch(e) {
        status.textContent = "❌ " + e;
        status.className = 'err';
      }
    }
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."})

    img_array = decode_image(data['image'])
    vector, err = extract_face_vector(img_array)

    if err:
        return jsonify({"success": False, "message": f"❌ {err}"})

    np.save(KNOWN_FILE, vector)
    return jsonify({"success": True, "message": "✅ Chehra register ho gaya! Ab Unlock try karo."})


@app.route('/unlock', methods=['POST'])
def unlock():
    if not os.path.exists(KNOWN_FILE):
        return jsonify({"success": False, "message": "❌ Pehle face register karo."})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."})

    known_vector = np.load(KNOWN_FILE)
    img_array    = decode_image(data['image'])
    vector, err  = extract_face_vector(img_array)

    if err:
        return jsonify({"success": False, "message": f"❌ {err}"})

    similarity = cosine_similarity(known_vector, vector)

    # 0.92+ = match (tune kar sakte ho)
    THRESHOLD = 0.92

    if similarity >= THRESHOLD:
        files = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
        return jsonify({
            "success": True,
            "message": f"✅ Match! Vault unlock ho gaya. (similarity: {similarity:.3f})",
            "files": files
        })
    else:
        return jsonify({
            "success": False,
            "message": f"❌ Chehra match nahi hua. (similarity: {similarity:.3f})"
        })


@app.route('/secret/<filename>')
def serve_secret(filename):
    safe = os.path.basename(filename)
    return send_from_directory(SECRET_DIR, safe)


if __name__ == '__main__':
    print("\n🔒 Face Vault start ho raha hai...")
    print("📱 Browser mein kholo: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

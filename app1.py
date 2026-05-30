import os
import json
import hashlib
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image
import face_recognition
import numpy as np
import io
import base64

app = Flask(__name__)

# ── Paths ──────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, ".face_data")
SECRET_DIR  = os.path.join(BASE_DIR, ".secret_vault")
KNOWN_FILE  = os.path.join(DATA_DIR, "known_face.npy")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(SECRET_DIR, exist_ok=True)

# Secret folder mein kuch sample files daalo (test ke liye)
sample = os.path.join(SECRET_DIR, "secret_note.txt")
if not os.path.exists(sample):
    with open(sample, "w") as f:
        f.write("Yeh tumhari secret file hai. Sirf tumhara chehra dekh sakta hai.\n")


# ── HTML ───────────────────────────────────────
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
      background: #0f0f1a;
      color: #cdd6f4;
      font-family: 'Courier New', monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      min-height: 100vh;
    }
    h1 { color: #cba6f7; margin-bottom: 6px; font-size: 1.5rem; }
    p.sub { color: #6c7086; font-size: 0.85rem; margin-bottom: 16px; }
    video, canvas {
      border-radius: 12px;
      border: 2px solid #313244;
      width: 320px; height: 240px;
      object-fit: cover;
    }
    canvas { display: none; }
    .btn-row { display: flex; gap: 10px; margin: 14px 0; flex-wrap: wrap; justify-content: center; }
    button {
      background: #cba6f7;
      color: #1e1e2e;
      border: none;
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: bold;
      font-family: inherit;
      cursor: pointer;
      font-size: 0.9rem;
    }
    button:hover { background: #b4a0e5; }
    button.danger { background: #f38ba8; }
    #status {
      background: #1e1e2e;
      border: 1px solid #313244;
      border-radius: 10px;
      padding: 12px 18px;
      margin-top: 12px;
      width: 320px;
      min-height: 50px;
      font-size: 0.9rem;
      color: #a6e3a1;
      text-align: center;
    }
    #vault {
      display: none;
      background: #1e1e2e;
      border: 2px solid #a6e3a1;
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
      width: 320px;
    }
    #vault h2 { color: #a6e3a1; margin-bottom: 10px; }
    #vault ul { list-style: none; padding: 0; }
    #vault ul li {
      background: #313244;
      margin: 6px 0;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
    }
    #vault ul li a { color: #89dceb; text-decoration: none; }
  </style>
</head>
<body>
  <h1>🔒 Face Vault</h1>
  <p class="sub">Apna chehra scan karo — secret folder unlock hoga</p>

  <video id="video" autoplay playsinline></video>
  <canvas id="canvas"></canvas>

  <div class="btn-row">
    <button onclick="registerFace()">📸 Face Register Karo</button>
    <button onclick="unlockVault()">🔓 Vault Unlock Karo</button>
  </div>

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

    // Camera start
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      .then(stream => {
        video.srcObject = stream;
        status.textContent = "✅ Camera ready hai.";
      })
      .catch(err => {
        status.textContent = "❌ Camera error: " + err.message;
      });

    function captureFrame() {
      canvas.width  = video.videoWidth  || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
    }

    async function registerFace() {
      status.textContent = "📸 Face register ho raha hai...";
      vault.style.display = 'none';
      const img = captureFrame();
      try {
        const res  = await fetch('/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: img })
        });
        const data = await res.json();
        status.textContent = data.message;
      } catch(e) {
        status.textContent = "❌ Error: " + e;
      }
    }

    async function unlockVault() {
      status.textContent = "🔍 Chehra scan ho raha hai...";
      vault.style.display = 'none';
      const img = captureFrame();
      try {
        const res  = await fetch('/unlock', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: img })
        });
        const data = await res.json();
        status.textContent = data.message;
        if (data.success) {
          const fl = document.getElementById('file-list');
          fl.innerHTML = '';
          (data.files || []).forEach(f => {
            const li = document.createElement('li');
            li.innerHTML = `<a href="/secret/${f}" target="_blank">📄 ${f}</a>`;
            fl.appendChild(li);
          });
          vault.style.display = 'block';
        }
      } catch(e) {
        status.textContent = "❌ Error: " + e;
      }
    }
  </script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


def decode_image(b64_str):
    """Base64 → PIL Image → numpy array"""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"message": "❌ Image nahi mili."}), 400

    img_array = decode_image(data['image'])
    encodings = face_recognition.face_encodings(img_array)

    if not encodings:
        return jsonify({"message": "❌ Koi chehra detect nahi hua. Seedha camera ke saamne baitho."}), 400

    np.save(KNOWN_FILE, encodings[0])
    return jsonify({"message": "✅ Chehra successfully register ho gaya!"}), 200


@app.route('/unlock', methods=['POST'])
def unlock():
    if not os.path.exists(KNOWN_FILE):
        return jsonify({
            "success": False,
            "message": "❌ Pehle face register karo."
        }), 400

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."}), 400

    known_encoding = np.load(KNOWN_FILE)
    img_array      = decode_image(data['image'])
    encodings      = face_recognition.face_encodings(img_array)

    if not encodings:
        return jsonify({
            "success": False,
            "message": "❌ Chehra detect nahi hua. Theek se camera ke saamne baitho."
        }), 400

    match = face_recognition.compare_faces(
        [known_encoding], encodings[0], tolerance=0.5
    )

    if match[0]:
        files = os.listdir(SECRET_DIR)
        files = [f for f in files if not f.startswith('.')]
        return jsonify({
            "success": True,
            "message": "✅ Chehra match hua! Vault unlock ho gaya.",
            "files": files
        })
    else:
        return jsonify({
            "success": False,
            "message": "❌ Chehra match nahi hua. Access denied."
        })


@app.route('/secret/<filename>')
def serve_secret(filename):
    # Sirf text files serve karo safely
    safe_name = os.path.basename(filename)
    return send_from_directory(SECRET_DIR, safe_name)


if __name__ == '__main__':
    print("\n🔒 Face Vault server start ho raha hai...")
    print("📱 Browser mein kholo: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

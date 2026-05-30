import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import cv2

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "face_data")
SECRET_DIR = os.path.join(BASE_DIR, "secret_vault")
KNOWN_FILE = os.path.join(DATA_DIR, "known_face.npy")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(SECRET_DIR, exist_ok=True)

sample = os.path.join(SECRET_DIR, "secret_note.txt")
if not os.path.exists(sample):
    with open(sample, "w") as f:
        f.write("Yeh tumhari secret file hai!\n")

# ── Global error handlers ──────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

# ── Lazy load InsightFace ──────────────────────
_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def get_face_embedding(img_array):
    try:
        fa   = get_face_app()
        bgr  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = fa.get(bgr)
    except Exception as e:
        return None, f"Model error: {str(e)}"

    if not faces:
        return None, "Koi chehra detect nahi hua. Seedha camera ke saamne baitho."

    if len(faces) > 1:
        return None, "Multiple faces detected. Akele frame mein aao."

    face = faces[0]

    if face.det_score < 0.75:
        return None, "Detection confidence kam hai. Better lighting mein try karo."

    bbox   = face.bbox.astype(int)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    img_h, img_w = img_array.shape[:2]

    if (face_w * face_h) / (img_w * img_h) < 0.04:
        return None, "Chehra bahut chota hai. Camera ke paas aao."

    gray_face     = cv2.cvtColor(
        img_array[max(0,bbox[1]):bbox[3], max(0,bbox[0]):bbox[2]],
        cv2.COLOR_RGB2GRAY
    )
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if laplacian_var < 20:
        return None, "Image bahut blurry hai. Stable raho aur phir try karo."

    embedding = face.normed_embedding
    if np.linalg.norm(embedding) < 0.5:
        return None, "Face embedding weak hai. Real face camera ke saamne rakho."

    return embedding, None


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "❌ Image nahi mili."})

        img_array      = decode_image(data['image'])
        embedding, err = get_face_embedding(img_array)

        if err:
            return jsonify({"success": False, "message": f"❌ {err}"})

        np.save(KNOWN_FILE, embedding)
        return jsonify({"success": True, "message": "✅ Chehra register ho gaya!"})

    except Exception as e:
        return jsonify({"success": False, "message": f"❌ Server error: {str(e)}"})


@app.route('/unlock', methods=['POST'])
def unlock():
    try:
        if not os.path.exists(KNOWN_FILE):
            return jsonify({"success": False, "message": "❌ Pehle face register karo."})

        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "❌ Image nahi mili."})

        known_embedding = np.load(KNOWN_FILE)
        img_array       = decode_image(data['image'])
        embedding, err  = get_face_embedding(img_array)

        if err:
            return jsonify({"success": False, "message": f"❌ {err}", "similarity": 0.0})

        similarity = round(cosine_similarity(known_embedding, embedding), 4)
        THRESHOLD  = 0.35

        if similarity >= THRESHOLD:
            files = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
            return jsonify({
                "success":    True,
                "message":    f"✅ Vault unlock! ({similarity*100:.1f}%)",
                "files":      files,
                "similarity": similarity
            })
        else:
            return jsonify({
                "success":    False,
                "message":    f"❌ Match nahi hua. ({similarity*100:.1f}%)",
                "similarity": similarity
            })

    except Exception as e:
        return jsonify({"success": False, "message": f"❌ Server error: {str(e)}"})


@app.route('/secret/<filename>')
def serve_secret(filename):
    safe = os.path.basename(filename)
    return send_from_directory(SECRET_DIR, safe)


@app.route('/status')
def status():
    registered = os.path.exists(KNOWN_FILE)
    files      = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
    return jsonify({"registered": registered, "secret_files": len(files)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
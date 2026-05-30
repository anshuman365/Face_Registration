import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

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

# ── InsightFace load ───────────────────────────
import insightface
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(
    name="buffalo_sc",        # lightest model — free tier ke liye
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(320, 320))


def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def get_face_embedding(img_array):
    """
    InsightFace ArcFace 512-D embedding return karta hai.
    - Real face pe hi kaam karta hai
    - Cartoon/sketch/printed photo mostly fail hoti hai
    - Liveness ke liye blur + sharpness check bhi hai
    """
    import cv2
    # BGR convert (insightface BGR expect karta hai)
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)

    if not faces:
        return None, "Koi chehra detect nahi hua. Seedha camera ke saamne baitho, achhi roshni rakho."

    if len(faces) > 1:
        return None, "Multiple faces detected. Akele frame mein aao."

    face = faces[0]

    # ── Anti-spoofing checks ───────────────────

    # 1. Detection confidence
    if face.det_score < 0.75:
        return None, "Face detection confidence kam hai. Better lighting mein try karo."

    # 2. Face size check — printed/far photo filter
    bbox = face.bbox.astype(int)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    img_h, img_w = img_array.shape[:2]
    face_ratio = (face_w * face_h) / (img_w * img_h)

    if face_ratio < 0.04:
        return None, "Chehra bahut chota hai. Camera ke paas aao."

    # 3. Blur detection — printed photo pe focus sharp hoti hai differently
    gray_face = cv2.cvtColor(
        img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]],
        cv2.COLOR_RGB2GRAY
    )
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    if laplacian_var < 20:
        return None, "Image bahut blurry hai. Stable raho aur phir try karo."

    # 4. Embedding norm check — fake face pe embedding weak hoti hai
    embedding = face.normed_embedding
    emb_norm  = np.linalg.norm(embedding)
    if emb_norm < 0.5:
        return None, "Face embedding weak hai. Real face camera ke saamne rakho."

    return embedding, None


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."})

    img_array       = decode_image(data['image'])
    embedding, err  = get_face_embedding(img_array)

    if err:
        return jsonify({"success": False, "message": f"❌ {err}"})

    np.save(KNOWN_FILE, embedding)
    return
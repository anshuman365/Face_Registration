import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

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
        f.write("Yeh tumhari secret file hai. Sirf tumhara chehra dekh sakta hai.\n")


def decode_image(b64_str):
    from PIL import Image
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def extract_face_vector(img_array):
    import cv2
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        return None, f"OpenCV load error: {e}"

    gray  = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        return None, "Koi chehra detect nahi hua. Seedha camera ke saamne baitho, achhi roshni rakho."

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    from PIL import Image
    face_crop    = img_array[y:y+h, x:x+w]
    face_resized = np.array(Image.fromarray(face_crop).resize((64, 64)))
    vector       = face_resized.flatten().astype(np.float32)
    norm         = np.linalg.norm(vector)
    vector       = vector / (norm + 1e-6)
    return vector, None


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

    img_array    = decode_image(data['image'])
    vector, err  = extract_face_vector(img_array)

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
    THRESHOLD  = 0.90

    if similarity >= THRESHOLD:
        files = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
        return jsonify({
            "success":    True,
            "message":    f"✅ Match! Vault unlock ho gaya. ({similarity:.3f})",
            "files":      files,
            "similarity": round(similarity, 3)
        })
    else:
        return jsonify({
            "success":    False,
            "message":    f"❌ Chehra match nahi hua. ({similarity:.3f})",
            "similarity": round(similarity, 3)
        })


@app.route('/secret/<filename>')
def serve_secret(filename):
    safe = os.path.basename(filename)
    return send_from_directory(SECRET_DIR, safe)


@app.route('/status')
def status():
    registered = os.path.exists(KNOWN_FILE)
    files      = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
    return jsonify({
        "registered":   registered,
        "secret_files": len(files)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

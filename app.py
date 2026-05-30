import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import face_recognition

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


def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def get_face_encoding(img_array):
    """
    face_recognition 128-dimensional encoding return karta hai.
    Ye sirf REAL face pe kaam karta hai — cartoons/sketches/photos
    mein face landmarks detect nahi hote properly.
    """
    # Pehle face locations dhundho
    face_locations = face_recognition.face_locations(
        img_array,
        number_of_times_to_upsample=1,
        model="hog"   # "cnn" more accurate but slow on free tier
    )

    if not face_locations:
        return None, "Koi chehra detect nahi hua. Seedha camera ke saamne baitho."

    if len(face_locations) > 1:
        return None, "Multiple faces detected. Akele frame mein aao."

    # 128-D encoding
    encodings = face_recognition.face_encodings(
        img_array,
        known_face_locations=face_locations,
        num_jitters=2   # zyada jitters = zyada accurate
    )

    if not encodings:
        return None, "Face encoding fail hua. Phir try karo."

    return encodings[0], None


def get_face_landmarks_score(img_array):
    """
    Extra check: landmarks detect hone chahiye (eyes, nose, lips).
    Cartoons/sketches mein ye properly nahi milte.
    """
    landmarks_list = face_recognition.face_landmarks(img_array)
    if not landmarks_list:
        return False
    lm = landmarks_list[0]
    # Real face mein ye sab hone chahiye
    required = ['left_eye', 'right_eye', 'nose_tip', 'top_lip', 'bottom_lip']
    found = sum(1 for part in required if part in lm and len(lm[part]) > 0)
    return found >= 4


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."})

    img_array = decode_image(data['image'])

    # Landmarks check — cartoon/sketch filter
    if not get_face_landmarks_score(img_array):
        return jsonify({
            "success": False,
            "message": "❌ Real human face detect nahi hua. Photo ya cartoon se register nahi hoga."
        })

    encoding, err = get_face_encoding(img_array)
    if err:
        return jsonify({"success": False, "message": f"❌ {err}"})

    np.save(KNOWN_FILE, encoding)
    return jsonify({"success": True, "message": "✅ Chehra register ho gaya! Ab Unlock try karo."})


@app.route('/unlock', methods=['POST'])
def unlock():
    if not os.path.exists(KNOWN_FILE):
        return jsonify({"success": False, "message": "❌ Pehle face register karo."})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "❌ Image nahi mili."})

    known_encoding = np.load(KNOWN_FILE)
    img_array      = decode_image(data['image'])

    # Step 1 — Landmarks check (cartoon/sketch/photo block)
    if not get_face_landmarks_score(img_array):
        return jsonify({
            "success": False,
            "message": "❌ Real human face detect nahi hua. Photo/cartoon/sketch se access nahi milega.",
            "similarity": 0.0
        })

    # Step 2 — Encoding
    encoding, err = get_face_encoding(img_array)
    if err:
        return jsonify({"success": False, "message": f"❌ {err}", "similarity": 0.0})

    # Step 3 — Distance comparison
    # face_recognition distance: 0.0 = perfect match, 0.6+ = different person
    distance  = face_recognition.face_distance([known_encoding], encoding)[0]
    # Convert to similarity percentage (0.6 distance = 0% match for display)
    similarity = max(0.0, 1.0 - (distance / 0.6))
    similarity = round(float(similarity), 3)

    # Strict threshold — 0.45 distance = same person (tighter than default 0.6)
    DISTANCE_THRESHOLD = 0.45

    if distance <= DISTANCE_THRESHOLD:
        files = [f for f in os.listdir(SECRET_DIR) if not f.startswith('.')]
        return jsonify({
            "success":    True,
            "message":    f"✅ Match! Vault unlock ho gaya. ({similarity*100:.1f}% match)",
            "files":      files,
            "similarity": similarity,
            "distance":   round(float(distance), 4)
        })
    else:
        return jsonify({
            "success":    False,
            "message":    f"❌ Chehra match nahi hua. ({similarity*100:.1f}% match)",
            "similarity": similarity,
            "distance":   round(float(distance), 4)
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
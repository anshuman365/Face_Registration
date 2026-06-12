import os
import io
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import cv2

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── Error handlers ─────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

# ── InsightFace lazy load (single instance) ────
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
    """Convert base64 string to RGB numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)

def get_face_embedding(img_array):
    """
    Extract face embedding with quality checks.
    Returns (embedding, error_message) – error_message is None on success.
    """
    try:
        fa = get_face_app()
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = fa.get(bgr)
    except Exception as e:
        return None, f"Model error: {str(e)}"

    if not faces:
        return None, "No face detected. Look straight into the camera."

    if len(faces) > 1:
        return None, "Multiple faces detected. Please be alone in the frame."

    face = faces[0]

    # Confidence check
    if face.det_score < 0.75:
        return None, "Detection confidence low. Improve lighting and try again."

    # Face size check (at least 4% of image area)
    bbox = face.bbox.astype(int)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    img_h, img_w = img_array.shape[:2]
    if (face_w * face_h) / (img_w * img_h) < 0.04:
        return None, "Face is too small. Move closer to the camera."

    # Blur detection using Laplacian variance
    gray_face = cv2.cvtColor(
        img_array[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]],
        cv2.COLOR_RGB2GRAY
    )
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if laplacian_var < 20:
        return None, "Image is too blurry. Stay steady and try again."

    embedding = face.normed_embedding
    if np.linalg.norm(embedding) < 0.5:
        return None, "Weak face embedding. Ensure real face is in front of camera."

    return embedding, None

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

# ── API Routes ─────────────────────────────────

@app.route('/')
def index():
    """Test page with camera UI."""
    return render_template('index.html')

@app.route('/ping')
def ping():
    """Keep-alive endpoint to prevent Render from sleeping."""
    return jsonify({"status": "alive", "timestamp": datetime.utcnow().isoformat()})

@app.route('/register/<int:student_id>', methods=['POST'])
def register_face(student_id):
    """
    Store face embedding for a student using registration photo.
    Request: JSON { "image": "base64_image_data" }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "Image not provided."})

        img_array = decode_image(data['image'])
        embedding, err = get_face_embedding(img_array)

        if err:
            return jsonify({"success": False, "message": err})

        # Save embedding
        save_path = os.path.join(EMBEDDINGS_DIR, f"{student_id}.npy")
        np.save(save_path, embedding)

        return jsonify({
            "success": True,
            "message": f"Face registered for student ID {student_id}."
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/verify/<int:student_id>', methods=['POST'])
def verify_face(student_id):
    """
    Compare live photo with stored embedding.
    Request: JSON { "image": "base64_image_data" }
    Returns similarity score and match boolean.
    """
    try:
        embed_path = os.path.join(EMBEDDINGS_DIR, f"{student_id}.npy")
        if not os.path.exists(embed_path):
            return jsonify({
                "success": False,
                "message": f"No registered face for student ID {student_id}. Please register first."
            })

        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "Image not provided."})

        known_embedding = np.load(embed_path)
        img_array = decode_image(data['image'])
        embedding, err = get_face_embedding(img_array)

        if err:
            return jsonify({"success": False, "message": err, "similarity": 0.0})

        similarity = round(cosine_similarity(known_embedding, embedding), 4)
        THRESHOLD = 0.35   # Same as original

        if similarity >= THRESHOLD:
            return jsonify({
                "success": True,
                "message": f"Match successful ({similarity*100:.1f}%)",
                "similarity": similarity,
                "matched": True
            })
        else:
            return jsonify({
                "success": True,   # operation succeeded, but face didn't match
                "message": f"Match failed ({similarity*100:.1f}%)",
                "similarity": similarity,
                "matched": False
            })
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/delete/<int:student_id>', methods=['DELETE'])
def delete_face(student_id):
    """Remove stored embedding for a student (e.g., when student is deleted)."""
    embed_path = os.path.join(EMBEDDINGS_DIR, f"{student_id}.npy")
    if os.path.exists(embed_path):
        os.remove(embed_path)
        return jsonify({"success": True, "message": f"Face data for student {student_id} deleted."})
    else:
        return jsonify({"success": False, "message": "No face data found for this student."})

@app.route('/status')
def status():
    """Return service health and registration count."""
    registered = len([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')])
    return jsonify({
        "service": "Face Verification Service",
        "registered_faces": registered,
        "status": "operational"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
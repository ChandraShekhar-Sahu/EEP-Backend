import os
import cv2
import time
import numpy as np
import face_recognition
import dlib

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# =========================
# BASE PATH CONFIG (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "api",
    "models",
    "shape_predictor_68_face_landmarks.dat"
)

# Load models once
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

UPLOAD_DIR = os.path.join(BASE_DIR, "api", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_DIR, "reference_face.jpg")


# =========================
# IMAGE READER (FROM REACT)
# =========================
def read_image(request):
    file = request.FILES.get("image")
    if not file:
        return None

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return frame


# =========================
# HEAD MOVEMENT DETECTION
# =========================
def detect_head_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return "No Face Detected"

    face = faces[0]
    landmarks = predictor(gray, face)

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    _, yaw, _ = angles

    if yaw > 15:
        return "Looking Right"
    elif yaw < -15:
        return "Looking Left"
    else:
        return "Looking Center"


# =========================
# LIVENESS DETECTION
# =========================
liveness_state = {
    "last_blink_time": time.time(),
    "blink_detected": False
}

def detect_liveness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return "No Face Detected"

    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    left_ear = ear(left_eye)
    right_ear = ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    EAR_THRESHOLD = 0.21
    BLINK_INTERVAL = 3.0

    current_time = time.time()

    if avg_ear < EAR_THRESHOLD:
        liveness_state["blink_detected"] = True
        liveness_state["last_blink_time"] = current_time

    if current_time - liveness_state["last_blink_time"] <= BLINK_INTERVAL:
        return "Blink Detected - Real Person"

    return "No Recent Blink - Potential Risk"


# =========================
# API 1: ANALYZE FRAME
# =========================
@csrf_exempt
def analyze_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    frame = read_image(request)
    if frame is None:
        return JsonResponse({"error": "No image provided"}, status=400)

    return JsonResponse({
        "head_movement": detect_head_movement(frame),
        "liveness": detect_liveness(frame)
    })


# =========================
# API 2: UPLOAD REFERENCE FACE
# =========================
@csrf_exempt
def upload_photo(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    file = request.FILES.get("image")
    if not file:
        return JsonResponse({"error": "No image"}, status=400)

    path = REFERENCE_IMAGE_PATH

    with open(path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    return JsonResponse({"message": "Reference uploaded"})


# =========================
# API 3: FACE VERIFICATION
# =========================
@csrf_exempt
def verify_face(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    frame = read_image(request)
    if frame is None:
        return JsonResponse({"error": "No image provided"}, status=400)

    if not os.path.exists(REFERENCE_IMAGE_PATH):
        return JsonResponse({"error": "Reference face not found"}, status=400)

    uploaded_enc = face_recognition.face_encodings(frame)
    if len(uploaded_enc) == 0:
        return JsonResponse({
            "match": False,
            "face_detected": False,
            "message": "No face detected"
        })

    ref_img = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
    ref_enc = face_recognition.face_encodings(ref_img)

    if len(ref_enc) == 0:
        return JsonResponse({"error": "Invalid reference image"}, status=400)

    match = face_recognition.compare_faces([ref_enc[0]], uploaded_enc[0])[0]

    return JsonResponse({
        "match": bool(match),
        "face_detected": True,
        "message": "Face Matched" if match else "Face Not Matched",
        "head_movement": detect_head_movement(frame),
        "liveness": detect_liveness(frame)
    })
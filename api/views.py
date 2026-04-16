# //env\Scripts\activate

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import cv2
import numpy as np
import face_recognition
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import dlib
import time

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\DELL\Downloads\f drive\MinorProject\ExamMonitoring\exam\api\shape_predictor_68_face_landmarks.dat")  # Update this path


# Path to reference image (update with the actual reference image path)

UPLOAD_DIR = "api/uploads/"
REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_DIR, "reference_face.jpg")

# ---------------------- HEAD MOVEMENT DETECTION ---------------------- #
def detect_head_movement(frame):
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        _, yaw, _ = angles

        if yaw > 15:
            return "Looking Right"
        elif yaw < -15:
            return "Looking Left"
        else:
            return "Looking Center"
    
    return "No Face Detected"

# ---------------------- LIVENESS DETECTION ---------------------- #

# Use a dictionary to simulate session storage for blinking detection
liveness_state = {"last_blink_time": time.time(), "blink_detected": False}

def detect_liveness(frame):
    EAR_THRESHOLD = 0.21  # Slightly lower to detect subtle blinks
    BLINK_INTERVAL = 3.0  # Seconds within which blink should happen

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        current_time = time.time()

        if ear < EAR_THRESHOLD:
            if not liveness_state["blink_detected"]:
                liveness_state["blink_detected"] = True
                liveness_state["last_blink_time"] = current_time
        else:
            # Reset blink status if enough time has passed
            if current_time - liveness_state["last_blink_time"] > BLINK_INTERVAL:
                liveness_state["blink_detected"] = False

        if current_time - liveness_state["last_blink_time"] <= BLINK_INTERVAL:
            return "Blink Detected - Real Person"
        else:
            return "No Recent Blink - Potential Video"

    return "No Face Detected"

# ---------------------- SAMPLE DJANGO VIEW ---------------------- #
def analyze_frame(request):
    # In real scenario, image data would come from frontend POST
    # For testing, use a static webcam frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JsonResponse({"error": "Failed to capture image"}, status=500)

    head_status = detect_head_movement(frame)
    liveness_status = detect_liveness(frame)

    return JsonResponse({
        "head_movement": head_status,
        "liveness": liveness_status
    })


@csrf_exempt
def upload_photo(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
            os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the directory exists

            image = request.FILES["image"]
            file_path = os.path.join(UPLOAD_DIR, "reference_face.jpg")  # Save as reference face

            # Save the uploaded image as the reference image
            with open(file_path, "wb") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            return JsonResponse({"message": "Reference photo uploaded successfully!", "path": file_path})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def verify_face(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            image_file = request.FILES["image"]
            file_path = os.path.join(UPLOAD_DIR, "temp_face.jpg")

            with open(file_path, "wb") as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return JsonResponse({"error": "File not saved or empty"}, status=400)

            try:
                img = Image.open(file_path)
                img.verify()
            except Exception as e:
                return JsonResponse({"error": "Invalid image file"}, status=400)

            uploaded_image = face_recognition.load_image_file(file_path)
            uploaded_encodings = face_recognition.face_encodings(uploaded_image)

            if not uploaded_encodings:
                # Cleanup file
                if os.path.exists(file_path): os.remove(file_path)
                
                return JsonResponse({
                    "match": False,
                    "face_detected": False, 
                    "message": "No face detected in uploaded image"})

            uploaded_encoding = uploaded_encodings[0]
            ref_image = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
            ref_encodings = face_recognition.face_encodings(ref_image)

            if not ref_encodings:
                return JsonResponse({"error": "No face found in reference image"}, status=400)

            ref_encoding = ref_encodings[0]
            match = bool(face_recognition.compare_faces([ref_encoding], uploaded_encoding)[0])
            message = "Face Matched!" if match else "Face Not Matched!"

            # Read the image again using OpenCV for additional analysis
            cv_image = cv2.imread(file_path)

            # Liveness and Head Movement Detection
            liveness = detect_liveness(cv_image)
            head_status = detect_head_movement(cv_image)

            return JsonResponse({
                "match": match,
                "face_detected": True,
                "message": message,
                "liveness": liveness == "Blink Detected - Real Person",
                "liveness_text": liveness,
                "head_movement": (
                    "suspicious" if head_status in ["Looking Left", "Looking Right"]
                    else "normal" if head_status == "Looking Center"
                    else "unknown"
                )
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)

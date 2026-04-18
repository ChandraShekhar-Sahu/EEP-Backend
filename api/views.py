import os
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(BASE_DIR, "api", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_DIR, "reference_face.jpg")

ML_API = "https://face-recognition-service-beng.onrender.com/analyze"


# =========================
# API 1: ANALYZE FRAME (via ML service)
# =========================
@csrf_exempt
def analyze_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"error": "No image provided"}, status=400)

    try:
        response = requests.post(
            ML_API,
            files={"image": (image.name, image, image.content_type)},
            timeout=10
        )
        print(response.status_code)
        print(response.text)
        return JsonResponse(response.json())

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


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

    with open(REFERENCE_IMAGE_PATH, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    return JsonResponse({"message": "Reference uploaded"})


# =========================
# API 3: FACE VERIFICATION (via ML service)
# =========================
@csrf_exempt
def verify_face(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"error": "No image provided"}, status=400)

    if not os.path.exists(REFERENCE_IMAGE_PATH):
        return JsonResponse({"error": "Reference image not found"}, status=400)

    try:
        with open(REFERENCE_IMAGE_PATH, "rb") as ref_file:
            response = requests.post(
                "https://face-recognition-service-beng.onrender.com/api/verify_face",
                files={
                    "reference": ("reference.jpg", ref_file, "image/jpeg"),
                    "image": (image.name, image, image.content_type)
                },
                timeout=15
            )
            # ✅ Debug logs (AFTER response)
        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        # ✅ Safe JSON parsing
        try:
            data = response.json()
        except Exception:
            return JsonResponse({
                "error": "Invalid response from ML service",
                "status_code": response.status_code,
                "raw": response.text[:200]
            }, status=500)
        
        return JsonResponse(response.json())

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
from django.urls import path
from .views import upload_photo, verify_face

urlpatterns = [
    path("upload_photo/", upload_photo, name="upload_photo"),
    path("verify_face/", verify_face, name="verify_face"),
]

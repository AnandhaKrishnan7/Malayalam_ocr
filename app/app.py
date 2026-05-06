import os
import cv2
import json
import base64
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from google.cloud import vision
from google.oauth2 import service_account

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def get_vision_client():
    """
    Initialize Google Vision client.
    Supports two auth methods:
      1. GOOGLE_APPLICATION_CREDENTIALS env var pointing to a JSON key file
      2. GOOGLE_CREDENTIALS_JSON env var containing the JSON key content directly
         (useful for Render where you can't upload files easily)
    """
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        import json as json_mod
        creds_dict = json_mod.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return vision.ImageAnnotatorClient(credentials=credentials)
    # Fallback: GOOGLE_APPLICATION_CREDENTIALS file path
    return vision.ImageAnnotatorClient()


def preprocess_for_handwriting(path):
    """
    Gentle preprocessing: mild sharpening + 2x upscale.
    Do NOT use morphological ops — they break Malayalam curved strokes.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness > 180:
        # Clean image: mild sharpening
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        processed = cv2.filter2D(gray, -1, kernel)
    else:
        # Noisy image: denoise first
        processed = cv2.fastNlMeansDenoising(gray, h=10)

    # 2x upscale for better character resolution
    upscaled = cv2.resize(processed, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

    name, ext = os.path.splitext(path)
    processed_path = f"{name}_ocr_ready{ext}"
    cv2.imwrite(processed_path, upscaled)
    return processed_path


MALAYALAM_UNICODE_RANGE = (0x0D00, 0x0D7F)

def contains_malayalam(text):
    """Check if text contains Malayalam Unicode characters."""
    for char in text:
        code = ord(char)
        if MALAYALAM_UNICODE_RANGE[0] <= code <= MALAYALAM_UNICODE_RANGE[1]:
            return True
    return False


def detect_text_with_google_vision(image_path):
    """
    Use Google Cloud Vision DOCUMENT_TEXT_DETECTION for handwriting.
    This is purpose-built for handwritten documents and supports Malayalam.
    Free tier: 1000 images/month at no cost.
    """
    try:
        vision_client = get_vision_client()

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)

        # Use image_context to hint Malayalam language for better accuracy
        image_context = vision.ImageContext(
            language_hints=["ml"]  # 'ml' = Malayalam BCP-47 code
        )

        # DOCUMENT_TEXT_DETECTION is optimized for handwritten documents
        response = vision_client.document_text_detection(
            image=image,
            image_context=image_context
        )

        if response.error.message:
            return {
                "text": "",
                "is_malayalam": False,
                "language": "Unknown",
                "error": f"Google Vision API error: {response.error.message}"
            }

        full_text = response.full_text_annotation.text.strip()

        if not full_text:
            return {
                "text": "",
                "is_malayalam": False,
                "language": "Unknown",
                "error": None
            }

        is_malayalam = contains_malayalam(full_text)

        # Try to detect language from response pages
        language = "Unknown"
        if response.full_text_annotation.pages:
            page = response.full_text_annotation.pages[0]
            if page.property.detected_languages:
                language = page.property.detected_languages[0].language_code

        return {
            "text": full_text,
            "is_malayalam": is_malayalam,
            "language": language or ("Malayalam" if is_malayalam else "Unknown"),
            "error": None
        }

    except Exception as e:
        return {
            "text": "",
            "is_malayalam": False,
            "language": "Unknown",
            "error": str(e)
        }


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    image_path = ""
    processed_path = ""
    error_msg = ""
    language = ""

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            processed_path = preprocess_for_handwriting(image_path)

            if processed_path is None:
                error_msg = "Error: Could not read image."
            else:
                # Send original to Vision API (better quality than preprocessed)
                result = detect_text_with_google_vision(image_path)

                if result["error"]:
                    error_msg = result["error"]
                elif result["is_malayalam"]:
                    text = result["text"] or "No clear text detected."
                    language = result["language"]
                elif result["text"]:
                    # Text detected but not Malayalam
                    error_msg = f"Not Malayalam. Detected language: {result['language']}. Text: {result['text']}"
                else:
                    error_msg = "No text detected in the image."
        else:
            error_msg = "Please upload an image."

    return render_template(
        "index.html",
        text=text,
        language=language,
        image=os.path.basename(image_path) if image_path else "",
        processed=os.path.basename(processed_path) if processed_path else "",
        error=error_msg
    )


if __name__ == "__main__":
    app.run(debug=True)

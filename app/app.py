import os
import cv2
import json
import base64
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from groq import Groq

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ✅ Better: use environment variable instead of hardcoding
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def preprocess_for_handwriting(path):
    """Preprocess the image: remove lines, denoise, upscale."""
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    remove_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 5)

    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Crop text area
    coords = cv2.findNonZero(opening)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        opening = opening[max(0, y - pad):y + h + pad,
                          max(0, x - pad):x + w + pad]

    # Upscale
    upscaled = cv2.resize(opening, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

    final = cv2.bitwise_not(upscaled)

    processed_path = path.replace(".", "_ocr_ready.")
    cv2.imwrite(processed_path, final)
    return processed_path


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_text_with_groq(image_path):
    try:
        image_data = encode_image_to_base64(image_path)

        ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        prompt = """You are an expert in reading handwritten Malayalam text.

Respond ONLY in JSON:
{
  "language": "",
  "is_malayalam": true/false,
  "text": ""
}"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=512,
            temperature=0.1
        )

        response_text = response.choices[0].message.content.strip()

        # Clean markdown if present
        if response_text.startswith("```"):
            parts = response_text.split("```")
            response_text = parts[1] if len(parts) > 1 else response_text
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        return {
            "text": result.get("text", ""),
            "is_malayalam": result.get("is_malayalam", False),
            "language": result.get("language", "Unknown"),
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

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            processed_path = preprocess_for_handwriting(image_path)

            if processed_path is None:
                error_msg = "Error: Could not read image."
            else:
                # ✅ FIXED: use processed image
                result = detect_text_with_groq(processed_path)

                if result["error"]:
                    error_msg = result["error"]

                elif result["is_malayalam"]:
                    text = result["text"] or "No clear text detected."

                else:
                    error_msg = f"Not Malayalam. Detected: {result['language']}"

        else:
            error_msg = "Please upload an image."

   return render_template(
    "index.html",
    text=text,
    image=os.path.basename(image_path) if image_path else "",
    processed=os.path.basename(processed_path) if processed_path else "",
    error=error_msg
)


if __name__ == "__main__":
    app.run(debug=True)

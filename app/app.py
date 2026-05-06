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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def preprocess_for_handwriting(path):
    """
    Gentle preprocessing: only denoise + upscale.
    Avoid aggressive morphological ops that break Malayalam character strokes.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    # Check if image is mostly white/light background (typical scan)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # If image is already clean (bright background), just upscale
    if mean_brightness > 180:
        # Mild sharpening only
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        upscaled = cv2.resize(sharpened, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
    else:
        # Darker/noisy image: denoise then upscale
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        upscaled = cv2.resize(denoised, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)

    name, ext = os.path.splitext(path)
    processed_path = f"{name}_ocr_ready{ext}"
    cv2.imwrite(processed_path, upscaled)
    return processed_path


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_text_with_groq(original_path, processed_path):
    """
    Send BOTH original and processed images for better accuracy.
    Use a strict transcription prompt.
    """
    try:
        if not GROQ_API_KEY:
            return {
                "text": "",
                "is_malayalam": False,
                "language": "Unknown",
                "error": "GROQ_API_KEY not set"
            }

        # Prefer original image — preprocessing can distort strokes
        image_data = encode_image_to_base64(original_path)

        ext = os.path.splitext(original_path)[1].lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        # Strict transcription prompt — prevents hallucination
        prompt = """You are a Malayalam handwriting transcription expert.

Your ONLY job is to read and transcribe EXACTLY what is written in this image.

STRICT RULES:
- Transcribe the Malayalam text character by character, exactly as written
- Do NOT translate, interpret, or guess meaning
- Do NOT add words that are not clearly visible
- Do NOT replace or substitute characters — copy them exactly
- If a word is unclear, write your best reading of those exact characters
- Respond ONLY in this exact JSON format, nothing else:

{
  "language": "Malayalam",
  "is_malayalam": true,
  "text": "<exact transcription here>"
}

If the image contains no Malayalam text:
{
  "language": "<detected language>",
  "is_malayalam": false,
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
            temperature=0.0  # Zero temperature = no creativity, pure transcription
        )

        response_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response_text = part
                    break

        # Find JSON object in response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            response_text = response_text[start:end]

        result = json.loads(response_text)

        return {
            "text": result.get("text", "").strip(),
            "is_malayalam": result.get("is_malayalam", False),
            "language": result.get("language", "Unknown"),
            "error": None
        }

    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract text directly
        return {
            "text": "",
            "is_malayalam": False,
            "language": "Unknown",
            "error": f"JSON parse error: {str(e)} | Raw: {response_text[:200]}"
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
                # Pass both original and processed paths
                result = detect_text_with_groq(image_path, processed_path)

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

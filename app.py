import os
import io
from flask import Flask, request, send_file, jsonify
import torch
from diffusers import StableDiffusionPipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -----------------------------
#       Загрузка модели
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models/anythingV3_fp16.safetensors")

pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    use_safetensors=True,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# -----------------------------
#       Эндпоинт для генерации
# -----------------------------
@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    prompt = data["prompt"]

    try:
        image = pipe(prompt).images[0]

        # Сохраняем в буфер
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="result.png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

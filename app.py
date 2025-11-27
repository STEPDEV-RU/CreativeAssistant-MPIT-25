import io
import os
from flask import Flask, request, send_file, jsonify
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from transformers import MarianTokenizer, MarianMTModel
from flask_cors import CORS

# app init
app = Flask(__name__)
CORS(app)

# ============================================
#   LOAD LOCAL KANDINSKY 2.2 (prior + decoder)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models/kandinsky-2-2")
TRANSLATE_MODEL_DIR = os.path.join(BASE_DIR, "models/opus-mt-ru-en")

print("Loading KANDINSKY 2.2 PRIOR ...")
pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    os.path.join(MODEL_DIR, "prior"),
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

print("Loading KANDINSKY 2.2 DECODER ...")
pipe_decoder = KandinskyV22Pipeline.from_pretrained(
    os.path.join(MODEL_DIR, "decoder"),
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

print("Loading OPUS MT RU-EN ...")

translator_tokenizer = MarianTokenizer.from_pretrained(TRANSLATE_MODEL_DIR, local_files_only=True)
translator_model = MarianMTModel.from_pretrained(TRANSLATE_MODEL_DIR, local_files_only=True)

print("Models loaded successfully!")

# вспомогательная ф-ция для перевода русского промпта
def translate_ru_to_en(text: str):
    tokens = translator_tokenizer(text, return_tensors="pt", padding=True)
    translated = translator_model.generate(**tokens)
    return translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# ============================================
#   API ENDPOINT
# ============================================

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    prompt = data["prompt"] # todo сделать проверку, вдруг это инглиш, тогда переводить не следует
    prompt_en = translate_ru_to_en(prompt)
    negative_prompt = data.get("negative_prompt", "low quality, bad quality")

    print(f"Generating image from prompt: {prompt} (EN: {prompt_en})")

    try:
        # 1) PRIOR — текст → latent embeddings
        prior_out = pipe_prior(
            prompt=prompt_en,
            negative_prompt=negative_prompt
        )

        # 2) DECODER — latent → изображение
        image = pipe_decoder(
            **prior_out,
            num_inference_steps=20,
            # prior_guidance_scale=1.0,
            height=512,
            width=512
        ).images[0]

        # save to memory buffer
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        print("Generation OK")

        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="result.png")

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

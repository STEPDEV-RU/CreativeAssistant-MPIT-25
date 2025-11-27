import random

from flask import Blueprint, request, jsonify
from .model_manager import ModelManager
from .generator import ImageGenerator
from random import randint

api_bp = Blueprint("api", __name__)

model_manager = ModelManager()
generator = ImageGenerator(model_manager)

# -----------------------------------------
# МОДЕЛИ
# -----------------------------------------
@api_bp.route("/update_model_list", methods=["GET"])
def update_model_list():
    return jsonify(model_manager.update_model_list())

@api_bp.route("/models", methods=["GET"])
def view_models():
    return jsonify(model_manager.view_models())

@api_bp.route("/load_model/<uid>", methods=["POST"])
def load_model(uid):
    try:
        model_manager.load_model(uid)
        return jsonify({"loaded_model": uid})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route("/unload_model", methods=["POST"])
def unload_model():
    try:
        model_manager.unload_model()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route("/reload_model", methods=["POST"])
def reload_model():
    try:
        uid = model_manager.reload_model()
        return jsonify({"reloaded_model": uid})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route("/loaded_model", methods=["GET"])
def loaded_model():
    return jsonify({"uid": model_manager.loaded_model()})

# -----------------------------------------
# ГЕНЕРАЦИЯ
# -----------------------------------------
@api_bp.route("/generate", methods=["POST"])
def generate():
    data = request.json


    try:
        res = generator.generate_image(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            inference_steps=data.get("inference_steps", 20),
            guidance_scale=data.get("guidance_scale", 7.5),
            seed=data.get("seed", None),
            height=data.get("height", 512),
            width=data.get("width", 512),
            count=data.get("count", 1)
        )


        return jsonify(res)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
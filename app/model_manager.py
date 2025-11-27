import os
import uuid
import json
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from config import Config

class ModelManager:

    INDEX_FILE = "models.index"

    def __init__(self):
        self.models = {}
        self.loaded_model_uid = None
        self.pipeline = None
        self._load_index()

    # -----------------------
    # Работа с индексом
    # -----------------------
    def _index_path(self):
        return os.path.join(Config.MODELS_PATH, self.INDEX_FILE)

    def _load_index(self):
        """Загружаем индекс из файла"""
        try:
            with open(self._index_path(), "r") as f:
                self.models = json.load(f)
        except FileNotFoundError:
            self.models = {}

    def _save_index(self):
        """Сохраняем индекс в файл"""
        with open(self._index_path(), "w") as f:
            json.dump(self.models, f, indent=2)

    # -----------------------
    # Работа с моделями
    # -----------------------
    def update_model_list(self):
        """Обновляем индекс моделей на основе текущих файлов"""
        updated = False
        current_files = {
            f for f in os.listdir(Config.MODELS_PATH)
            if f.endswith(".safetensors") and os.path.isfile(os.path.join(Config.MODELS_PATH, f))
        }

        # Добавляем новые модели
        for filename in current_files:
            if filename not in [m["filename"] for m in self.models.values()]:
                uid = str(uuid.uuid4())
                self.models[uid] = {
                    "filename": filename,
                    "filesize": os.path.getsize(os.path.join(Config.MODELS_PATH, filename))
                }
                updated = True

        # Удаляем отсутствующие модели
        to_delete = [
            uid for uid, m in self.models.items()
            if m["filename"] not in current_files
        ]
        for uid in to_delete:
            del self.models[uid]
            updated = True

        if updated:
            self._save_index()

        return self.models

    def view_models(self):
        return self.models

    def unload_model(self):
        self.pipeline = None
        self.loaded_model_uid = None
        torch.cuda.empty_cache()

    def reload_model(self):
        if self.loaded_model_uid is None:
            raise RuntimeError("No model loaded")
        uid = self.loaded_model_uid
        self.unload_model()
        return self.load_model(uid)

    def loaded_model(self):
        return self.loaded_model_uid

    def get_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("No model is loaded")
        return self.pipeline

    def load_model(self, uid: str):
        if uid not in self.models:
            raise ValueError("Invalid model UID")

        self.unload_model()

        filename = self.models[uid]["filename"]
        path = os.path.join(Config.MODELS_PATH, filename)

        print(f"[INFO] Loading local model: {path}")

        # читаем header safetensors
        weight_data = load_file(path)

        # определяем SDXL
        is_sdxl = any(
            "text_encoder_2" in k
            or "add_time_cond" in k
            or "pooled" in k
            for k in weight_data.keys()
        )

        device = torch.device(Config.device())

        try:
            if is_sdxl:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=Config.TORCH_DTYPE
                )
            else:
                pipe = StableDiffusionPipeline.from_single_file(
                    path,
                    torch_dtype=Config.TORCH_DTYPE
                )

            if Config.USE_XFORMERS and device.type == "cuda":
                pipe.enable_xformers_memory_efficient_attention()

            pipe.to(device)

            self.pipeline = pipe
            self.loaded_model_uid = uid

            print("[INFO] Local model loaded successfully.")
            return uid

        except Exception as e:
            print(f"[ERROR] {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    # -----------------------
    # Генерация уникальных имён изображений
    # -----------------------
    def generate_image_filename(self):
        from datetime import datetime
        import random
        now = datetime.now()
        hh_mm = now.strftime("%H_%M")
        random_id = random.randint(0, 999)
        return f"{hh_mm}_{random_id}.png"

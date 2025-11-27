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

    # =====================================================================
    #                           INDEX
    # =====================================================================

    def _index_path(self):
        return os.path.join(Config.MODELS_PATH, self.INDEX_FILE)

    def _load_index(self):
        try:
            with open(self._index_path(), "r") as f:
                self.models = json.load(f)
        except FileNotFoundError:
            self.models = {}

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump(self.models, f, indent=2)

    # =====================================================================
    #                   INDEX SCAN + MODEL DETECTION
    # =====================================================================

    def update_model_list(self):
        """Сканируем содержимое папки models/ и обновляем индекс."""

        updated = False
        base = Config.MODELS_PATH

        file_items = []
        dir_items = []

        for entry in os.listdir(base):
            full = os.path.join(base, entry)

            # --- одиночный safetensors файл ---
            if os.path.isfile(full) and entry.endswith(".safetensors"):
                file_items.append(entry)

            # --- директория модели ---
            elif os.path.isdir(full):
                # если есть model_index.json, это папочная модель
                index_file = os.path.join(full, "model_index.json")
                if os.path.isfile(index_file):
                    dir_items.append(entry)
                # иначе можно проверить наличие хотя бы одного .safetensors
                elif any(f.endswith(".safetensors") for f in os.listdir(full)):
                    dir_items.append(entry)

        # ----------------------------
        # Добавляем новые модели
        # ----------------------------
        # одиночные файлы
        for f in file_items:
            exists = any(info.get("filename") == f and info.get("dirname") is None for info in self.models.values())
            if not exists:
                uid = str(uuid.uuid4())
                self.models[uid] = {
                    "type": "file",
                    "filename": f,
                    "dirname": None,
                    "filesize": os.path.getsize(os.path.join(base, f)),
                    "loader": None
                }
                updated = True

        # папочные модели
        for d in dir_items:
            exists = any(info.get("dirname") == d for info in self.models.values())
            if not exists:
                uid = str(uuid.uuid4())
                self.models[uid] = {
                    "type": "directory",
                    "dirname": d,
                    "loader": None,
                    "filesize": 0  # размер можно не учитывать для папки
                }
                updated = True

        # ----------------------------
        # Удаляем записи, которых больше нет
        # ----------------------------
        to_remove = []
        for uid, info in self.models.items():
            if info.get("type") == "file":
                if info.get("filename") not in file_items:
                    to_remove.append(uid)
            else:  # directory
                if info.get("dirname") not in dir_items:
                    to_remove.append(uid)

        for uid in to_remove:
            del self.models[uid]
            updated = True

        # ----------------------------
        # Сохраняем индекс при изменениях
        # ----------------------------
        if updated:
            self._save_index()

        return self.models

    def view_models(self):
        return self.models

    # =====================================================================
    #                            LOADING
    # =====================================================================

    def unload_model(self):
        self.pipeline = None
        self.loaded_model_uid = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

    def reload_model(self):
        if self.loaded_model_uid is None:
            raise RuntimeError("No model loaded")
        uid = self.loaded_model_uid
        self.unload_model()
        return self.load_model(uid)

    def loaded_model(self):
        return self.loaded_model_uid

    def get_pipeline(self):
        if not self.pipeline:
            raise RuntimeError("No model is loaded")
        return self.pipeline

    def load_model(self, uid: str):
        if uid not in self.models:
            raise ValueError("Invalid model UID")

        self.unload_model()

        info = self.models[uid]
        item_type = info["type"]
        device = torch.device(Config.device())

        path = os.path.join(Config.MODELS_PATH, info.get("dirname") or info.get("filename"))
        weight_data = None

        from app.customload import CUSTOM_LOADERS

        # =============================
        # 1. ФАЙЛОВАЯ МОДЕЛЬ (.safetensors)
        # =============================
        if item_type == "file":
            weight_data = load_file(path)

            # SD / SDXL авто-детект
            try:
                is_sdxl = any(
                    "text_encoder_2" in k
                    or "add_time_cond" in k
                    or "pooled" in k
                    for k in weight_data.keys()
                )

                if is_sdxl:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        path,
                        torch_dtype=Config.TORCH_DTYPE
                    )
                    info["loader"] = "SDXL_BUILTIN"
                else:
                    pipe = StableDiffusionPipeline.from_single_file(
                        path,
                        torch_dtype=Config.TORCH_DTYPE
                    )
                    info["loader"] = "SD15_BUILTIN"

                if Config.USE_XFORMERS and device.type == "cuda":
                    pipe.enable_xformers_memory_efficient_attention()

                pipe.to(device)
                self.pipeline = pipe
                self.loaded_model_uid = uid
                self._save_index()
                print("[INFO] File model loaded successfully.")
                return uid

            except Exception as e:
                raise RuntimeError(f"File model loading failed: {e}")

        # =============================
        # 2. ПАПОЧНАЯ МОДЕЛЬ (customload)
        # =============================
        elif item_type == "directory":
            found = False
            for loader_cls in CUSTOM_LOADERS.values():
                loader = loader_cls()
                try:
                    if loader.can_load(None, info["dirname"]):
                        print(f"[INFO] Using custom loader: {loader_cls.__name__}")
                        pipe = loader.load(path, device, Config)

                        self.pipeline = pipe
                        self.loaded_model_uid = uid

                        info["loader"] = loader_cls.__name__
                        self._save_index()
                        found = True
                        break
                except Exception as e:
                    print(f"[ERROR] Custom loader {loader_cls.__name__} failed: {e}")

            if not found:
                raise RuntimeError("No suitable custom loader found for this directory model")

            print("[INFO] Directory model loaded successfully.")
            return uid

        else:
            raise RuntimeError(f"Unknown model type: {item_type}")

    # =====================================================================
    #                         IMAGE NAME GENERATOR
    # =====================================================================

    def generate_image_filename(self):
        from datetime import datetime
        import random
        now = datetime.now()
        hh_mm = now.strftime("%H_%M")
        random_id = random.randint(0, 999)
        return f"{hh_mm}_{random_id}.png"

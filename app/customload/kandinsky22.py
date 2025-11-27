import os
import torch
from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel
from app.customload.base import BaseModelLoader, ModelLoadError
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import logging

log = logging.getLogger("kandinsky22")


class Kandinsky22Loader(BaseModelLoader):
    model_name = "kandinsky22"

    local_paths = [
        "models/kandinsky-2-2"
    ]

    @classmethod
    def can_load(cls, weight_data, model_id: str) -> bool:
        return "kandinsky" in model_id.lower()

    def load(self, path, device, config):
        """Метод для совместимости с ModelManager"""
        loaded = self._load_from_local(path)

        # Переносим все на device, если нужно
        loaded["prior"].to(device)
        loaded["decoder"].to(device)
        # переводчик оставляем на CPU

        return loaded

    def _load_from_local(self, path: Path):
        """Загружаем модель из локального каталога"""

        log.info(f"[kandinsky22] Загрузка локальной модели: {path}")

        if not path.is_dir():
            raise ModelLoadError(f"Ожидалась папка модели, а получен файл: {path}")

        # --------------------------
        # Загружаем Prior
        # --------------------------
        prior_path = path / "prior"
        if not prior_path.exists():
            raise ModelLoadError(f"Не найдена папка prior: {prior_path}")

        log.info("Loading KANDINSKY 2.2 PRIOR ...")
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            prior_path,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to("cuda")

        # --------------------------
        # Загружаем Decoder
        # --------------------------
        decoder_path = path / "decoder"
        if not decoder_path.exists():
            raise ModelLoadError(f"Не найдена папка decoder: {decoder_path}")

        log.info("Loading KANDINSKY 2.2 DECODER ...")
        pipe_decoder = KandinskyV22Pipeline.from_pretrained(
            decoder_path,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to("cuda")

        # --------------------------
        # Загружаем переводчик OPUS MT RU-EN
        # --------------------------
        translator_path = Path("models/opus-mt-ru-en")
        if not translator_path.exists():
            raise ModelLoadError(f"Не найдена папка перевода: {translator_path}")

        log.info("Loading OPUS MT RU-EN ...")
        tokenizer = MarianTokenizer.from_pretrained(translator_path, local_files_only=True)
        model = MarianMTModel.from_pretrained(translator_path, local_files_only=True)

        log.info("Kandinsky 2.2 loaded successfully!")

        return {
            "prior": pipe_prior,
            "decoder": pipe_decoder,
            "translator": {
                "tokenizer": tokenizer,
                "model": model
            }
        }

import os
import torch
from datetime import datetime
from config import Config
from .utils import ensure_dir
from .utils import clear_generated_folder


class ImageGenerator:

    def __init__(self, model_manager):
        self.mm = model_manager

    def _get_daily_folder(self):
        folder = datetime.now().strftime("%d.%m.%Y")
        target = os.path.join(Config.GENERATE_PATH, folder)
        ensure_dir(target)
        return target

    def generate_image(self, prompt, negative_prompt, inference_steps,
                       guidance_scale, seed, height, width, count):

        pipe = self.mm.get_pipeline()
        device = Config.device()

        clear_generated_folder()

        ensure_dir(os.path.join('static/', Config.PUBLIC_PATH))

        images = []

        generator = torch.Generator(device)
        if seed is not None:
            generator.manual_seed(int(seed))

        daily_folder = self._get_daily_folder()

        for _ in range(count):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            img = result.images[0]

            filename = self.mm.generate_image_filename()
            path = os.path.join(daily_folder, filename)
            img.save(path)

            public_path = os.path.join('static/', Config.PUBLIC_PATH, filename)
            if not os.path.exists(public_path):
                os.link(path, public_path)

            images.append(f"/generated/{filename}")

        return {
            "status": "ok",
            "images": images
        }

import os
import torch
from diffusers import StableDiffusionPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models/anythingV3_fp16.safetensors")

# Загрузка модели из одного файла safetensors
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    use_safetensors=True,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# -----------------------------
#       ГЕНЕРАЦИЯ СХЕМЫ
# -----------------------------

prompt = (
    "simple diagram"
)

image = pipe(prompt).images[0]

output_path = os.path.join(current_dir, "schematic_result.png")
image.save(output_path)

print(f"Готово! Изображение сохранено: {output_path}")

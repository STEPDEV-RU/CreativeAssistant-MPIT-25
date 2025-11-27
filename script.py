import os
import subprocess
from diffusers import StableDiffusionPipeline
import torch

# --- Шаг 1: Определяем пути ---
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
kandinsky_dir = os.path.join(models_dir, "kandinsky")

# Создаём папки, если их нет
os.makedirs(kandinsky_dir, exist_ok=True)

# --- Шаг 2: Клонируем репозиторий модели ---
repo_url = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"

# Проверяем, не клонирован ли уже репозиторий
if not os.path.exists(os.path.join(kandinsky_dir, ".git")):
    print("Клонируем модель...")
    subprocess.run(["git", "clone", repo_url, kandinsky_dir], check=True)
else:
    print("Репозиторий уже клонирован.")

# --- Шаг 3: Загружаем модель через diffusers ---
print("Загружаем модель локально...")
pipe = StableDiffusionPipeline.from_pretrained(
    kandinsky_dir,
    torch_dtype=torch.float16,
    safety_checker=None
)

# Переносим на GPU (если доступен)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("CUDA не доступна, используем CPU.")

# --- Шаг 4: Генерация тестового изображения ---
prompt = "A beautiful futuristic cityscape, digital art"
image = pipe(prompt).images[0]

# --- Шаг 5: Сохраняем результат ---
output_path = os.path.join(current_dir, "output.png")
image.save(output_path)
print(f"Изображение сохранено в {output_path}")

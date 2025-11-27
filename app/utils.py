import os
import shutil
from config import Config

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_generated_folder():
    folder = os.path.join('static', Config.PUBLIC_PATH)
    if os.path.exists(folder):
        # удаляем все файлы и подпапки
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # удаляем файл или ссылку
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # удаляем папку рекурсивно
            except Exception as e:
                print(f"[WARNING] Не удалось удалить {file_path}: {e}")
    else:
        os.makedirs(folder, exist_ok=True)
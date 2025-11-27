from flask import Blueprint, send_from_directory
import os
from config import Config

static_bp = Blueprint('generated', __name__)

@static_bp.route('/<filename>')
def generated_files(filename):
    # Абсолютный путь к папке static/generated
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', Config.PUBLIC_PATH))
    print("Serving from folder:", folder)
    print("Filename:", filename)
    return send_from_directory(folder, filename)
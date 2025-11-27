from flask import Flask
from .routes import api_bp
from .static import static_bp

def create_app():
    app = Flask(__name__)

    # подключаем blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(static_bp, url_prefix="/generated")

    return app

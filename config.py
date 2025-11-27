import torch

class Config:

    # ========================
    #  DEVICE SETTINGS
    # ========================
    USE_GPU = True    # если False → использовать CPU
    FORCE_CPU = False # если True → отключает CUDA даже если доступна

    @staticmethod
    def device():
        if Config.FORCE_CPU:
            return "cpu"
        if Config.USE_GPU and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # Precision / dtype
    TORCH_DTYPE = torch.float16   # на CPU можно поставить float32

    # Xformers ускорение
    USE_XFORMERS = True

    # ========================
    #  PATHS
    # ========================
    MODELS_PATH = "models/"
    GENERATE_PATH = "generates/"
    TMP_PATH = "generates/tmp/"
    PUBLIC_PATH = "generated/" # in static/

    # ========================
    #  GENERATION DEFAULTS
    # ========================
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    MAX_RESOLUTION = 2048

    # ========================
    #  API SETTINGS
    # ========================
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    DEBUG = False

    # ========================
    #  QUEUE POLICY / LIMITS
    # ========================
    MAX_QUEUE = 1      # 1 клиент — 1 модель, без очередей

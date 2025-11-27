#!/usr/bin/env python3
import argparse
import requests
import json
import sys
import random
from pathlib import Path

API_URL = "http://127.0.0.1:5000/api"


def safe_request(method, endpoint, **kwargs):
    """Выполняет HTTP-запрос с обработкой всех ошибок."""
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
    except requests.exceptions.Timeout:
        print("❌ Ошибка: сервер слишком долго не отвечает (timeout).")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения: сервер недоступен.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Неизвестная ошибка запроса: {e}")
        sys.exit(1)

    try:
        data = response.json()
    except ValueError:
        print("❌ Ошибка: сервер вернул не JSON.")
        print("Ответ:", response.text)
        sys.exit(1)

    if response.status_code >= 400:
        print(f"❌ Ошибка API ({response.status_code}): {data}")
        sys.exit(1)

    return data


def action_update_models(args):
    data = safe_request("GET", "/update_model_list")
    print(json.dumps(data, indent=4, ensure_ascii=False))


def action_view_models(args):
    data = safe_request("GET", "/models")
    print(json.dumps(data, indent=4, ensure_ascii=False))


def action_load_model(args):
    data = safe_request("POST", f"/load_model/{args.uid}")
    print(json.dumps(data, indent=4, ensure_ascii=False))


def action_loaded_model(args):
    data = safe_request("GET", "/loaded_model")
    print(json.dumps(data, indent=4, ensure_ascii=False))

def action_unload_model(args):
    data = safe_request("POST", "/unload_model")
    print(json.dumps(data, indent=4, ensure_ascii=False))


def action_generate(args):
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "inference_steps": args.steps,
        "guidance_scale": args.scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "count": args.count
    }

    data = safe_request("POST", "/generate", json=payload)
    print(json.dumps(data, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="CLI клиент для Image Generation API"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # update models
    subparsers.add_parser("update", help="Обновить список моделей") \
        .set_defaults(func=action_update_models)

    # list models
    subparsers.add_parser("models", help="Показать список моделей") \
        .set_defaults(func=action_view_models)

    # load model
    load = subparsers.add_parser("load", help="Загрузить модель")
    load.add_argument("uid", type=str, help="UID модели")
    load.set_defaults(func=action_load_model)

    # unload model
    subparsers.add_parser("unload", help="Выгрузить загруженную модель") \
        .set_defaults(func=action_unload_model)

    # show loaded model
    subparsers.add_parser("loaded", help="Показать загруженную модель") \
        .set_defaults(func=action_loaded_model)

    # generate
    gen = subparsers.add_parser("gen", help="Генерация изображения")
    gen.add_argument("--prompt", "-p", required=True)
    gen.add_argument("--negative", "-n", default="")
    gen.add_argument("--steps", "-s", default=30, type=int)
    gen.add_argument("--scale", "-g", default=7.5, type=float)
    gen.add_argument("--seed", default=random.randint(0, 9999999))
    gen.add_argument("--height", default=512, type=int)
    gen.add_argument("--width", default=512, type=int)
    gen.add_argument("--count", default=1, type=int)
    gen.set_defaults(func=action_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

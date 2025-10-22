import toml
import os
import sys

def load_secrets(config_path="secrets.toml"):
    if not os.path.exists(config_path):
        print(f"❌ Файл {config_path} не найден!")
        print("📝 Создайте файл secrets.toml с вашими API-ключами.")
        print("Пример содержимого:")
        print("""
OPENROUTER_API_KEY = "ваш_ключ"
ASSEMBLYAI_API_KEY = "ваш_ключ"

[api_keys]
OPENROUTER_API_KEY = "ваш_ключ"
ASSEMBLYAI_API_KEY = "ваш_ключ"
        """)
        sys.exit(1)
    
    try:
        config = toml.load(config_path)
        api_keys = config.get("api_keys", {})
        
        return {
            "OPENROUTER_API_KEY": api_keys.get("OPENROUTER_API_KEY", config.get("OPENROUTER_API_KEY")),
            "ASSEMBLYAI_API_KEY": api_keys.get("ASSEMBLYAI_API_KEY", config.get("ASSEMBLYAI_API_KEY")),
            "model_preference": config.get("medical_analyzer", {}).get("model_preference"),
            "timeout": config.get("medical_analyzer", {}).get("timeout", 90),
            "max_retries": config.get("medical_analyzer", {}).get("max_retries", 2)
        }
    except Exception as e:
        print(f"❌ Ошибка чтения {config_path}: {e}")
        sys.exit(1)

secrets = load_secrets()
OPENROUTER_API_KEY = secrets["OPENROUTER_API_KEY"]
ASSEMBLYAI_API_KEY = secrets["ASSEMBLYAI_API_KEY"]

# В конце config.py
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "ваш_ключ":
    print("⚠️ OPENROUTER_API_KEY не настроен!")
    
if not ASSEMBLYAI_API_KEY or ASSEMBLYAI_API_KEY == "ваш_ключ":
    print("⚠️ ASSEMBLYAI_API_KEY не настроен!")


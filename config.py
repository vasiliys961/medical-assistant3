import toml
import os

def load_secrets(config_path="secrets.toml"):
    if os.path.exists(config_path):
        config = toml.load(config_path)
        # Прямые ключи для совместимости
        api_keys = config.get("api_keys", {})
        # Современный стиль: весь конфиг через config["section"]["param"]
        return {
            "OPENROUTER_API_KEY": api_keys.get("OPENROUTER_API_KEY", config.get("OPENROUTER_API_KEY")),
            "ASSEMBLYAI_API_KEY": api_keys.get("ASSEMBLYAI_API_KEY", config.get("ASSEMBLYAI_API_KEY")),
            "model_preference": config.get("medical_analyzer", {}).get("model_preference"),
            "timeout": config.get("medical_analyzer", {}).get("timeout", 90),
            "max_retries": config.get("medical_analyzer", {}).get("max_retries", 2)
        }
    else:
        raise FileNotFoundError("secrets.toml not found! Place it next to your main file.")

# Использование:
secrets = load_secrets()
OPENROUTER_API_KEY = secrets["OPENROUTER_API_KEY"]
ASSEMBLYAI_API_KEY = secrets["ASSEMBLYAI_API_KEY"]

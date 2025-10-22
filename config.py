import toml
import os
import sys

def load_secrets(config_path="secrets.toml"):
    # Приоритет: сначала переменные окружения (для Railway/облака)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    # Если переменные окружения найдены — используем их
    if openrouter_key and assemblyai_key:
        return {
            "OPENROUTER_API_KEY": openrouter_key,
            "ASSEMBLYAI_API_KEY": assemblyai_key,
            "model_preference": os.getenv("MODEL_PREFERENCE", "anthropic/claude-3-5-sonnet-20241022"),
            "timeout": int(os.getenv("TIMEOUT", "180")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3"))
        }
    
    # Иначе пробуем загрузить из secrets.toml (для локальной разработки)
    if not os.path.exists(config_path):
        print(f"❌ Файл {config_path} не найден!")
        print("📝 Создайте файл secrets.toml с вашими API-ключами или установите переменные окружения.")
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

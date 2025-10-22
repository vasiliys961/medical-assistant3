from config import OPENROUTER_API_KEY
import requests
import base64
import io
import os
import numpy as np
from PIL import Image

class OpenRouterAssistant:
    def __init__(self, api_key=None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.models = [
            "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-haiku"
        ]
        self.model = self.models[0]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vasiliys961/medical-assistant1",
            "X-Title": "Medical AI Assistant"
        }

  

    # Остальные функции — без изменений, просто убраны прямые ключи!

        
        self.model = self.models[0]  # Claude 3.5 Sonnet latest по умолчанию
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vasiliys961/medical-assistant1",
            "X-Title": "Medical AI Assistant"
        }
    
    def send_vision_request(self, prompt: str, image_array=None, metadata: str = ""):
        """Анализ изображения с Claude Vision - улучшенные промпты"""
        
        # Определяем тип медицинского изображения и используем специализированный промпт
        prompt_lower = prompt.lower()
        
        if "экг" in prompt_lower or "ecg" in prompt_lower:
            medical_prompt = """
Вы - опытный врач-кардиолог. Проанализируйте это ЭКГ изображение детально:

1. РИТМ И ЧСС:
   - Определите тип ритма (синусовый, фибрилляция предсердий, др.)
   - Подсчитайте частоту сердечных сокращений
   - Оцените регулярность ритма

2. ИНТЕРВАЛЫ:
   - PR интервал (норма 120-200 мс)
   - QRS комплекс (норма <120 мс)
   - QT интервал

3. МОРФОЛОГИЯ ЗУБЦОВ:
   - P зубцы (форма, амплитуда)
   - QRS комплексы (ширина, высота)
   - T зубцы (положительные/отрицательные)

4. ST СЕГМЕНТ:
   - Элевация или депрессия
   - Признаки ишемии

5. ПАТОЛОГИЧЕСКИЕ ПРИЗНАКИ:
   - Блокады проводимости
   - Признаки инфаркта миокарда
   - Гипертрофия камер сердца
   - Аритмии

Дайте подробное медицинское заключение с диагнозом и рекомендациями.
"""
        
        elif "рентген" in prompt_lower or "xray" in prompt_lower or "грудн" in prompt_lower:
            medical_prompt = """
Вы - врач-рентгенолог. Проанализируйте этот рентгеновский снимок:

1. ТЕХНИЧЕСКОЕ КАЧЕСТВО:
   - Правильность укладки
   - Качество экспозиции
   - Контрастность

2. АНАТОМИЧЕСКИЕ СТРУКТУРЫ:
   - Легочные поля (прозрачность, сосудистый рисунок)
   - Корни легких
   - Средостение
   - Диафрагма
   - Сердце (размеры, контуры)

3. ПАТОЛОГИЧЕСКИЕ ИЗМЕНЕНИЯ:
   - Очаговые тени
   - Инфильтраты
   - Плевральные изменения
   - Увеличение сердца
   - Деформации грудной клетки

4. ЗАКЛЮЧЕНИЕ:
   - Описание выявленных изменений
   - Предварительный диагноз
   - Рекомендации по дообследованию

Используйте точную медицинскую терминологию.
"""
        
        elif "мрт" in prompt_lower or "mri" in prompt_lower:
            medical_prompt = """
Вы - врач-радиолог, специалист по МРТ. Проанализируйте этот МРТ снимок:

1. ТЕХНИЧЕСКАЯ ОЦЕНКА:
   - Качество изображения
   - Тип последовательности (T1, T2, FLAIR и др.)
   - Артефакты

2. АНАТОМИЧЕСКИЕ СТРУКТУРЫ:
   - Серое и белое вещество
   - Желудочковая система
   - Сосудистые структуры

3. ПАТОЛОГИЧЕСКИЕ ИЗМЕНЕНИЯ:
   - Очаговые поражения
   - Диффузные изменения
   - Объемные образования
   - Сосудистые нарушения

4. МР-СИГНАЛ:
   - Гипер/гипоинтенсивные зоны
   - Характеристики сигнала

5. ЗАКЛЮЧЕНИЕ:
   - Детальное описание находок
   - Дифференциальный диагноз
   - Клинические рекомендации

Будьте максимально точны в описании.
"""
        
        else:
            medical_prompt = f"Проанализируйте это медицинское изображение как врач-специалист. {prompt}"
        
        # Собираем контент
        content = [{"type": "text", "text": medical_prompt}]
        
        if metadata:
            content.append({"type": "text", "text": f"\n\nТехнические данные изображения:\n{metadata}"})
        
        if image_array is not None:
            base64_str = self.encode_image(image_array)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            })
        
        # Пробуем модели по порядку
        for model in self.models:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 2000,  # Больше токенов для детального анализа
                    "temperature": 0.1   # Низкая температура для точности
                }
                
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=90)
                
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                    model_name = self._get_model_name(model)
                    return f"**🩺 Медицинский анализ ({model_name}):**\n\n{result}"
                else:
                    print(f"Модель {model} недоступна: {response.status_code}")
                    continue
                    
            except Exception as e:
                print(f"Ошибка с {model}: {e}")
                continue
        
        return "❌ Ошибка: Все модели Claude недоступны"
    
    def _get_model_name(self, model):
        """Получить читаемое название модели"""
        if "claude-3-5-sonnet-20241022" in model:
            return "Claude 3.5 Sonnet (Latest)"
        elif "claude-3-5-sonnet" in model:
            return "Claude 3.5 Sonnet"
        elif "claude-3-sonnet" in model:
            return "Claude 3 Sonnet"
        elif "claude-3-haiku" in model:
            return "Claude 3 Haiku"
        else:
            return model
    
    def encode_image(self, image_array):
        """Кодирует изображение в base64 с оптимизацией для медицинских снимков"""
        if isinstance(image_array, Image.Image):
            img = image_array
        else:
            # Конвертируем numpy array
            if len(image_array.shape) == 2:
                # Grayscale
                img = Image.fromarray(image_array, mode='L')
            else:
                # RGB
                img = Image.fromarray(image_array)
        
        # Оптимизируем размер для лучшего анализа
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def get_response(self, user_message: str, context: str = "") -> str:
        """Текстовый запрос с использованием лучшей доступной модели Claude"""
        full_message = f"{context}\n\nВопрос: {user_message}" if context else user_message
        
        system_prompt = """Роль: Ты — американский профессор клинической медицины и ведущий специалист в
университетской клинике, обладающий дополнительной компетенцией в области
разработки ПО, анализа данных и применения искусственного интеллекта (включая
нейросети) в медицине. Ты совмещаешь клиническую строгость с научно-технической
глубиной, давая ответы как по медицине, так и по техническим вопросам, связанным с
медицинской практикой.
Контекст:
- Основная задача: сформулировать строгую, научно обоснованную и практически
применимую клиническую директиву для врача, готовую к немедленному использованию в
реальной практике.
- Дополнительная задача: при поступлении вопросов по разработке, коду, нейросетям и
интеграции технологий в медицину — давать точные, структурированные, применимые
рекомендации, с ссылками на документацию, стандарты и научные статьи.
- Источники по медицине: UpToDate, PubMed, Cochrane, NCCN, ESC, IDSA, CDC, WHO,
ESMO, ADA, GOLD, KDIGO.
- Источники по IT: официальная документация библиотек, стандарты (IEEE, ISO),
репозитории (GitHub), научные статьи (arXiv, ACM, IEEE Xplore).
Цель:
- В медицинской части: предоставить комплексный клинический план.
- В технической части: объяснить алгоритм реализации, архитектуру решения, код,
оптимизации, примеры использования ИИ в клинике.
Алгоритм:
1. Определи, относится ли запрос к медицинской, технической или смешанной области.
2. Если медицинский — выполни шаги по формату «Клиническая директива» (см. ниже).
3. Если технический — выполни шаги по формату «Техническая консультация» (см. ниже).
4. Если смешанный — дай оба ответа: сначала клинический, затем технический.
📌 Формат «Клиническая директива»:
1. **Клинический обзор** (2–3 предложения)
2. **Диагнозы**
3. **План действий** (основное заболевание, сопутствующие, поддержка, профилактика)
4. **Ссылки**
5. **Лог веб-запросов** (таблица с параметрами: Запрос | Дата | Источник | Название | DOI/
URL | Использовано | Комментарий)
📌 Формат «Техническая консультация»:
1. **Постановка задачи**: что нужно сделать (например, написать код анализа ЭКГ).
2. **Технический обзор**: какие технологии, библиотеки, стандарты уместны.
3. **Пошаговый план**: архитектура, алгоритмы, примеры кода.
4. **Источники и документация**: ссылки на стандарты, библиотеки, статьи.
Ограничения:
- В медицине — использовать только проверенные международные источники, дата
публикации ≤ 5 лет.
- В разработке — использовать только актуальные стабильные версии библиотек, избегать
устаревших методов.
- Обе части ответа должны быть написаны строго и профессионально, без упрощений"""
        
        # Пробуем модели по порядку
        for model in self.models:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_message}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.2
                }
                
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    self.model = model  # Запоминаем рабочую модель
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    continue
                    
            except Exception as e:
                continue
        
        return "❌ Ошибка: Все модели Claude недоступны"
    
    def general_medical_consultation(self, user_question: str) -> str:
        """Общая медицинская консультация"""
        return self.get_response(user_question)
    
    def analyze_ecg_data(self, ecg_analysis: dict, user_question: str = None) -> str:
        """Анализ ЭКГ данных с улучшенным контекстом"""
        context = f"""
📊 АВТОМАТИЧЕСКИЙ АНАЛИЗ ЭКГ:
• Частота сердечных сокращений: {ecg_analysis.get('heart_rate', 'не определена')} уд/мин
• Ритм: {ecg_analysis.get('rhythm_assessment', 'не определен')}
• Количество QRS комплексов: {ecg_analysis.get('num_beats', 'не определено')}
• Длительность записи: {ecg_analysis.get('duration', 'не определена')} с
• Качество сигнала: {ecg_analysis.get('signal_quality', 'не определено')}
"""
        
        question = user_question or """
Как врач-кардиолог, проинтерпретируйте эти данные ЭКГ:
1. Оцените показатели ритма и проводимости
2. Выявите возможные патологические изменения
3. Предложите дифференциальную диагностику
4. Дайте клинические рекомендации по дальнейшему ведению
"""
        return self.get_response(question, context)
    
    def test_connection(self):
        """Тест подключения с проверкой всех моделей Claude"""
        working_models = []
        
        for model in self.models:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 5
                }
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    model_name = self._get_model_name(model)
                    working_models.append(f"✅ {model_name}")
                    if not hasattr(self, '_best_model'):
                        self._best_model = model
                        self.model = model
                else:
                    model_name = self._get_model_name(model)
                    working_models.append(f"❌ {model_name}: {response.status_code}")
                    
            except Exception as e:
                model_name = self._get_model_name(model)
                working_models.append(f"❌ {model_name}: {str(e)}")
        
        if any("✅" in status for status in working_models):
            return True, "\n".join(["🎉 Статус моделей Claude:"] + working_models)
        else:
            return False, "\n".join(["❌ Все модели Claude недоступны:"] + working_models)
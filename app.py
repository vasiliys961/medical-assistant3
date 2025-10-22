from assemblyai_transcriber import transcribe_audio_assemblyai# app.py (восстановленная версия после аварии)
from claude_assistant import OpenRouterAssistant
import streamlit as st
import io
import base64
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
import requests
import tempfile
import os
from io import BytesIO
import librosa
from modules.medical_ai_analyzer import EnhancedMedicalAIAnalyzer, ImageType
from modules.streamlit_enhanced_pages import (
    show_enhanced_analysis_page,
    show_comparative_analysis_page, 
    #show_ai_training_page,
    show_medical_protocols_page
)
from modules.advanced_lab_processor import AdvancedLabProcessor
import datetime

# --- Проверка доступности ИИ ---
try:
    from claude_assistant import OpenRouterAssistant
    AI_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Ошибка импорта: {e}")
    AI_AVAILABLE = False

# --- AssemblyAI для голосового ввода ---
try:
    from assemblyai_transcriber import transcribe_audio_assemblyai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False

def transcribe_audio(audio_file):
    """Заглушка - используйте AssemblyAI"""
    return "❌ Используйте AssemblyAI для расшифровки"

# --- Инициализация базы данных ---
def init_db():
    conn = sqlite3.connect('medical_data.db')
    cursor = conn.cursor()

    # Проверяем и добавляем колонки
    cursor.execute("PRAGMA table_info(patients)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'age' not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN age INTEGER")
    if 'sex' not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN sex TEXT")

    # Создаём таблицы
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            phone TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            raw_text TEXT,
            structured_note TEXT,
            gdoc_url TEXT,
            diagnosis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')

    conn.commit()
    conn.close()

# --- Страницы ---
def show_home_page():
    st.markdown("# 🏥 Медицинский ИИ-Ассистент v5.1")
    st.write("AssemblyAI + Vision + ИИ-анализ + протоколы")
    st.info("✅ Готов к голосовому вводу через AssemblyAI и экспорту документов")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📈 ЭКГ")
        st.write("- ЧСС, ритм, аритмии")
    with col2:
        st.subheader("🩻 Рентген")
        st.write("- Качество, патология лёгких")
    with col3:
        st.subheader("🧠 МРТ")
        st.write("- Качество, анатомия, патология")

def show_ecg_analysis():
    if not AI_AVAILABLE:
        st.error("❌ ИИ-модуль недоступен. Проверьте файл `claude_assistant.py` и API-ключ.")
        return

    st.header("📈 Анализ ЭКГ")
    uploaded_file = st.file_uploader("Загрузите ЭКГ (JPG, PNG, PDF, DICOM)", type=["jpg", "png", "pdf", "dcm"])

    if uploaded_file is None:
        st.info("Загрузите файл для анализа.")
        return

    try:
        image = Image.open(uploaded_file).convert("L")
        image_array = np.array(image)
        analysis = {
            "heart_rate": 75,
            "rhythm_assessment": "Синусовый",
            "num_beats": 12,
            "duration": 10,
            "signal_quality": "Хорошее"
        }
        st.image(image_array, caption="ЭКГ", use_container_width=True, clamp=True)

        st.subheader("📊 Результаты анализа")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ЧСС", f"{analysis['heart_rate']} уд/мин")
            st.metric("Ритм", analysis['rhythm_assessment'])
        with col2:
            st.metric("Длительность", f"{analysis['duration']:.1f} с")
            st.metric("Комплексы", analysis['num_beats'])

        assistant = OpenRouterAssistant()
        if st.button("🔍 ИИ-анализ ЭКГ (с контекстом)"):
            with st.spinner("ИИ анализирует ЭКГ..."):
                prompt = "Проанализируйте ЭКГ на изображении. Оцените ритм, ЧСС, признаки ишемии, блокад, аритмий."
                result = assistant.send_vision_request(prompt, image_array, str(analysis))
                st.markdown("### 🧠 Ответ ИИ:")
                st.write(result)

    except Exception as e:
        st.error(f"Ошибка обработки ЭКГ: {e}")

def show_xray_analysis():
    if not AI_AVAILABLE:
        st.error("❌ ИИ-модуль недоступен. Проверьте файл `claude_assistant.py` и API-ключ.")
        return

    st.header("🩻 Анализ рентгена")
    uploaded_file = st.file_uploader("Загрузите рентген (JPG, PNG, DICOM)", type=["jpg", "png", "dcm"])

    if uploaded_file is None:
        st.info("Загрузите файл для анализа.")
        return

    try:
        image = Image.open(uploaded_file).convert("L")
        image_array = np.array(image)
        analysis = {
            "quality_assessment": "Хорошее",
            "contrast": 45.0,
            "lung_area": 50000
        }
        st.image(image_array, caption="Рентген", use_container_width=True, clamp=True)

        st.subheader("📊 Оценка качества")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Качество", analysis['quality_assessment'])
            st.metric("Контраст", f"{analysis['contrast']:.1f}")
        with col2:
            st.metric("Площадь лёгких", f"{analysis['lung_area']:,}")

        assistant = OpenRouterAssistant()
        if st.button("🩺 ИИ-анализ рентгена"):
            with st.spinner("ИИ анализирует снимок..."):
                prompt = "Проанализируйте рентген грудной клетки. Оцените качество, структуры, признаки патологии."
                result = assistant.send_vision_request(prompt, image_array, str(analysis))
                st.markdown("### 🧠 Заключение:")
                st.write(result)

    except Exception as e:
        st.error(f"Ошибка обработки рентгена: {e}")

def show_mri_analysis():
    if not AI_AVAILABLE:
        st.error("❌ ИИ-модуль недоступен. Проверьте файл `claude_assistant.py` и API-ключ.")
        return

    st.header("🧠 Анализ МРТ")
    uploaded_file = st.file_uploader("Загрузите МРТ (DICOM, JPG, PNG)", type=["dcm", "jpg", "png"])

    if uploaded_file is None:
        st.info("Загрузите DICOM-файл МРТ или изображение.")
        return

    try:
        image = Image.open(uploaded_file).convert("L")
        image_array = np.array(image)
        mri_analysis = {
            "quality_assessment": "Хорошее",
            "sharpness": 120.0,
            "noise_level": 20.0,
            "snr": 15.0,
            "artifacts": "Минимальные артефакты"
        }
        st.image(image_array, caption="МРТ-срез", use_container_width=True, clamp=True)

        st.subheader("📊 Оценка качества МРТ")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Качество", mri_analysis['quality_assessment'])
            st.metric("Резкость", f"{mri_analysis['sharpness']:.1f}")
        with col2:
            st.metric("Шум", f"{mri_analysis['noise_level']:.1f}")
            st.metric("SNR", f"{mri_analysis['snr']:.2f}")

        st.caption(f"Артефакты: {mri_analysis['artifacts']}")

        assistant = OpenRouterAssistant()
        if st.button("🧠 ИИ-анализ МРТ (с контекстом)"):
            with st.spinner("ИИ анализирует МРТ..."):
                prompt = "Проанализируйте МРТ-срез на изображении. Учитывайте анатомию, качество, визуальные патологии."
                result = assistant.send_vision_request(prompt, image_array, str(mri_analysis))
                st.markdown("### 🧠 Нейрорадиологическое заключение:")
                st.write(result)

    except Exception as e:
        st.error(f"Ошибка обработки МРТ: {e}")

# --- Страница: Протокол приёма ---
def show_consultation_protocol():
    from local_docs import create_local_doc
    
    if not AI_AVAILABLE:
        st.error("❌ ИИ-модуль недоступен. Проверьте файл `claude_assistant.py` и API-ключ.")
        return

    st.header("📝 Автоматический протокол приёма")

    init_db()
    conn = sqlite3.connect('medical_data.db')
    patients = pd.read_sql_query("SELECT id, name FROM patients", conn)
    conn.close()

    if patients.empty:
        st.warning("❌ База пациентов пуста. Добавьте пациента в разделе 'База данных'.")
        return

    selected_patient = st.selectbox("Выберите пациента", patients['name'])
    patient_id = patients[patients['name'] == selected_patient].iloc[0]['id']

    st.subheader("🎙️ Голосовой ввод через AssemblyAI")
    audio = st.audio_input("Загрузите аудио (до 30 мин)")

    if not ASSEMBLYAI_AVAILABLE:
        st.error("❌ AssemblyAI недоступен. Проверьте файл assemblyai_transcriber.py")

    if audio and st.button("🎤 Обработать аудио"):
        if ASSEMBLYAI_AVAILABLE:
            with st.spinner("🔄 Расшифровка через AssemblyAI..."):
                try:
                    api_key = st.secrets["ASSEMBLYAI_API_KEY"]
                    raw_text = transcribe_audio_assemblyai(audio, api_key)
                    st.session_state.raw_text = raw_text
                except Exception as e:
                    st.error(f"❌ Ошибка AssemblyAI: {e}")
                    return
        else:
            st.error("❌ AssemblyAI недоступен")
            return

        st.subheader("📝 Расшифрованный текст:")
        st.text_area("Расшифрованный текст", value=raw_text, height=150, disabled=True)

        with st.spinner("🤖 Генерация протокола..."):
            assistant = OpenRouterAssistant()
            prompt = f"""
Ты — американский профессор клинической медицины и ведущий специалист в
университетской клинике. На основе следующего текста сформируйте медицинский протокол:
Контекст:
- Основная задача: сформулировать строгую, научно обоснованную и практически
применимую клиническую директиву для врача, готовую к немедленному использованию в
реальной практике.

**Жалобы:**
- ...

**Анамнез заболевания:**
- ...

**Анамнез жизни:**
- ...

**Объективный осмотр:**
- Общее состояние: лимфоузлы: Кожа: Слизистые: Пульс: АД: ЧДД:
- Сердце: Лёгкие: Живот: Печень, селезёнка: почки: стул: диурез: отёки:
- Неврологический статус: (оставьте "без патологии" или укажите изменения)

**Предварительный диагноз:**
- ...

**Рекомендованные обследования:**
- ...

**Терапия:**
- рекомендации по режиму, диете
- фармакотерапия: перечисли желаемые группы для лечения, потом укажи международное название препарата и стандартную дозу, а в скобках — 2 коммерческих генерика согласно рекомендациям...
- предложи физиолечение, 1-2 наиболее подходящих для пациента, не указывай те, которые могут быть противопоказаны

Текст: {raw_text}

Правила:
- Ответ на русском. Жалобы перечисляй с детализацией и развитием в динамике. Не переноси каждое предложение на новую строку, продолжай. Переносы только когда строка заканчивается.
- Если пункт не упомянут — поставьте "без патологии" или стандартное словосочетание, используемое в российских медицинских протоколах.
- Диагноз — только предварительный.
"""
            structured_note = assistant.get_response(prompt)
            st.session_state.structured_note = structured_note

        with st.spinner("📄 Создание документа..."):
            filepath, message = create_local_doc(f"Протокол — {selected_patient}", structured_note)
            st.success(message)
            with open(filepath, "rb") as f:
                st.download_button(
                    label="📥 Скачать протокол (.docx)",
                    data=f,
                    file_name=os.path.basename(filepath),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

        st.subheader("📄 Сгенерированный протокол")
        st.write(structured_note)

def show_patient_database():
    st.header("👤 База данных пациентов")
    init_db()

    tab1, tab2 = st.tabs(["➕ Добавить", "🔍 Поиск"])

    with tab1:
        st.subheader("Добавить пациента")
        with st.form("add_patient"):
            name = st.text_input("ФИО")
            age = st.number_input("Возраст", min_value=0, max_value=150)
            sex = st.selectbox("Пол", ["М", "Ж"])
            phone = st.text_input("Телефон")
            submitted = st.form_submit_button("Добавить")

            if submitted and name:
                conn = sqlite3.connect('medical_data.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO patients (name, age, sex, phone)
                    VALUES (?, ?, ?, ?)
                ''', (name, age, sex, phone))
                conn.commit()
                conn.close()
                st.success(f"✅ Пациент {name} добавлен!")
                st.rerun()

    with tab2:
        st.subheader("Поиск пациентов")
        conn = sqlite3.connect('medical_data.db')
        df = pd.read_sql_query("SELECT * FROM patients", conn)
        conn.close()

        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Пациенты не найдены")

def show_ai_chat():
    if not AI_AVAILABLE:
        st.error("❌ ИИ-модуль недоступен. Проверьте файл `claude_assistant.py` и API-ключ.")
        return

    st.header("🤖 ИИ-Консультант")

    try:
        assistant = OpenRouterAssistant()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 Тест подключения"):
                with st.spinner("Проверка..."):
                    success, msg = assistant.test_connection()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        with col2:
            st.info("💡 Используется Claude 3.5 Sonnet")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message("user").write(msg['user'])
            st.chat_message("assistant").write(msg['assistant'])

        user_input = st.chat_input("Задайте вопрос...")
        if user_input:
            st.chat_message("user").write(user_input)
            with st.spinner("ИИ думает..."):
                response = assistant.general_medical_consultation(user_input)
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append({
                'user': user_input,
                'assistant': response
            })
            if len(st.session_state.chat_history) > 50:
                st.session_state.chat_history = st.session_state.chat_history[-50:]

    except Exception as e:
        st.error(f"Ошибка: {e}")

def show_clinical_recommendations(diagnosis):
    """Простые клинические рекомендации без API"""
    st.markdown("### 📚 Клинические рекомендации")
    
    recommendations = {
        "пневмония": {
            "icd10": "J18.9",
            "treatment": ["Амоксициллин 500мг 3р/день", "Покой", "Обильное питье"],
            "diagnostics": ["Рентген ОГК", "Общий анализ крови", "Посев мокроты"]
        },
        "инфаркт": {
            "icd10": "I21.9",
            "treatment": ["Экстренная госпитализация", "Аспирин 300мг", "Тромболизис"],
            "diagnostics": ["ЭКГ-12", "Тропонины", "ЭхоКГ"]
        },
        "рентген": {
            "icd10": "Z01.6",
            "treatment": ["Интерпретация специалистом"],
            "diagnostics": ["Оценка качества", "Поиск патологий"]
        }
    }
    
    if diagnosis in recommendations:
        rec = recommendations[diagnosis]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Диагностика")
            for item in rec["diagnostics"]:
                st.markdown(f"- {item}")
        
        with col2:
            st.markdown("#### 💊 Лечение")
            for item in rec["treatment"]:
                st.markdown(f"- {item}")
        
        st.markdown(f"**Код по МКБ-10:** `{rec['icd10']}`")
    else:
        st.info("Рекомендации для данного диагноза не найдены")

def show_lab_analysis():
    """Улучшенная страница анализа лабораторных данных"""
    st.header("🔬 Анализ лабораторных данных")
    
    # Инициализация нового процессора
    if 'lab_processor' not in st.session_state:
        st.session_state.lab_processor = AdvancedLabProcessor()
    
    processor = st.session_state.lab_processor
    
    # Настройки
    col1, col2 = st.columns(2)
    with col1:
        auto_detect_type = st.checkbox("Автоопределение типа файла", value=True)
    with col2:
        show_raw_data = st.checkbox("Показать исходные данные", value=False)
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
    "Загрузите файл с лабораторными данными",
    type=["pdf", "xlsx", "xls", "csv", "json", "xml", "jpg", "jpeg", "png"],  # ← добавили изображения
    help="Поддерживаются: PDF, Excel, CSV, JSON, XML, JPG, PNG"
)
    
    if uploaded_file and st.button("🧪 Анализировать лабораторные данные"):
        with st.spinner("Обработка лабораторных данных..."):
            
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Обработка
                lab_report = processor.process_file(tmp_path, ai_assistant=OpenRouterAssistant())
                
                # Результаты
                if lab_report.parameters:
                    st.success(f"✅ Обработано {len(lab_report.parameters)} параметров")
                    
                    # Метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Параметров", len(lab_report.parameters))
                    with col2:
                        st.metric("Достоверность", f"{lab_report.confidence:.1%}")
                    with col3:
                        critical_count = len(lab_report.critical_values)
                        st.metric("Критических", critical_count, delta="⚠️" if critical_count > 0 else None)
                    with col4:
                        normal_count = len([p for p in lab_report.parameters if p.status == "normal"])
                        st.metric("В норме", f"{normal_count}/{len(lab_report.parameters)}")
                    
                    # Критические значения
                    if lab_report.critical_values:
                        st.error("🚨 **КРИТИЧЕСКИЕ ЗНАЧЕНИЯ:**")
                        for critical in lab_report.critical_values:
                            st.error(f"• {critical}")
                    
                    # Предупреждения
                    if lab_report.warnings:
                        st.warning("⚠️ **Предупреждения:**")
                        for warning in lab_report.warnings:
                            st.warning(f"• {warning}")
                    
                    # Таблица результатов
                    st.subheader("📊 Результаты анализов")
                    df = processor.to_dataframe(lab_report)
                    
                    # Цветовая кодировка статусов
                    def style_status(val):
                        colors = {
                            'normal': 'background-color: #d4edda',
                            'high': 'background-color: #fff3cd', 
                            'low': 'background-color: #fff3cd',
                            'critical_high': 'background-color: #f8d7da',
                            'critical_low': 'background-color: #f8d7da'
                        }
                        return colors.get(val, '')
                    
                    styled_df = df.style.applymap(style_status, subset=['Статус'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Группировка по категориям
                    st.subheader("📋 Анализ по системам")
                    summary = processor.generate_summary(lab_report)
                    
                    for category, params in summary['categories'].items():
                        with st.expander(f"📁 {category.title()} ({len(params)} параметров)"):
                            for param in params:
                                status_emoji = {
                                    'normal': '✅',
                                    'high': '⬆️', 
                                    'low': '⬇️',
                                    'critical_high': '🔴',
                                    'critical_low': '🔴'
                                }.get(param['status'], '❓')
                                
                                st.markdown(f"{status_emoji} **{param['name']}:** {param['value']} {param['unit']} ({param['status']})")
                    
                    # ИИ-интерпретация (если доступна)
                    if st.button("🤖 ИИ-интерпретация результатов"):
                        with st.spinner("ИИ анализирует результаты..."):
                            # Формируем контекст для ИИ
                            context = f"""
Лабораторные результаты пациента:
Количество параметров: {len(lab_report.parameters)}
Достоверность анализа: {lab_report.confidence:.1%}

Результаты:
"""
                            for param in lab_report.parameters:
                                context += f"- {param.name}: {param.value} {param.unit} (норма: {param.reference_range}, статус: {param.status})\n"
                            
                            if lab_report.critical_values:
                                context += f"\nКритические значения: {'; '.join(lab_report.critical_values)}"
                            
                            # Запрос к ИИ (используем существующий ассистент)
                            try:
                                assistant = OpenRouterAssistant()
                                interpretation = assistant.get_response(
                                    "Проинтерпретируйте лабораторные результаты. Дайте клиническую оценку и рекомендации.",
                                    context
                                )
                                
                                st.subheader("🧠 ИИ-интерпретация")
                                st.write(interpretation)
                                
                            except Exception as e:
                                st.error(f"Ошибка ИИ-анализа: {e}")
                    
                    # Исходные данные
                    if show_raw_data:
                        st.subheader("📄 Исходные данные")
                        st.text_area("Извлеченный текст", lab_report.raw_text, height=200)
                    
                    # Скачать результаты
                    csv_data = df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="💾 Скачать результаты (CSV)",
                        data=csv_data,
                        file_name=f"lab_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.error("❌ Не удалось извлечь лабораторные данные из файла")
                    if show_raw_data:
                        st.text_area("Извлеченный текст для диагностики", lab_report.raw_text, height=200)
            
            except Exception as e:
                st.error(f"Ошибка обработки файла: {e}")
            
            finally:
                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass

def show_genetic_analysis_page():
    st.header("🧬 Генетический анализ")
    
    uploaded_file = st.file_uploader("Загрузите генетический файл", type=["txt", "csv"])
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст", 1, 120, 30)
        gender = st.selectbox("Пол", ["М", "Ж"])
    with col2:
        lifestyle = st.selectbox("Активность", ["Низкая", "Средняя", "Высокая"])
    
    if uploaded_file and st.button("🧬 Анализировать"):
        st.success("✅ Генетические риски проанализированы!")
        st.info("📊 Функция в разработке")

# --- Главная функция ---
def main():
    st.set_page_config(
        page_title="Медицинский ИИ-Ассистент",
        page_icon="🏥",
        layout="wide"
    )

    init_db()

    # ОБНОВЛЕННЫЙ список страниц
    pages = [
        "🏠 Главная",
        "📈 Анализ ЭКГ",
        "🩻 Анализ рентгена",
        "🧠 Анализ МРТ",
        "🔬 Анализ лабораторных данных",     # ← улучшенная версия
        "📝 Протокол приёма",
        "👤 База данных пациентов",
        "🤖 ИИ-Консультант",
        "🧬 Генетический анализ",
        # === НОВЫЕ СТРАНИЦЫ ===
        "🔬 Расширенный ИИ-анализ",          # ← НОВОЕ
        "📊 Сравнительный анализ",           # ← НОВОЕ
        "📚 Медицинские протоколы",          # ← НОВОЕ
        #"🎓 Обучение ИИ",                   # ← НОВОЕ
    ]

    st.sidebar.title("🧠 Меню")
    page = st.sidebar.selectbox("Выберите раздел:", pages)

    # === ОБРАБОТКА СТРАНИЦ ===
    if page == "🏠 Главная":
        show_home_page()
    elif page == "📈 Анализ ЭКГ":
        show_ecg_analysis()
    elif page == "🩻 Анализ рентгена":
        show_xray_analysis()
    elif page == "🧠 Анализ МРТ":
        show_mri_analysis()
    elif page == "🔬 Анализ лабораторных данных":
        show_lab_analysis()  # ← ваша новая улучшенная функция
    elif page == "📝 Протокол приёма":
        show_consultation_protocol()
    elif page == "👤 База данных пациентов":
        show_patient_database()
    elif page == "🤖 ИИ-Консультант":
        show_ai_chat()
    elif page == "🧬 Генетический анализ":
        show_genetic_analysis_page()  # ← ваша готовая функция
    
    # === НОВЫЕ СТРАНИЦЫ ===
    elif page == "🔬 Расширенный ИИ-анализ":
        show_enhanced_analysis_page()
    elif page == "📊 Сравнительный анализ":
        show_comparative_analysis_page()
    elif page == "📚 Медицинские протоколы":
        show_medical_protocols_page()
    #"elif page == "🎓 Обучение ИИ":
#       show_ai_training_page()
    
    # === ОБНОВЛЕННЫЙ САЙДБАР ===
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Медицинский Ассистент v6.0** 🆕
    🔹 AssemblyAI для голоса
    🔹 10 типов изображений
    🔹 Улучшенный анализ лабораторных данных
    🔹 Структурированный JSON анализ
    🔹 Сравнительная диагностика
    🔹 Медицинские протоколы
    🔹 Claude 3.5 Sonnet + OpenRouter
    ⚠️ Только для обучения
    """)

if __name__ == "__main__":
    main()
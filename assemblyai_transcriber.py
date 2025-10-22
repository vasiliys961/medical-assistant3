from config import ASSEMBLYAI_API_KEY
# assemblyai_transcriber.py
import assemblyai as aai

def transcribe_audio_assemblyai(audio_file, api_key):
    """
    Расшифровка аудио через AssemblyAI с разделением на говорящих
    """
    try:
        aai.settings.api_key = api_key

        config = aai.TranscriptionConfig(
            speaker_labels=True,
            language_code="ru",
            speech_model=aai.SpeechModel.best,
            punctuate=True,
            format_text=True,
            disfluencies=False
        )

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file, config)

        # Проверка статуса
        if transcript.status == aai.TranscriptStatus.error:
            return f"❌ Ошибка AssemblyAI: {transcript.error}"

        # Проверка типа объекта
        if not isinstance(transcript, aai.Transcript):
            return "❌ Ошибка: transcript не является объектом Transcript"

        # Проверка utterances
        if not isinstance(transcript.utterances, list):
            return "❌ Ошибка: utterances не является списком"

        result = "🎙️ **Расшифровка разговора**\n\n"
        for utterance in transcript.utterances:
            result += f"**{utterance.speaker}**: {utterance.text}\n\n"

        return result

    except Exception as e:
        return f"❌ Ошибка при вызове AssemblyAI: {str(e)}"
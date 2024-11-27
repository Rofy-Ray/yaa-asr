import os
import torch
import tempfile
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, AutoProcessor, pipeline
import soundfile as sf
import numpy as np

class MMSASRService:
    def __init__(self):
        self.model_id = "facebook/mms-1b-all"
        self.language_map = {
            "english": {"code": "eng", "full_name": "English"},
            "akan": {"code": "aka", "full_name": "Akan"},
            "ewe": {"code": "ewe", "full_name": "Ewe"},
            "ga": {"code": "gaa", "full_name": "Ga"}
        }
        self.model = None
        self.processor = None
        self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize model with default English adapter"""
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, 
            target_lang="eng", 
            ignore_mismatched_sizes=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            target_lang="eng"
        )

    def _preprocess_audio(self, file):
        """
        Preprocess audio file to meet MMS requirements
        - Convert to 16kHz WAV
        - Ensure mono channel
        - Normalize audio
        """
        try:
            audio_data, original_sr = sf.read(file)

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            if original_sr != 16000:
                from scipy import signal
                number_of_samples = round(len(audio_data) * float(16000) / original_sr)
                audio_data = signal.resample(audio_data, number_of_samples)

            audio_data = audio_data / np.max(np.abs(audio_data))

            return audio_data
        except Exception as e:
            raise ValueError(f"Audio preprocessing error: {str(e)}")

    def transcribe(self, audio_file, target_language):
        """
        Transcribe audio for specified language
        """
        if target_language.lower() not in self.language_map:
            raise ValueError(f"Unsupported language: {target_language}")

        lang_code = self.language_map[target_language.lower()]['code']
        full_lang_name = self.language_map[target_language.lower()]['full_name']

        self.processor.tokenizer.set_target_lang(lang_code)
        self.model.load_adapter(lang_code)

        processed_audio = self._preprocess_audio(audio_file)

        inputs = self.processor(processed_audio, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)

        return {
            "transcription": transcription,
            "language": full_lang_name
        }

app = Flask(__name__)
mms_asr = MMSASRService()

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        if 'target_lang' not in request.form:
            return jsonify({"error": "Target language not specified"}), 400

        audio_file = request.files['audio']
        target_language = request.form['target_lang'].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            try:
                result = mms_asr.transcribe(temp_file.name, target_language)
                return jsonify(result), 200
            finally:
                os.unlink(temp_file.name) 

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
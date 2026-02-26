from functools import lru_cache
from pathlib import Path
from typing import Literal

import click
import librosa
import numpy as np
from numpy.typing import NDArray

from ..utils import audio_to_float32
from .stt_ import STTModel

curr_dir = Path(__file__).parent

# All languages supported by canary-1b-v2
CANARY_SUPPORTED_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr",
    "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
    "ro", "sk", "sl", "es", "sv", "ru", "uk",
)

CanaryLanguage = Literal[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr",
    "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
    "ro", "sk", "sl", "es", "sv", "ru", "uk",
]


class CanarySTT(STTModel):
    """Speech-to-Text model using NVIDIA's Canary-1B-v2.

    Canary-1B-v2 is a powerful 1-billion parameter encoder-decoder model
    (FastConformer Encoder + Transformer Decoder) for high-quality speech
    transcription (ASR) and translation (AST) across 25 European languages.

    Install:
        pip install nemo_toolkit[asr]

    Usage:
        from fastrtc.speech_to_text.canary_stt import CanarySTT

        model = CanarySTT(source_lang="en", target_lang="en")
        text = model.stt((16000, audio_array))
    """

    def __init__(
        self,
        model_name: str = "nvidia/canary-1b-v2",
        source_lang: CanaryLanguage = "en",
        target_lang: CanaryLanguage = "en",
    ):
        """Initialize the Canary STT model.

        Args:
            model_name: HuggingFace model identifier or path to a .nemo file.
                Defaults to "nvidia/canary-1b-v2".
            source_lang: Source language code for transcription/translation.
                Must be one of the 25 supported languages. Defaults to "en".
            target_lang: Target language code. Set equal to source_lang for
                transcription (ASR), or to a different language for translation (AST).
                Defaults to "en".
        """
        if source_lang not in CANARY_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"source_lang '{source_lang}' is not supported. "
                f"Supported languages: {CANARY_SUPPORTED_LANGUAGES}"
            )
        if target_lang not in CANARY_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"target_lang '{target_lang}' is not supported. "
                f"Supported languages: {CANARY_SUPPORTED_LANGUAGES}"
            )

        try:
            from nemo.collections.asr.models import ASRModel
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Install nemo_toolkit[asr] for Canary-1B-v2 support:\n"
                "  pip install nemo_toolkit[asr]"
            )

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.asr_model = ASRModel.from_pretrained(model_name=model_name)

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """Transcribe or translate audio to text.

        Args:
            audio: Tuple of (sample_rate, audio_array) where audio_array is
                a 1D numpy array of int16 or float32 samples.

        Returns:
            Transcribed (or translated) text string.
        """
        import tempfile
        import soundfile as sf

        sr, audio_np = audio
        audio_np = audio_to_float32(audio_np)

        # Resample to 16kHz if needed (Canary expects 16kHz mono)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        # Ensure 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # Canary's transcribe API expects file paths, so write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, 16000)
            tmp_path = tmp.name

        try:
            output = self.asr_model.transcribe(
                [tmp_path],
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            # output is a list of hypothesis objects with a .text attribute
            if output and hasattr(output[0], "text"):
                return output[0].text
            # Fallback for older NeMo versions that return plain strings
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], str):
                    return output[0]
                # Some versions return list of lists
                if isinstance(output[0], list) and len(output[0]) > 0:
                    return output[0][0]
            return str(output[0]) if output else ""
        finally:
            import os
            os.unlink(tmp_path)


@lru_cache
def get_canary_model(
    model_name: str = "nvidia/canary-1b-v2",
    source_lang: CanaryLanguage = "en",
    target_lang: CanaryLanguage = "en",
) -> CanarySTT:
    m = CanarySTT(model_name=model_name, source_lang=source_lang, target_lang=target_lang)
    from moonshine_onnx import load_audio

    audio = load_audio(str(curr_dir / "test_file.wav"))
    print(click.style("INFO", fg="green") + ":\t  Warming up Canary STT model.")
    m.stt((16000, audio))
    print(click.style("INFO", fg="green") + ":\t  Canary STT model warmed up.")
    return m

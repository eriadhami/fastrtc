import asyncio
import importlib.util
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol, TypeVar

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

from fastrtc.utils import async_aggregate_bytes_to_16bit


class TTSOptions:
    pass


T = TypeVar("T", bound=TTSOptions, contravariant=True)


class TTSModel(Protocol[T]):
    def tts(
        self, text: str, options: T | None = None
    ) -> tuple[int, NDArray[np.float32] | NDArray[np.int16]]: ...

    def stream_tts(
        self, text: str, options: T | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None]: ...

    def stream_tts_sync(
        self, text: str, options: T | None = None
    ) -> Generator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None, None]: ...


@dataclass
class KokoroTTSOptions(TTSOptions):
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"


@lru_cache
def get_tts_model(
    model: Literal["kokoro", "cartesia", "magpie"] = "kokoro", **kwargs
) -> TTSModel:
    if model == "kokoro":
        m = KokoroTTSModel()
        m.tts("Hello, world!")
        return m
    elif model == "cartesia":
        m = CartesiaTTSModel(api_key=kwargs.get("cartesia_api_key", ""))
        return m
    elif model == "magpie":
        m = MagpieTTSModel(
            model_name=kwargs.get(
                "model_name", "nvidia/magpie_tts_multilingual_357m"
            ),
        )
        return m
    else:
        raise ValueError(f"Invalid model: {model}")


class KokoroFixedBatchSize:
    # Source: https://github.com/thewh1teagle/kokoro-onnx/issues/115#issuecomment-2676625392
    def _split_phonemes(self, phonemes: str) -> list[str]:
        MAX_PHONEME_LENGTH = 510
        max_length = MAX_PHONEME_LENGTH - 1
        batched_phonemes = []
        while len(phonemes) > max_length:
            # Find best split point within limit
            split_idx = max_length

            # Try to find the last period before max_length
            period_idx = phonemes.rfind(".", 0, max_length)
            if period_idx != -1:
                split_idx = period_idx + 1  # Include period

            else:
                # Try other punctuation
                match = re.search(
                    r"[!?;,]", phonemes[:max_length][::-1]
                )  # Search backwards
                if match:
                    split_idx = max_length - match.start()

                else:
                    # Try last space
                    space_idx = phonemes.rfind(" ", 0, max_length)
                    if space_idx != -1:
                        split_idx = space_idx

            # If no good split point is found, force split at max_length
            chunk = phonemes[:split_idx].strip()
            batched_phonemes.append(chunk)

            # Move to the next part
            phonemes = phonemes[split_idx:].strip()

        # Add remaining phonemes
        if phonemes:
            batched_phonemes.append(phonemes)
        return batched_phonemes


class KokoroTTSModel(TTSModel):
    def __init__(self):
        from kokoro_onnx import Kokoro

        self.model = Kokoro(
            model_path=hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx"),
            voices_path=hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin"),
        )

        self.model._split_phonemes = KokoroFixedBatchSize()._split_phonemes

    def tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or KokoroTTSOptions()
        a, b = self.model.create(
            text, voice=options.voice, speed=options.speed, lang=options.lang
        )
        return b, a

    async def stream_tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or KokoroTTSOptions()

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            chunk_idx = 0
            async for chunk in self.model.create_stream(
                sentence, voice=options.voice, speed=options.speed, lang=options.lang
            ):
                if s_idx != 0 and chunk_idx == 0:
                    yield chunk[1], np.zeros(chunk[1] // 7, dtype=np.float32)
                chunk_idx += 1
                yield chunk[1], chunk[0]

    def stream_tts_sync(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()

        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break


@dataclass
class CartesiaTTSOptions(TTSOptions):
    voice: str = "71a7ad14-091c-4e8e-a314-022ece01c121"
    language: str = "en"
    emotion: list[str] = field(default_factory=list)
    cartesia_version: str = "2024-06-10"
    model: str = "sonic-2"
    sample_rate: int = 22_050


MagpieSpeaker = Literal["John", "Sofia", "Aria", "Jason", "Leo"]
MagpieLanguage = Literal["en", "es", "de", "fr", "vi", "it", "zh"]

MAGPIE_SPEAKER_MAP: dict[str, int] = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}


@dataclass
class MagpieTTSOptions(TTSOptions):
    speaker: MagpieSpeaker = "Sofia"
    language: MagpieLanguage = "en"
    apply_text_normalization: bool = True


class CartesiaTTSModel(TTSModel):
    def __init__(self, api_key: str):
        if importlib.util.find_spec("cartesia") is None:
            raise RuntimeError(
                "cartesia is not installed. Please install it using 'pip install cartesia'."
            )
        from cartesia import AsyncCartesia

        self.client = AsyncCartesia(api_key=api_key)

    async def stream_tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        options = options or CartesiaTTSOptions()

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue
            async for output in async_aggregate_bytes_to_16bit(
                self.client.tts.bytes(
                    model_id="sonic-2",
                    transcript=sentence,
                    voice={"id": options.voice},  # type: ignore
                    language="en",
                    output_format={
                        "container": "raw",
                        "sample_rate": options.sample_rate,
                        "encoding": "pcm_s16le",
                    },
                )
            ):
                yield options.sample_rate, np.frombuffer(output, dtype=np.int16)

    def stream_tts_sync(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        loop = asyncio.new_event_loop()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break

    def tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        loop = asyncio.new_event_loop()
        buffer = np.array([], dtype=np.int16)

        options = options or CartesiaTTSOptions()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                _, chunk = loop.run_until_complete(iterator.__anext__())
                buffer = np.concatenate([buffer, chunk])
            except StopAsyncIteration:
                break
        return options.sample_rate, buffer


class MagpieTTSModel(TTSModel):
    """Text-to-Speech model using NVIDIA's MagpieTTS Multilingual.

    MagpieTTS is a 357M-parameter transformer encoder-decoder model that
    generates speech in 7 languages (en, es, de, fr, vi, it, zh) with
    5 speaker voices.

    Install:
        pip install nemo_toolkit[tts] kaldialign

    Usage:
        from fastrtc.text_to_speech import get_tts_model, MagpieTTSOptions

        model = get_tts_model("magpie")
        options = MagpieTTSOptions(speaker="Sofia", language="en")
        sr, audio = model.tts("Hello world!", options)
    """

    SAMPLE_RATE = 22_050

    def __init__(
        self,
        model_name: str = "nvidia/magpie_tts_multilingual_357m",
    ):
        try:
            from nemo.collections.tts.models import MagpieTTSModel as _MagpieTTSModel
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "nemo_toolkit[tts] is not installed. Install it for MagpieTTS support:\n"
                "  pip install nemo_toolkit[tts] kaldialign"
            )
        self.model = _MagpieTTSModel.from_pretrained(model_name=model_name)

    def tts(
        self, text: str, options: MagpieTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or MagpieTTSOptions()
        speaker_idx = MAGPIE_SPEAKER_MAP[options.speaker]
        audio, audio_len = self.model.do_tts(
            text,
            language=options.language,
            apply_TN=options.apply_text_normalization,
            speaker_index=speaker_idx,
        )
        # audio is a torch tensor on GPU/CPU; convert to numpy float32
        audio_np = audio.cpu().numpy().astype(np.float32)
        # Trim to actual length
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        if audio_len is not None:
            length = int(audio_len) if not hasattr(audio_len, "item") else audio_len.item()
            audio_np = audio_np[:length]
        return self.SAMPLE_RATE, audio_np

    async def stream_tts(
        self, text: str, options: MagpieTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or MagpieTTSOptions()
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            if not sentence.strip():
                continue
            sr, audio_np = await asyncio.get_event_loop().run_in_executor(
                None, self.tts, sentence, options
            )
            yield sr, audio_np

    def stream_tts_sync(
        self, text: str, options: MagpieTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        options = options or MagpieTTSOptions()
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            if not sentence.strip():
                continue
            yield self.tts(sentence, options)

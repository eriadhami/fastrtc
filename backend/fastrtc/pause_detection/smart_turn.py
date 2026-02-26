"""Smart Turn detection using the pipecat-ai smart-turn ONNX model.

This module provides ML-based end-of-turn detection that analyzes audio
to determine whether a user has finished speaking. Unlike simple VAD-based
pause detection (which only measures silence duration), Smart Turn uses a
Whisper Tiny-based model to recognize natural conversational cues like
intonation patterns and linguistic signals.

The model works in conjunction with Silero VAD:
1. VAD detects that the user has stopped speaking (silence)
2. Smart Turn analyzes the audio from the user's turn
3. If Smart Turn says "complete" → trigger response
4. If Smart Turn says "incomplete" → wait for more speech (up to a timeout)

Pipeline integration:
    ReplyOnPause detects pause → SmartTurnAnalyzer.analyze(audio) → complete/incomplete

Model details:
    - Architecture: Whisper Tiny encoder + linear classifier (~8M params)
    - Input: 16kHz mono PCM audio, up to 8 seconds (padded/truncated)
    - Output: probability of turn completion (0.0 - 1.0)
    - Inference: ~10-100ms on CPU (8MB int8 quantized ONNX)
    - Languages: 23 supported
    - License: BSD-2-Clause (fully open source)

References:
    - Model: https://huggingface.co/pipecat-ai/smart-turn-v3
    - Source: https://github.com/pipecat-ai/smart-turn
    - Docs: https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TurnState(Enum):
    """Result of Smart Turn analysis."""

    COMPLETE = "complete"
    """The user has finished their turn — trigger a response."""

    INCOMPLETE = "incomplete"
    """The user is still speaking (mid-sentence pause, hesitation, etc.)."""


@dataclass
class SmartTurnResult:
    """Result from Smart Turn analysis.

    Attributes:
        state: Whether the turn is complete or incomplete.
        probability: Raw model probability of turn completion (0.0-1.0).
        inference_time_ms: Time spent running the ONNX model in milliseconds.
    """

    state: TurnState
    probability: float
    inference_time_ms: float


@dataclass
class SmartTurnOptions:
    """Configuration for Smart Turn detection.

    Attributes:
        enabled: Whether Smart Turn detection is active.
        model_path: Path to the Smart Turn ONNX model file. If None,
            downloads from HuggingFace (pipecat-ai/smart-turn-v3).
        model_repo: HuggingFace repo ID to download the model from.
            Default: "pipecat-ai/smart-turn-v3".
        model_filename: Filename of the ONNX model in the HuggingFace repo.
            Default: "smart-turn-v3.2-cpu.onnx" (int8 quantized, 8MB).
        completion_threshold: Probability threshold for marking a turn
            as complete. Higher = more conservative (waits longer).
            Default: 0.5.
        stop_secs: Maximum silence duration in seconds before forcing
            turn completion, even if the model says "incomplete". This is
            the fallback timeout. Default: 3.0.
        max_duration_secs: Maximum audio duration (in seconds) to feed
            to the model. Audio is truncated from the beginning if longer.
            Default: 8.0 (model's native window size).
        pre_speech_ms: Milliseconds of audio to include before speech onset.
            Captures the start of the utterance more accurately. Default: 0.0.
        cpu_threads: Number of CPU threads for ONNX inference. Default: 1.
    """

    enabled: bool = True
    model_path: Optional[str] = None
    model_repo: str = "pipecat-ai/smart-turn-v3"
    model_filename: str = "smart-turn-v3.2-cpu.onnx"
    completion_threshold: float = 0.5
    stop_secs: float = 3.0
    max_duration_secs: float = 8.0
    pre_speech_ms: float = 0.0
    cpu_threads: int = 1


class SmartTurnAnalyzer:
    """ML-based end-of-turn detector using the pipecat-ai smart-turn model.

    This analyzer uses a Whisper Tiny-based ONNX model to determine whether
    a user has finished speaking. It should be called when VAD detects a
    pause in speech, providing the accumulated audio from the user's turn.

    Usage pattern:
        1. Accumulate audio while user speaks (handled by ReplyOnPause)
        2. When VAD detects silence, call analyze(audio)
        3. If result is COMPLETE → trigger response
        4. If result is INCOMPLETE → wait for more speech or timeout

    The model expects 16kHz mono audio and internally handles padding/truncation
    to its 8-second window.
    """

    def __init__(self, options: Optional[SmartTurnOptions] = None):
        """Initialize the Smart Turn analyzer.

        Args:
            options: Configuration options. Uses defaults if None.

        Raises:
            RuntimeError: If required dependencies are not installed.
        """
        self.options = options or SmartTurnOptions()
        self._session = None
        self._feature_extractor = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization — loads model on first use.

        Raises:
            RuntimeError: If onnxruntime or transformers are not installed.
        """
        if self._initialized:
            return

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "Smart Turn requires onnxruntime. "
                "Install with: pip install 'fastrtc[smart-turn]' "
                "or: pip install onnxruntime transformers"
            ) from e

        try:
            from transformers import WhisperFeatureExtractor
        except ImportError as e:
            raise RuntimeError(
                "Smart Turn requires the transformers library for audio "
                "feature extraction. Install with: pip install transformers"
            ) from e

        # Get model path
        model_path = self.options.model_path
        if model_path is None:
            try:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id=self.options.model_repo,
                    filename=self.options.model_filename,
                )
            except ImportError as e:
                raise RuntimeError(
                    "Smart Turn requires huggingface_hub to download the model. "
                    "Install with: pip install huggingface_hub"
                ) from e

        logger.info("Loading Smart Turn model from %s", model_path)

        # Configure ONNX session
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = self.options.cpu_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 4

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

        # WhisperFeatureExtractor with 8-second chunk length (model's window)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        self._initialized = True
        logger.info("Smart Turn model loaded successfully")

    @staticmethod
    def _truncate_or_pad(
        audio: NDArray[np.float32],
        max_seconds: float = 8.0,
        sample_rate: int = 16000,
    ) -> NDArray[np.float32]:
        """Truncate audio to last N seconds or pad with zeros at the beginning.

        The model expects exactly max_seconds of audio. If shorter, zero-pad
        at the beginning. If longer, keep only the last max_seconds.

        Args:
            audio: Float32 audio array.
            max_seconds: Maximum duration in seconds.
            sample_rate: Audio sample rate.

        Returns:
            Audio array of exactly max_seconds * sample_rate samples.
        """
        max_samples = int(max_seconds * sample_rate)
        if len(audio) > max_samples:
            return audio[-max_samples:]
        elif len(audio) < max_samples:
            padding = max_samples - len(audio)
            return np.pad(audio, (padding, 0), mode="constant", constant_values=0)
        return audio

    def analyze(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> SmartTurnResult:
        """Analyze audio to determine if the user's turn is complete.

        Should be called when VAD detects silence after speech. Provide
        the full audio from the user's current turn (not just the last chunk).

        Args:
            audio: Float32 mono audio of the user's turn at 16kHz.
                If sample_rate differs, the caller must resample first.
            sample_rate: Sample rate of the audio. Must be 16000.

        Returns:
            SmartTurnResult with state, probability, and inference time.

        Raises:
            RuntimeError: If dependencies are not installed.
            ValueError: If sample_rate is not 16000.
        """
        if sample_rate != 16000:
            raise ValueError(
                f"Smart Turn requires 16kHz audio, got {sample_rate}Hz. "
                "Resample before calling analyze()."
            )

        self._ensure_initialized()
        assert self._session is not None
        assert self._feature_extractor is not None

        # Truncate/pad to model's window
        audio_prepared = self._truncate_or_pad(
            audio,
            max_seconds=self.options.max_duration_secs,
            sample_rate=sample_rate,
        )

        # Extract features using Whisper's feature extractor
        inputs = self._feature_extractor(
            audio_prepared,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # Prepare for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        # Run inference
        start_time = time.perf_counter()
        outputs = self._session.run(None, {"input_features": input_features})
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000

        # Extract probability (model returns sigmoid probability)
        probability = float(outputs[0][0].item())  # type: ignore[index]

        # Determine state
        state = (
            TurnState.COMPLETE
            if probability >= self.options.completion_threshold
            else TurnState.INCOMPLETE
        )

        logger.debug(
            "Smart Turn: %s (prob=%.4f, threshold=%.2f, inference=%.1fms)",
            state.value,
            probability,
            self.options.completion_threshold,
            inference_time_ms,
        )

        return SmartTurnResult(
            state=state,
            probability=probability,
            inference_time_ms=inference_time_ms,
        )

    def warmup(self) -> None:
        """Warm up the model with dummy data to avoid cold-start latency.

        Call this at startup to ensure the first real inference is fast.
        """
        self._ensure_initialized()

        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = self.analyze(dummy_audio, sample_rate=16000)
        logger.info(
            "Smart Turn warmup complete (inference=%.1fms)",
            result.inference_time_ms,
        )

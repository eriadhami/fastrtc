"""ReplyOnSmartTurn — ML-based end-of-turn detection for voice conversations.

Extends ReplyOnPause to add a Smart Turn model check when VAD detects a pause.
Instead of immediately triggering a response when silence is detected, this
handler runs the pipecat-ai smart-turn ONNX model on the accumulated audio to
decide whether the user actually finished their turn.

Pipeline flow:
    1. Audio arrives → VAD (Silero) detects speech / silence
    2. When silence detected after speech → Smart Turn model analyzes audio
    3. If model says COMPLETE (prob ≥ threshold) → trigger response immediately
    4. If model says INCOMPLETE → wait for more speech or forced timeout

This avoids premature responses during mid-sentence pauses, hesitations,
or "thinking" silences while still responding quickly to genuine turn endings.

Example usage::

    from fastrtc import ReplyOnSmartTurn, SmartTurnOptions

    def response_fn(audio):
        # Process audio and generate response
        yield (sample_rate, response_audio)

    handler = ReplyOnSmartTurn(
        fn=response_fn,
        smart_turn_options=SmartTurnOptions(
            completion_threshold=0.5,
            stop_secs=3.0,
        ),
    )
"""

import logging
import time
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .pause_detection import (
    DeepFilterOptions,
    ModelOptions,
    PauseDetectionModel,
    RMSGateOptions,
    get_silero_model,
)
from .pause_detection.smart_turn import (
    SmartTurnAnalyzer,
    SmartTurnOptions,
    SmartTurnResult,
    TurnState,
)
from .reply_on_pause import (
    AlgoOptions,
    AppState,
    ReplyFnGenerator,
    ReplyOnPause,
)
from .utils import audio_to_float32, create_message

logger = logging.getLogger(__name__)


class SmartTurnState(AppState):
    """Extends AppState with Smart Turn tracking fields.

    Attributes:
        awaiting_smart_turn: Whether we are in the silence window waiting
            for the Smart Turn model to confirm turn completion or timeout.
        silence_start_time: Monotonic timestamp of when the first silence
            chunk was detected (for forced timeout calculation).
        last_smart_turn_result: The most recent SmartTurnResult, useful for
            debugging and logging.
    """

    awaiting_smart_turn: bool = False
    silence_start_time: float = 0.0
    last_smart_turn_result: SmartTurnResult | None = None

    def new(self):
        """Creates a fresh SmartTurnState instance."""
        return SmartTurnState()


class ReplyOnSmartTurn(ReplyOnPause):
    """A stream handler that uses ML-based Smart Turn detection.

    Extends ReplyOnPause with a secondary check: when VAD detects silence
    after speech, the Smart Turn model analyzes the accumulated audio to
    determine if the user genuinely finished their turn.

    Behavior:
        - If Smart Turn says COMPLETE → trigger response (same as ReplyOnPause)
        - If Smart Turn says INCOMPLETE → keep waiting for more speech
        - If silence exceeds ``stop_secs`` → force-trigger response (safety timeout)
        - If user resumes speaking → cancel the Smart Turn wait

    The Smart Turn model (Whisper Tiny + linear classifier, ~8M params) runs
    in ~10-100ms on CPU, so it adds minimal latency to the response pipeline.

    Attributes:
        smart_turn: The SmartTurnAnalyzer instance for ML-based turn detection.
        smart_turn_options: Configuration for the Smart Turn model.
    """

    def __init__(
        self,
        fn: ReplyFnGenerator,
        startup_fn=None,
        algo_options: AlgoOptions | None = None,
        model_options: ModelOptions | None = None,
        can_interrupt: bool = True,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int | None = None,
        input_sample_rate: int = 48000,
        model: PauseDetectionModel | None = None,
        needs_args: bool = False,
        enable_deepfilter: bool = False,
        deepfilter_options: DeepFilterOptions | None = None,
        enable_rms_gate: bool = False,
        rms_gate_options: RMSGateOptions | None = None,
        smart_turn_options: SmartTurnOptions | None = None,
    ):
        """Initialize the ReplyOnSmartTurn handler.

        All parameters from ReplyOnPause are supported, plus:

        Args:
            smart_turn_options: Configuration for the Smart Turn model.
                Uses SmartTurnOptions defaults if None.
        """
        super().__init__(
            fn,
            startup_fn=startup_fn,
            algo_options=algo_options,
            model_options=model_options,
            can_interrupt=can_interrupt,
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            input_sample_rate=input_sample_rate,
            model=model,
            needs_args=needs_args,
            enable_deepfilter=enable_deepfilter,
            deepfilter_options=deepfilter_options,
            enable_rms_gate=enable_rms_gate,
            rms_gate_options=rms_gate_options,
        )
        self.smart_turn_options = smart_turn_options or SmartTurnOptions()
        self.smart_turn = SmartTurnAnalyzer(self.smart_turn_options)
        self.state = SmartTurnState()

    def copy(self):
        """Creates a new instance of ReplyOnSmartTurn with the same configuration."""
        return ReplyOnSmartTurn(
            self.fn,
            self.startup_fn,
            self.algo_options,
            self.model_options,
            self.can_interrupt,
            self.expected_layout,
            self.output_sample_rate,
            self.output_frame_size,
            self.input_sample_rate,
            self.model,
            self.needs_args,
            self.enable_deepfilter,
            self.deepfilter_options,
            self.enable_rms_gate,
            self.rms_gate_options,
            self.smart_turn_options,
        )

    def reset(self):
        """Resets handler state including Smart Turn tracking."""
        super().reset()
        self.state = SmartTurnState()

    def _resample_for_smart_turn(
        self,
        audio: NDArray[np.int16],
        input_sample_rate: int,
    ) -> NDArray[np.float32]:
        """Convert and resample raw audio for the Smart Turn model.

        Converts int16 → float32 and resamples to 16kHz mono.

        Args:
            audio: Raw int16 audio at the input sample rate.
            input_sample_rate: Sample rate of the input audio.

        Returns:
            Float32 audio at 16kHz, suitable for SmartTurnAnalyzer.analyze().
        """
        audio_float = audio_to_float32(audio)

        if input_sample_rate != 16000:
            try:
                import librosa  # type: ignore

                audio_float = librosa.resample(  # type: ignore[attr-defined]
                    audio_float,
                    orig_sr=input_sample_rate,
                    target_sr=16000,
                )
            except ImportError as e:
                raise RuntimeError(
                    "Smart Turn resampling requires librosa when input sample rate "
                    f"is not 16kHz (got {input_sample_rate}Hz). "
                    "Install with: pip install librosa"
                ) from e

        return audio_float

    def _run_smart_turn(
        self,
        stream_audio: NDArray[np.int16],
        input_sample_rate: int,
    ) -> SmartTurnResult:
        """Run the Smart Turn model on accumulated speech audio.

        Handles conversion from the raw pipeline format (int16, 48kHz) to
        the model's expected format (float32, 16kHz).

        Args:
            stream_audio: Accumulated speech audio from the user's turn.
            input_sample_rate: Sample rate of the stream audio.

        Returns:
            SmartTurnResult with turn state and probability.
        """
        audio_16k = self._resample_for_smart_turn(stream_audio, input_sample_rate)
        return self.smart_turn.analyze(audio_16k, sample_rate=16000)

    def determine_pause(
        self, audio: np.ndarray, sampling_rate: int, state: AppState
    ) -> bool:
        """Detect pauses with Smart Turn confirmation.

        Extends the base VAD pause detection with a secondary ML-based check.
        When VAD detects silence after speech, the Smart Turn model analyzes
        the full utterance to determine if the turn is actually complete.

        Flow:
            1. Run VAD on the audio chunk (same as ReplyOnPause)
            2. If VAD says silence after speech:
                a. Run Smart Turn → COMPLETE → return True
                b. Run Smart Turn → INCOMPLETE → return False, start timeout
                c. If already waiting and timeout exceeded → return True
            3. If user resumed speaking → reset Smart Turn wait

        Args:
            audio: The audio chunk to analyze.
            sampling_rate: Sample rate of the audio.
            state: Application state (should be SmartTurnState).

        Returns:
            True if a confirmed turn ending is detected, False otherwise.
        """
        duration = len(audio) / sampling_rate

        if duration < self.algo_options.audio_chunk_duration:
            return False

        dur_vad, _ = self.model.vad((sampling_rate, audio), self.model_options)
        logger.debug("VAD duration: %s", dur_vad)

        # Track speech onset
        if (
            dur_vad > self.algo_options.started_talking_threshold
            and not state.started_talking
        ):
            state.started_talking = True
            self._speech_started_at = time.monotonic()
            logger.debug("Started talking")
            self.send_message_sync(create_message("log", "started_talking"))

        if state.started_talking:
            # Accumulate audio into the stream buffer
            if state.stream is None:
                state.stream = audio
            else:
                state.stream = np.concatenate((state.stream, audio))

            # Check if continuous speech limit has been reached
            current_duration = len(state.stream) / sampling_rate
            if current_duration >= self.algo_options.max_continuous_speech_s:
                if isinstance(state, SmartTurnState):
                    state.awaiting_smart_turn = False
                return True

        state.buffer = None

        # If user resumed speaking while we were waiting for smart turn
        if (
            isinstance(state, SmartTurnState)
            and state.awaiting_smart_turn
            and dur_vad >= self.algo_options.speech_threshold
        ):
            logger.debug(
                "Smart Turn: user resumed speaking, canceling wait "
                "(silence was %.3fs)",
                time.monotonic() - state.silence_start_time
                if state.silence_start_time > 0
                else 0,
            )
            state.awaiting_smart_turn = False
            state.silence_start_time = 0.0
            state.last_smart_turn_result = None
            return False

        # Check if VAD detected a pause (silence after speech)
        if dur_vad < self.algo_options.speech_threshold and state.started_talking:
            # Smart Turn is disabled — fall back to normal ReplyOnPause behavior
            if not self.smart_turn_options.enabled:
                return True

            if isinstance(state, SmartTurnState):
                if not state.awaiting_smart_turn:
                    # First silence chunk — run Smart Turn model
                    state.awaiting_smart_turn = True
                    state.silence_start_time = time.monotonic()

                    if state.stream is not None and len(state.stream) > 0:
                        result = self._run_smart_turn(state.stream, sampling_rate)
                        state.last_smart_turn_result = result

                        logger.debug(
                            "Smart Turn analysis: %s (prob=%.4f, "
                            "threshold=%.2f, inference=%.1fms)",
                            result.state.value,
                            result.probability,
                            self.smart_turn_options.completion_threshold,
                            result.inference_time_ms,
                        )

                        if result.state == TurnState.COMPLETE:
                            state.awaiting_smart_turn = False
                            return True

                    # Model says INCOMPLETE — wait for more speech or timeout
                    return False
                else:
                    # Subsequent silence chunk — check for forced timeout
                    silence_elapsed = time.monotonic() - state.silence_start_time
                    if silence_elapsed >= self.smart_turn_options.stop_secs:
                        logger.debug(
                            "Smart Turn: forced timeout after %.2fs silence "
                            "(stop_secs=%.1f)",
                            silence_elapsed,
                            self.smart_turn_options.stop_secs,
                        )
                        state.awaiting_smart_turn = False
                        return True

                    # Optionally re-run the model as more silence accumulates
                    # (the model might change its mind with more context)
                    if state.stream is not None and len(state.stream) > 0:
                        result = self._run_smart_turn(state.stream, sampling_rate)
                        state.last_smart_turn_result = result

                        if result.state == TurnState.COMPLETE:
                            logger.debug(
                                "Smart Turn: reclassified as COMPLETE on retry "
                                "(prob=%.4f, silence=%.2fs)",
                                result.probability,
                                silence_elapsed,
                            )
                            state.awaiting_smart_turn = False
                            return True

                    return False
            else:
                # Not a SmartTurnState — fall back to base behavior
                return True

        return False

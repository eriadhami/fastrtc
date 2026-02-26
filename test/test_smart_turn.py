"""Tests for Smart Turn detection.

Tests cover:
- SmartTurnOptions defaults and customization
- SmartTurnAnalyzer audio truncation/padding
- SmartTurnAnalyzer initialization errors (missing deps)
- SmartTurnResult construction
- ReplyOnSmartTurn.determine_pause integration with Smart Turn
- SmartTurnState lifecycle (awaiting, cancellation, timeout)
- Resampling helpers
"""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastrtc.pause_detection.smart_turn import (
    SmartTurnAnalyzer,
    SmartTurnOptions,
    SmartTurnResult,
    TurnState,
)
from fastrtc.reply_on_smart_turn import (
    ReplyOnSmartTurn,
    SmartTurnState,
)
from fastrtc.reply_on_pause import AlgoOptions, AppState


# ---------------------------------------------------------------------------
# SmartTurnOptions
# ---------------------------------------------------------------------------
class TestSmartTurnOptions:
    def test_defaults(self):
        opts = SmartTurnOptions()
        assert opts.enabled is True
        assert opts.model_path is None
        assert opts.model_repo == "pipecat-ai/smart-turn-v3"
        assert opts.model_filename == "smart-turn-v3.2-cpu.onnx"
        assert opts.completion_threshold == 0.5
        assert opts.stop_secs == 3.0
        assert opts.max_duration_secs == 8.0
        assert opts.pre_speech_ms == 0.0
        assert opts.cpu_threads == 1

    def test_custom_values(self):
        opts = SmartTurnOptions(
            enabled=False,
            model_path="/custom/model.onnx",
            completion_threshold=0.7,
            stop_secs=5.0,
            max_duration_secs=10.0,
            cpu_threads=4,
        )
        assert opts.enabled is False
        assert opts.model_path == "/custom/model.onnx"
        assert opts.completion_threshold == 0.7
        assert opts.stop_secs == 5.0
        assert opts.max_duration_secs == 10.0
        assert opts.cpu_threads == 4


# ---------------------------------------------------------------------------
# SmartTurnResult
# ---------------------------------------------------------------------------
class TestSmartTurnResult:
    def test_complete_result(self):
        result = SmartTurnResult(
            state=TurnState.COMPLETE,
            probability=0.85,
            inference_time_ms=42.5,
        )
        assert result.state == TurnState.COMPLETE
        assert result.probability == 0.85
        assert result.inference_time_ms == 42.5

    def test_incomplete_result(self):
        result = SmartTurnResult(
            state=TurnState.INCOMPLETE,
            probability=0.3,
            inference_time_ms=18.0,
        )
        assert result.state == TurnState.INCOMPLETE
        assert result.probability == 0.3


# ---------------------------------------------------------------------------
# TurnState enum
# ---------------------------------------------------------------------------
class TestTurnState:
    def test_values(self):
        assert TurnState.COMPLETE.value == "complete"
        assert TurnState.INCOMPLETE.value == "incomplete"

    def test_enum_members(self):
        assert len(TurnState) == 2


# ---------------------------------------------------------------------------
# SmartTurnAnalyzer._truncate_or_pad
# ---------------------------------------------------------------------------
class TestTruncateOrPad:
    def test_exact_length(self):
        """Audio exactly 8 seconds should be returned as-is."""
        audio = np.zeros(8 * 16000, dtype=np.float32)
        result = SmartTurnAnalyzer._truncate_or_pad(audio, max_seconds=8.0)
        assert len(result) == 8 * 16000
        np.testing.assert_array_equal(result, audio)

    def test_longer_audio_truncates_from_beginning(self):
        """Audio >8s should keep only the last 8 seconds."""
        # 12 seconds of audio with a ramp so we can verify which part is kept
        audio = np.arange(12 * 16000, dtype=np.float32)
        result = SmartTurnAnalyzer._truncate_or_pad(audio, max_seconds=8.0)
        assert len(result) == 8 * 16000
        # Should be the last 8 seconds
        np.testing.assert_array_equal(result, audio[-8 * 16000:])

    def test_shorter_audio_pads_beginning(self):
        """Audio <8s should be zero-padded at the beginning."""
        # 3 seconds of ones
        audio = np.ones(3 * 16000, dtype=np.float32)
        result = SmartTurnAnalyzer._truncate_or_pad(audio, max_seconds=8.0)
        assert len(result) == 8 * 16000
        # First 5s should be zeros, last 3s should be ones
        expected_padding = 5 * 16000
        np.testing.assert_array_equal(result[:expected_padding], 0.0)
        np.testing.assert_array_equal(result[expected_padding:], 1.0)

    def test_empty_audio(self):
        """Empty audio should be fully zero-padded."""
        audio = np.array([], dtype=np.float32)
        result = SmartTurnAnalyzer._truncate_or_pad(audio, max_seconds=8.0)
        assert len(result) == 8 * 16000
        np.testing.assert_array_equal(result, 0.0)

    def test_custom_duration_and_sample_rate(self):
        """Should handle custom max_seconds and sample_rate."""
        audio = np.ones(5 * 8000, dtype=np.float32)  # 5s at 8kHz
        result = SmartTurnAnalyzer._truncate_or_pad(
            audio, max_seconds=3.0, sample_rate=8000
        )
        assert len(result) == 3 * 8000
        # Last 3 seconds of the 5-second audio
        np.testing.assert_array_equal(result, audio[-3 * 8000:])

    def test_one_sample(self):
        """Single sample should be padded to full length."""
        audio = np.array([0.5], dtype=np.float32)
        result = SmartTurnAnalyzer._truncate_or_pad(audio, max_seconds=1.0, sample_rate=16000)
        assert len(result) == 16000
        assert result[-1] == 0.5
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# SmartTurnAnalyzer initialization (missing dependencies)
# ---------------------------------------------------------------------------
class TestSmartTurnAnalyzerInit:
    def test_missing_onnxruntime_raises(self):
        """Should raise RuntimeError if onnxruntime is not installed."""
        analyzer = SmartTurnAnalyzer()
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(RuntimeError, match="onnxruntime"):
                analyzer._ensure_initialized()

    def test_missing_transformers_raises(self):
        """Should raise RuntimeError if transformers is not installed."""
        analyzer = SmartTurnAnalyzer()
        # Mock onnxruntime as available but transformers as missing
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "transformers": None}):
            with pytest.raises(RuntimeError, match="transformers"):
                analyzer._ensure_initialized()

    def test_sample_rate_validation(self):
        """Should raise ValueError for non-16kHz audio."""
        analyzer = SmartTurnAnalyzer()
        analyzer._initialized = True  # Skip actual model loading
        analyzer._session = MagicMock()
        analyzer._feature_extractor = MagicMock()

        audio = np.zeros(48000, dtype=np.float32)
        with pytest.raises(ValueError, match="16kHz"):
            analyzer.analyze(audio, sample_rate=48000)

    def test_lazy_initialization(self):
        """Analyzer should not load model until first use."""
        analyzer = SmartTurnAnalyzer()
        assert analyzer._initialized is False
        assert analyzer._session is None
        assert analyzer._feature_extractor is None


# ---------------------------------------------------------------------------
# SmartTurnAnalyzer.analyze (with mocked model)
# ---------------------------------------------------------------------------
class TestSmartTurnAnalyzerAnalyze:
    def _make_analyzer(self, prob: float = 0.8) -> SmartTurnAnalyzer:
        """Create an analyzer with mocked ONNX session and feature extractor."""
        analyzer = SmartTurnAnalyzer(SmartTurnOptions(completion_threshold=0.5))
        analyzer._initialized = True

        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[prob]])]
        analyzer._session = mock_session

        # Mock feature extractor
        mock_fe = MagicMock()
        mock_result = MagicMock()
        mock_result.input_features = np.zeros((1, 80, 3000), dtype=np.float32)
        mock_fe.return_value = mock_result
        analyzer._feature_extractor = mock_fe

        return analyzer

    def test_complete_turn(self):
        """High probability should return COMPLETE."""
        analyzer = self._make_analyzer(prob=0.85)
        audio = np.random.randn(16000 * 3).astype(np.float32)
        result = analyzer.analyze(audio, sample_rate=16000)
        assert result.state == TurnState.COMPLETE
        assert result.probability == pytest.approx(0.85, abs=1e-6)
        assert result.inference_time_ms >= 0

    def test_incomplete_turn(self):
        """Low probability should return INCOMPLETE."""
        analyzer = self._make_analyzer(prob=0.2)
        audio = np.random.randn(16000 * 2).astype(np.float32)
        result = analyzer.analyze(audio, sample_rate=16000)
        assert result.state == TurnState.INCOMPLETE
        assert result.probability == pytest.approx(0.2, abs=1e-6)

    def test_threshold_boundary_complete(self):
        """Probability exactly at threshold should be COMPLETE."""
        analyzer = self._make_analyzer(prob=0.5)
        audio = np.zeros(16000, dtype=np.float32)
        result = analyzer.analyze(audio, sample_rate=16000)
        assert result.state == TurnState.COMPLETE

    def test_threshold_boundary_incomplete(self):
        """Probability just below threshold should be INCOMPLETE."""
        analyzer = self._make_analyzer(prob=0.499)
        audio = np.zeros(16000, dtype=np.float32)
        result = analyzer.analyze(audio, sample_rate=16000)
        assert result.state == TurnState.INCOMPLETE

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        analyzer = self._make_analyzer(prob=0.6)
        analyzer.options = SmartTurnOptions(completion_threshold=0.7)
        audio = np.zeros(16000, dtype=np.float32)
        result = analyzer.analyze(audio, sample_rate=16000)
        assert result.state == TurnState.INCOMPLETE  # 0.6 < 0.7

    def test_feature_extractor_called_correctly(self):
        """Feature extractor should be called with correct parameters."""
        analyzer = self._make_analyzer(prob=0.5)
        audio = np.zeros(16000 * 5, dtype=np.float32)
        analyzer.analyze(audio, sample_rate=16000)

        analyzer._feature_extractor.assert_called_once()
        call_kwargs = analyzer._feature_extractor.call_args
        # Should receive padded/truncated audio
        assert call_kwargs[1]["sampling_rate"] == 16000
        assert call_kwargs[1]["return_tensors"] == "np"


# ---------------------------------------------------------------------------
# SmartTurnState
# ---------------------------------------------------------------------------
class TestSmartTurnState:
    def test_defaults(self):
        state = SmartTurnState()
        assert state.awaiting_smart_turn is False
        assert state.silence_start_time == 0.0
        assert state.last_smart_turn_result is None
        assert state.started_talking is False  # From AppState
        assert state.stream is None

    def test_new_returns_fresh_state(self):
        state = SmartTurnState()
        state.awaiting_smart_turn = True
        state.silence_start_time = 123.456
        state.started_talking = True

        new_state = state.new()
        assert isinstance(new_state, SmartTurnState)
        assert new_state.awaiting_smart_turn is False
        assert new_state.silence_start_time == 0.0
        assert new_state.started_talking is False


# ---------------------------------------------------------------------------
# ReplyOnSmartTurn.determine_pause (integration tests with mocked model)
# ---------------------------------------------------------------------------
class TestDeterminePause:
    """Tests for SmartTurn-enhanced pause detection."""

    def _make_handler(
        self,
        smart_turn_prob: float = 0.8,
        smart_turn_enabled: bool = True,
        stop_secs: float = 3.0,
        completion_threshold: float = 0.5,
    ) -> ReplyOnSmartTurn:
        """Create a handler with mocked VAD model and Smart Turn analyzer."""
        # Mock VAD model
        mock_model = MagicMock()

        # Mock reply function
        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(
            fn=mock_fn,
            model=mock_model,
            algo_options=AlgoOptions(
                audio_chunk_duration=0.6,
                started_talking_threshold=0.2,
                speech_threshold=0.1,
            ),
            smart_turn_options=SmartTurnOptions(
                enabled=smart_turn_enabled,
                completion_threshold=completion_threshold,
                stop_secs=stop_secs,
            ),
            input_sample_rate=16000,
        )

        # Replace the smart turn analyzer with a mocked version
        mock_analyzer = MagicMock(spec=SmartTurnAnalyzer)
        mock_analyzer.analyze.return_value = SmartTurnResult(
            state=TurnState.COMPLETE if smart_turn_prob >= completion_threshold else TurnState.INCOMPLETE,
            probability=smart_turn_prob,
            inference_time_ms=15.0,
        )
        mock_analyzer.options = handler.smart_turn_options
        handler.smart_turn = mock_analyzer

        # Mock send_message_sync to prevent errors
        handler.send_message_sync = MagicMock()

        return handler

    def _set_vad_result(self, handler: ReplyOnSmartTurn, duration: float):
        """Configure the mock VAD to return a specific speech duration."""
        handler.model.vad.return_value = (duration, [])

    def test_no_pause_when_not_talking(self):
        """Should return False when user hasn't started talking yet."""
        handler = self._make_handler()
        self._set_vad_result(handler, 0.05)  # Below started_talking_threshold

        audio = np.zeros(int(16000 * 0.6), dtype=np.int16)
        result = handler.determine_pause(audio, 16000, handler.state)
        assert result is False
        assert handler.state.started_talking is False

    def test_smart_turn_complete_triggers_pause(self):
        """When Smart Turn says COMPLETE, should return True immediately."""
        handler = self._make_handler(smart_turn_prob=0.85)
        state = handler.state

        # First: simulate speech to start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)
        assert state.started_talking is True

        # Then: simulate silence with Smart Turn COMPLETE
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is True

    def test_smart_turn_incomplete_delays_pause(self):
        """When Smart Turn says INCOMPLETE, should return False and wait."""
        handler = self._make_handler(smart_turn_prob=0.2)
        state = handler.state

        # Start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)

        # Silence — Smart Turn says incomplete
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is False
        assert state.awaiting_smart_turn is True
        assert state.silence_start_time > 0

    def test_forced_timeout_after_stop_secs(self):
        """Should force-trigger after stop_secs of silence."""
        handler = self._make_handler(smart_turn_prob=0.2, stop_secs=0.5)
        state = handler.state

        # Start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)

        # First silence — Smart Turn says incomplete
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is False
        assert state.awaiting_smart_turn is True

        # Simulate time passing beyond stop_secs
        state.silence_start_time = time.monotonic() - 1.0  # 1s ago exceeds 0.5s
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is True

    def test_speech_resumed_cancels_smart_turn_wait(self):
        """If user speaks again while waiting, should cancel Smart Turn wait."""
        handler = self._make_handler(smart_turn_prob=0.2)
        state = handler.state

        # Start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)

        # Silence — Smart Turn says incomplete
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_silence, 16000, state)
        assert state.awaiting_smart_turn is True

        # User resumes speaking
        self._set_vad_result(handler, 0.3)  # Above speech_threshold (0.1)
        result = handler.determine_pause(audio_speech, 16000, state)
        assert result is False
        assert state.awaiting_smart_turn is False

    def test_smart_turn_disabled_falls_back(self):
        """When smart turn is disabled, should behave like normal ReplyOnPause."""
        handler = self._make_handler(smart_turn_enabled=False)
        state = handler.state

        # Start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)

        # Silence — should trigger immediately (bypasses Smart Turn)
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is True

    def test_max_continuous_speech_bypasses_smart_turn(self):
        """Max continuous speech limit should trigger regardless of Smart Turn."""
        handler = self._make_handler(smart_turn_prob=0.2)
        handler.algo_options = AlgoOptions(
            audio_chunk_duration=0.6,
            max_continuous_speech_s=1.0,  # Very short limit
        )
        state = handler.state

        # Start talking with lots of speech
        self._set_vad_result(handler, 0.4)
        # Create audio longer than max_continuous_speech_s
        audio_speech = np.ones(int(16000 * 1.5), dtype=np.int16)
        result = handler.determine_pause(audio_speech, 16000, state)
        assert result is True
        # awaiting_smart_turn should be False (bypassed)
        assert state.awaiting_smart_turn is False

    def test_chunk_too_short_returns_false(self):
        """Audio chunks shorter than audio_chunk_duration are ignored."""
        handler = self._make_handler()
        audio = np.zeros(int(16000 * 0.3), dtype=np.int16)  # 0.3s < 0.6s
        result = handler.determine_pause(audio, 16000, handler.state)
        assert result is False
        handler.model.vad.assert_not_called()

    def test_reclassification_on_retry(self):
        """Model may reclassify as COMPLETE during subsequent silence chunks."""
        handler = self._make_handler(smart_turn_prob=0.2)
        state = handler.state

        # Start talking
        self._set_vad_result(handler, 0.4)
        audio_speech = np.ones(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_speech, 16000, state)

        # First silence — incomplete
        self._set_vad_result(handler, 0.05)
        audio_silence = np.zeros(int(16000 * 0.6), dtype=np.int16)
        handler.determine_pause(audio_silence, 16000, state)
        assert state.awaiting_smart_turn is True

        # Second silence — model now says complete
        handler.smart_turn.analyze.return_value = SmartTurnResult(
            state=TurnState.COMPLETE,
            probability=0.75,
            inference_time_ms=12.0,
        )
        result = handler.determine_pause(audio_silence, 16000, state)
        assert result is True
        assert state.awaiting_smart_turn is False


# ---------------------------------------------------------------------------
# ReplyOnSmartTurn construction and copy
# ---------------------------------------------------------------------------
class TestReplyOnSmartTurnConstruction:
    def test_default_construction(self):
        """Should create handler with default Smart Turn options."""
        mock_model = MagicMock()

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(fn=mock_fn, model=mock_model)
        assert isinstance(handler.smart_turn, SmartTurnAnalyzer)
        assert isinstance(handler.smart_turn_options, SmartTurnOptions)
        assert isinstance(handler.state, SmartTurnState)

    def test_custom_smart_turn_options(self):
        """Should pass custom options to the analyzer."""
        mock_model = MagicMock()
        opts = SmartTurnOptions(completion_threshold=0.8, stop_secs=5.0)

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(
            fn=mock_fn,
            model=mock_model,
            smart_turn_options=opts,
        )
        assert handler.smart_turn_options.completion_threshold == 0.8
        assert handler.smart_turn_options.stop_secs == 5.0

    def test_copy_preserves_smart_turn(self):
        """Copy should preserve Smart Turn configuration."""
        mock_model = MagicMock()
        opts = SmartTurnOptions(completion_threshold=0.7, stop_secs=4.0)

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(
            fn=mock_fn,
            model=mock_model,
            smart_turn_options=opts,
        )
        copied = handler.copy()
        assert isinstance(copied, ReplyOnSmartTurn)
        assert copied.smart_turn_options.completion_threshold == 0.7
        assert copied.smart_turn_options.stop_secs == 4.0

    def test_reset_clears_smart_turn_state(self):
        """Reset should create fresh SmartTurnState."""
        mock_model = MagicMock()

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(fn=mock_fn, model=mock_model)
        handler.state.awaiting_smart_turn = True
        handler.state.silence_start_time = 100.0
        handler.state.started_talking = True

        handler.reset()

        assert isinstance(handler.state, SmartTurnState)
        assert handler.state.awaiting_smart_turn is False
        assert handler.state.silence_start_time == 0.0
        assert handler.state.started_talking is False


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------
class TestResampling:
    def test_resample_16k_noop(self):
        """16kHz input should not require librosa."""
        mock_model = MagicMock()

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(fn=mock_fn, model=mock_model)
        audio = np.zeros(16000, dtype=np.int16)
        result = handler._resample_for_smart_turn(audio, 16000)
        assert result.dtype == np.float32
        assert len(result) == 16000

    def test_resample_converts_int16_to_float32(self):
        """Should convert int16 to float32."""
        mock_model = MagicMock()

        def mock_fn(audio):
            yield audio

        handler = ReplyOnSmartTurn(fn=mock_fn, model=mock_model)
        audio = np.array([16384, -16384], dtype=np.int16)
        result = handler._resample_for_smart_turn(audio, 16000)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.5, abs=0.001)
        assert result[1] == pytest.approx(-0.5, abs=0.001)

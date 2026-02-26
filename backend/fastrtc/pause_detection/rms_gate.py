"""RMS amplitude gating preprocessor for fast silence rejection.

This module provides an adaptive RMS-based silence gate that runs before
Silero VAD to cheaply reject obvious silence chunks, saving ONNX inference cost.

Pipeline order: audio_to_float32 -> DeepFilter2 -> RMS gate -> resample -> Silero VAD
                                                                     \u2514\u2500> AND gate (confirms VAD)

The gate uses a rolling window of RMS values to compute a dynamic noise floor
(85th percentile), and rejects chunks whose RMS falls below a configurable
fraction of that baseline. It includes:

- A warmup period where the gate is disabled while the rolling window fills
- A grace period (configurable, default 500ms) to avoid cutting off word onsets
- Exponential smoothing for stable baseline updates
- Media detection for background audio (TV, music) with adaptive filtering modes
- Hysteresis with separate start/stop thresholds to prevent flickering
- Combined VAD+RMS confirmation (AND gate) for post-VAD speech validation
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FilteringMode(Enum):
    """Adaptive filtering mode based on background audio environment.

    NORMAL: Standard filtering — typical quiet environment.
    AGGRESSIVE: Background media detected (TV, music) — tighter thresholds,
        higher baseline offset to separate primary speaker from background.
    """

    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


@dataclass
class MediaDetectionOptions:
    """Configuration for background media detection.

    Media (TV, music) is detected by measuring the coefficient of variation (CV)
    of RMS values over a sliding window. Speech has high CV (bursty), while
    background media has low CV (sustained, steady levels).

    Attributes:
        enabled: Whether media detection is active. Default: True.
        analysis_window_s: Duration of the sliding window used to compute
            RMS variance for media detection. Default: 5.0s.
        cv_threshold: Coefficient of variation threshold. If CV < this value,
            the audio is classified as sustained background media.
            Lower = more sensitive. Default: 0.4.
        min_rms_db: Minimum RMS level (dB) for a window to be considered
            as potential media. Silence is not media. Default: -40.0 dB.
        confirmation_windows: Number of consecutive low-CV windows required
            before switching to AGGRESSIVE mode. Prevents false triggers
            from brief steady sounds. Default: 3.
        release_windows: Number of consecutive high-CV (speech-like) windows
            required before switching back to NORMAL mode. Default: 5.
        aggressive_gate_ratio: Gate ratio used in AGGRESSIVE mode. More
            restrictive than the normal gate_ratio. Default: 0.5.
        aggressive_baseline_offset_db: Additional dB offset applied to the
            baseline in AGGRESSIVE mode. Raises the effective threshold
            so that background media is more likely to be gated. Default: 6.0.
    """

    enabled: bool = True
    analysis_window_s: float = 5.0
    cv_threshold: float = 0.4
    min_rms_db: float = -40.0
    confirmation_windows: int = 3
    release_windows: int = 5
    aggressive_gate_ratio: float = 0.5
    aggressive_baseline_offset_db: float = 6.0


@dataclass
class RMSGateOptions:
    """Configuration for the RMS amplitude gate.

    Attributes:
        enabled: Whether the RMS gate is active.
        window_duration_s: Duration of the rolling RMS history window in seconds.
            Used to compute the dynamic noise floor. Default: 3.0s.
        percentile: Percentile of RMS history used as the dynamic baseline.
            Higher values = more conservative (fewer false rejections). Default: 85.
        gate_ratio: Fraction of the dynamic baseline below which audio is
            considered silence. E.g., 0.3 means chunks with RMS < 30% of
            the 85th percentile are gated. Default: 0.3.
        warmup_duration_s: Duration in seconds at startup during which the gate
            is bypassed (always passes audio through). Allows the rolling window
            to fill before gating. Default: 3.0s.
        grace_period_s: After the gate transitions from "pass" to "reject",
            audio continues to pass for this many seconds to avoid cutting
            off the start of quiet speech. Default: 0.5s.
        smoothing_alpha: Exponential smoothing factor for baseline updates.
            Higher = more responsive to recent changes. Range: 0.0-1.0.
            Default: 0.3.
        fallback_rms_db: Static fallback threshold in dB below which audio
            is always considered silence, regardless of dynamic baseline.
            Default: -50.0 dB (very conservative).
        chunk_duration_s: Expected duration of each audio chunk in seconds.
            Used to compute how many RMS entries fit in the rolling window.
            Default: 0.02 (20ms).
        vad_confirmation_ratio: Fraction of the dynamic baseline that RMS must
            exceed for VAD-detected speech to be confirmed (AND gate). This is
            applied AFTER Silero VAD runs — even if VAD says "speech", the
            volume must also be above this threshold. Should be >= gate_ratio.
            Default: 0.5.
        hysteresis_factor: Multiplier for the gate threshold when the gate is
            currently in "passing" state. Creates asymmetric thresholds:
            - Gated → passing: uses gate_ratio (higher, harder to open)
            - Passing → gated: uses gate_ratio * hysteresis_factor (lower,
              harder to close — prevents flickering)
            Range: 0.0-1.0. Lower = more hysteresis. Default: 0.6.
        media_detection: Configuration for background media detection.
            Set to None to disable media detection entirely.
    """

    enabled: bool = True
    window_duration_s: float = 3.0
    percentile: float = 85.0
    gate_ratio: float = 0.3
    warmup_duration_s: float = 3.0
    grace_period_s: float = 0.5
    smoothing_alpha: float = 0.3
    fallback_rms_db: float = -50.0
    chunk_duration_s: float = 0.02
    vad_confirmation_ratio: float = 0.5
    hysteresis_factor: float = 0.6
    media_detection: Optional[MediaDetectionOptions] = None


class MediaDetector:
    """Detects sustained background audio (TV, music) by analyzing RMS variance.

    Speech is bursty — it has high variance in RMS amplitude over short windows.
    Background media (TV, music, podcasts) tends to have sustained, relatively
    steady RMS levels with lower variance.

    This detector computes the coefficient of variation (CV = std/mean) of RMS
    values over a sliding window. When CV drops below a threshold for multiple
    consecutive windows, it signals that background media is present.
    """

    def __init__(self, options: MediaDetectionOptions, chunk_duration_s: float):
        self.options = options
        self._chunk_duration_s = chunk_duration_s

        # Sliding window of RMS values for variance analysis
        window_entries = max(1, int(options.analysis_window_s / chunk_duration_s))
        self._rms_window: deque[float] = deque(maxlen=window_entries)

        # State counters for hysteresis
        self._low_cv_count: int = 0   # Consecutive low-CV windows
        self._high_cv_count: int = 0  # Consecutive high-CV windows

        # Current detection state
        self._media_detected: bool = False
        self._mode: FilteringMode = FilteringMode.NORMAL

        # Analysis interval — don't recompute CV on every single chunk
        # Analyze every ~100ms worth of chunks
        self._analyze_every_n = max(1, int(0.1 / chunk_duration_s))
        self._chunk_counter: int = 0

    def reset(self) -> None:
        """Reset the detector state."""
        self._rms_window.clear()
        self._low_cv_count = 0
        self._high_cv_count = 0
        self._media_detected = False
        self._mode = FilteringMode.NORMAL
        self._chunk_counter = 0

    @property
    def media_detected(self) -> bool:
        """Whether background media is currently detected."""
        return self._media_detected

    @property
    def filtering_mode(self) -> FilteringMode:
        """Current adaptive filtering mode."""
        return self._mode

    def update(self, rms: float) -> FilteringMode:
        """Feed a new RMS value and return the current filtering mode.

        Args:
            rms: RMS amplitude of the latest audio chunk.

        Returns:
            Current FilteringMode (NORMAL or AGGRESSIVE).
        """
        self._rms_window.append(rms)
        self._chunk_counter += 1

        # Only analyze periodically (every ~100ms) to save compute
        if self._chunk_counter % self._analyze_every_n != 0:
            return self._mode

        # Need enough data to compute meaningful statistics
        if len(self._rms_window) < 10:
            return self._mode

        rms_array = np.array(self._rms_window)
        mean_rms = float(np.mean(rms_array))
        rms_db = RMSAmplitudeGate.rms_to_db(mean_rms)

        # Don't analyze silence — it's not media
        if rms_db < self.options.min_rms_db:
            return self._mode

        # Coefficient of variation: low CV = steady signal = likely media
        std_rms = float(np.std(rms_array))
        if mean_rms > 1e-10:
            cv = std_rms / mean_rms
        else:
            cv = float("inf")

        if cv < self.options.cv_threshold:
            # Low variance — looks like sustained background audio
            self._low_cv_count += 1
            self._high_cv_count = 0

            if (
                not self._media_detected
                and self._low_cv_count >= self.options.confirmation_windows
            ):
                self._media_detected = True
                self._mode = FilteringMode.AGGRESSIVE
                logger.info(
                    "Media detected: CV=%.3f (< %.3f), switching to AGGRESSIVE mode",
                    cv,
                    self.options.cv_threshold,
                )
        else:
            # High variance — normal speech-like pattern
            self._high_cv_count += 1
            self._low_cv_count = 0

            if (
                self._media_detected
                and self._high_cv_count >= self.options.release_windows
            ):
                self._media_detected = False
                self._mode = FilteringMode.NORMAL
                logger.info(
                    "Media cleared: CV=%.3f (>= %.3f), switching to NORMAL mode",
                    cv,
                    self.options.cv_threshold,
                )

        return self._mode

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the media detector."""
        rms_array = np.array(self._rms_window) if self._rms_window else np.array([0.0])
        mean_rms = float(np.mean(rms_array))
        std_rms = float(np.std(rms_array))
        cv = std_rms / mean_rms if mean_rms > 1e-10 else float("inf")

        return {
            "media_detected": self._media_detected,
            "filtering_mode": self._mode.value,
            "current_cv": cv,
            "cv_threshold": self.options.cv_threshold,
            "low_cv_count": self._low_cv_count,
            "high_cv_count": self._high_cv_count,
            "window_size": len(self._rms_window),
            "mean_rms_db": RMSAmplitudeGate.rms_to_db(mean_rms),
        }


class RMSAmplitudeGate:
    """Adaptive RMS-based silence gate with media detection.

    Tracks RMS amplitude over a rolling window and uses the 85th percentile
    as a dynamic noise floor. Chunks with RMS below a configurable fraction
    of the baseline are rejected as silence.

    When media detection is enabled, the gate monitors RMS variance to detect
    sustained background audio (TV, music). When detected, it switches to
    AGGRESSIVE filtering mode with tighter thresholds and a baseline offset.

    This is designed to sit between DeepFilter2 and Silero VAD in the
    preprocessing pipeline. It's essentially free (one numpy operation)
    and avoids expensive ONNX VAD inference on obvious silence.
    """

    def __init__(self, options: Optional[RMSGateOptions] = None):
        self.options = options or RMSGateOptions()

        # Rolling window of RMS values
        max_entries = max(
            1,
            int(self.options.window_duration_s / self.options.chunk_duration_s),
        )
        self._rms_history: deque[float] = deque(maxlen=max_entries)

        # Smoothed baseline (exponential moving average of the percentile)
        self._smoothed_baseline: float = 0.0
        self._baseline_initialized: bool = False

        # Timing
        self._start_time: Optional[float] = None
        self._last_pass_time: Optional[float] = None  # Last time gate passed audio

        # Counters for diagnostics
        self._total_chunks: int = 0
        self._gated_chunks: int = 0

        # Hysteresis state: tracks whether the gate is currently passing audio
        self._is_passing: bool = True  # Start in passing state (warmup passes everything)

        # Last RMS value (for combined VAD+RMS confirmation)
        self._last_rms: float = 0.0

        # Media detector
        self._media_detector: Optional[MediaDetector] = None
        if self.options.media_detection is not None and self.options.media_detection.enabled:
            self._media_detector = MediaDetector(
                self.options.media_detection,
                self.options.chunk_duration_s,
            )

    @property
    def filtering_mode(self) -> FilteringMode:
        """Current adaptive filtering mode."""
        if self._media_detector is not None:
            return self._media_detector.filtering_mode
        return FilteringMode.NORMAL

    @property
    def media_detected(self) -> bool:
        """Whether background media is currently detected."""
        if self._media_detector is not None:
            return self._media_detector.media_detected
        return False

    def reset(self) -> None:
        """Reset the gate state. Useful when starting a new session."""
        self._rms_history.clear()
        self._smoothed_baseline = 0.0
        self._baseline_initialized = False
        self._start_time = None
        self._last_pass_time = None
        self._total_chunks = 0
        self._gated_chunks = 0
        self._is_passing = True
        self._last_rms = 0.0
        if self._media_detector is not None:
            self._media_detector.reset()

    @staticmethod
    def compute_rms(audio: NDArray[np.float32]) -> float:
        """Compute RMS amplitude of an audio chunk.

        Args:
            audio: Float32 audio array (mono, normalized -1 to 1).

        Returns:
            RMS value as a float.
        """
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    @staticmethod
    def rms_to_db(rms: float) -> float:
        """Convert RMS amplitude to dB.

        Args:
            rms: RMS amplitude value.

        Returns:
            Amplitude in dB. Returns -120.0 for zero/near-zero RMS.
        """
        if rms < 1e-10:
            return -120.0
        return float(20.0 * np.log10(rms))

    def _update_baseline(self, rms: float) -> None:
        """Update the rolling window and smoothed baseline.

        Args:
            rms: Current chunk's RMS value.
        """
        self._rms_history.append(rms)

        if len(self._rms_history) < 5:
            # Not enough data yet — don't compute percentile
            return

        raw_baseline = float(np.percentile(list(self._rms_history), self.options.percentile))

        if not self._baseline_initialized:
            self._smoothed_baseline = raw_baseline
            self._baseline_initialized = True
        else:
            # Exponential smoothing
            alpha = self.options.smoothing_alpha
            self._smoothed_baseline = (
                alpha * raw_baseline + (1.0 - alpha) * self._smoothed_baseline
            )

    def process(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> tuple[bool, float, float]:
        """Evaluate whether an audio chunk should pass through or be gated.

        This does NOT modify the audio — it only returns a decision.
        The caller decides what to do (skip VAD, zero out audio, etc.).

        When media detection is enabled and background audio is detected,
        the gate automatically switches to AGGRESSIVE mode with tighter
        thresholds and a baseline offset.

        Args:
            audio: Float32 audio array (mono, normalized -1 to 1).
            sample_rate: Sample rate of the audio (used for timing only).

        Returns:
            Tuple of (should_pass, rms, dynamic_threshold):
            - should_pass: True if the chunk should be processed by VAD.
            - rms: The computed RMS of this chunk.
            - dynamic_threshold: The current dynamic threshold (for diagnostics).
        """
        if not self.options.enabled:
            return True, 0.0, 0.0

        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now

        self._total_chunks += 1

        # Compute RMS
        rms = self.compute_rms(audio)
        self._last_rms = rms
        rms_db = self.rms_to_db(rms)

        # Always update the rolling baseline, even during warmup
        self._update_baseline(rms)

        # Update media detector (always, to keep its window current)
        if self._media_detector is not None:
            self._media_detector.update(rms)

        # During warmup: always pass through
        elapsed = now - self._start_time
        if elapsed < self.options.warmup_duration_s:
            self._last_pass_time = now
            logger.debug(
                "RMS gate warmup (%.1fs/%.1fs): rms=%.6f (%.1f dB)",
                elapsed,
                self.options.warmup_duration_s,
                rms,
                rms_db,
            )
            return True, rms, 0.0

        # Static fallback: always reject chunks below absolute floor
        if rms_db < self.options.fallback_rms_db:
            self._gated_chunks += 1
            logger.debug(
                "RMS gate REJECT (below fallback %.1f dB): rms=%.6f (%.1f dB)",
                self.options.fallback_rms_db,
                rms,
                rms_db,
            )
            return False, rms, 0.0

        # If baseline not yet established, pass through
        if not self._baseline_initialized:
            self._last_pass_time = now
            return True, rms, 0.0

        # Select gate_ratio and baseline offset based on filtering mode
        mode = self.filtering_mode
        if mode == FilteringMode.AGGRESSIVE and self.options.media_detection is not None:
            effective_gate_ratio = self.options.media_detection.aggressive_gate_ratio
            # Apply baseline offset: raise the effective baseline so that
            # background media sits below the threshold
            offset_linear = 10 ** (
                self.options.media_detection.aggressive_baseline_offset_db / 20.0
            )
            effective_baseline = self._smoothed_baseline * offset_linear
        else:
            effective_gate_ratio = self.options.gate_ratio
            effective_baseline = self._smoothed_baseline

        # Hysteresis: use lower threshold when currently passing to prevent flickering
        # - Gated → passing: needs higher threshold (gate_ratio)
        # - Passing → gated: needs lower threshold (gate_ratio * hysteresis_factor)
        if self._is_passing:
            active_ratio = effective_gate_ratio * self.options.hysteresis_factor
        else:
            active_ratio = effective_gate_ratio

        # Dynamic threshold
        dynamic_threshold = effective_baseline * active_ratio

        if rms >= dynamic_threshold:
            # Above threshold — pass through
            self._is_passing = True
            self._last_pass_time = now
            logger.debug(
                "RMS gate PASS [%s] (hysteresis=%s): rms=%.6f threshold=%.6f baseline=%.6f",
                mode.value,
                "low" if self._is_passing else "high",
                rms,
                dynamic_threshold,
                effective_baseline,
            )
            return True, rms, dynamic_threshold

        # Below threshold — but check grace period
        if self._last_pass_time is not None:
            time_since_pass = now - self._last_pass_time
            if time_since_pass < self.options.grace_period_s:
                logger.debug(
                    "RMS gate GRACE [%s] (%.3fs remaining): rms=%.6f threshold=%.6f",
                    mode.value,
                    self.options.grace_period_s - time_since_pass,
                    rms,
                    dynamic_threshold,
                )
                return True, rms, dynamic_threshold

        # Reject — transition to gated state
        self._is_passing = False
        self._gated_chunks += 1
        logger.debug(
            "RMS gate REJECT [%s]: rms=%.6f threshold=%.6f baseline=%.6f",
            mode.value,
            rms,
            dynamic_threshold,
            effective_baseline,
        )
        return False, rms, dynamic_threshold

    @property
    def last_rms(self) -> float:
        """RMS amplitude of the last processed chunk."""
        return self._last_rms

    @property
    def is_passing(self) -> bool:
        """Whether the gate is currently in 'passing' state (for hysteresis)."""
        return self._is_passing

    def confirms_speech(self) -> bool:
        """Check if the last RMS confirms speech detected by VAD (AND gate).

        Used AFTER Silero VAD detects speech to verify the volume is sufficient.
        Even if VAD says 'speech', the RMS must also exceed a confirmation
        threshold (higher than the pre-filter gate threshold) for the detection
        to be trusted.

        Returns:
            True if the RMS confirms speech, or if the gate is disabled/
            not yet initialized (fail-open). False if the volume is too low
            to trust the VAD detection.
        """
        if not self.options.enabled or not self._baseline_initialized:
            return True

        # Compute effective baseline (same logic as process())
        mode = self.filtering_mode
        if mode == FilteringMode.AGGRESSIVE and self.options.media_detection is not None:
            offset_linear = 10 ** (
                self.options.media_detection.aggressive_baseline_offset_db / 20.0
            )
            effective_baseline = self._smoothed_baseline * offset_linear
        else:
            effective_baseline = self._smoothed_baseline

        confirmation_threshold = effective_baseline * self.options.vad_confirmation_ratio
        confirmed = self._last_rms >= confirmation_threshold

        if not confirmed:
            logger.debug(
                "VAD+RMS AND gate REJECT [%s]: rms=%.6f < confirmation=%.6f (ratio=%.2f)",
                mode.value,
                self._last_rms,
                confirmation_threshold,
                self.options.vad_confirmation_ratio,
            )

        return confirmed

    @property
    def gate_rate(self) -> float:
        """Fraction of chunks that have been gated (0.0-1.0)."""
        if self._total_chunks == 0:
            return 0.0
        return self._gated_chunks / self._total_chunks

    @property
    def current_baseline(self) -> float:
        """Current smoothed RMS baseline value."""
        return self._smoothed_baseline

    @property
    def current_baseline_db(self) -> float:
        """Current smoothed RMS baseline in dB."""
        return self.rms_to_db(self._smoothed_baseline)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the gate's state.

        Returns:
            Dictionary with gate statistics, current state, and media detection info.
        """
        diag = {
            "enabled": self.options.enabled,
            "total_chunks": self._total_chunks,
            "gated_chunks": self._gated_chunks,
            "gate_rate": self.gate_rate,
            "baseline_rms": self._smoothed_baseline,
            "baseline_db": self.current_baseline_db,
            "history_size": len(self._rms_history),
            "history_max_size": self._rms_history.maxlen,
            "baseline_initialized": self._baseline_initialized,
            "filtering_mode": self.filtering_mode.value,
            "media_detected": self.media_detected,
            "is_passing": self._is_passing,
            "last_rms": self._last_rms,
            "hysteresis_factor": self.options.hysteresis_factor,
            "vad_confirmation_ratio": self.options.vad_confirmation_ratio,
        }
        if self._media_detector is not None:
            diag["media_detection"] = self._media_detector.get_diagnostics()
        return diag

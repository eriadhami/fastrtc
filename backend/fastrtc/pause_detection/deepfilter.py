import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DeepFilterOptions:
    """DeepFilter2 preprocessing options.

    Attributes:
        enabled: Whether to enable DeepFilter2 preprocessing.
        attenuation_limit: Maximum attenuation in dB. Higher values = more aggressive noise reduction.
            Typical values: 6-100. Default: 100 (very aggressive noise suppression).
    """

    enabled: bool = True
    attenuation_limit: float = 100.0


class DeepFilter2Processor:
    """Audio preprocessor using DeepFilter2 for noise suppression."""

    def __init__(self, options: Optional[DeepFilterOptions] = None):
        """Initialize DeepFilter2 processor.

        Args:
            options: DeepFilter2 configuration options.

        Raises:
            RuntimeError: If DeepFilter2 is not installed or fails to initialize.
        """
        self.options = options or DeepFilterOptions()
        
        if not self.options.enabled:
            self.model = None
            self.df_state = None
            return

        try:
            # First, ensure torch and torchaudio are properly loaded
            import torch
            import torchaudio
            import sys
            
            # WORKAROUND: deepfilternet 0.5.6 tries to import torchaudio.backend
            # which doesn't exist in torchaudio 2.x
            # Create a minimal fake package to satisfy the import
            if 'torchaudio.backend' not in sys.modules:
                from types import ModuleType
                from typing import Optional, Tuple
                
                # Create main backend module as a package
                backend_module = ModuleType('torchaudio.backend')
                backend_module.__package__ = 'torchaudio.backend'
                backend_module.__path__ = []  # Makes it a package
                sys.modules['torchaudio.backend'] = backend_module
                
                # Create common submodule with required classes
                common_module = ModuleType('torchaudio.backend.common')
                common_module.__package__ = 'torchaudio.backend'
                
                # Add AudioMetaData class that deepfilternet expects
                class AudioMetaData:
                    """Minimal AudioMetaData for torchaudio backend compatibility."""
                    def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
                        self.sample_rate = sample_rate
                        self.num_frames = num_frames
                        self.num_channels = num_channels
                
                common_module.AudioMetaData = AudioMetaData
                sys.modules['torchaudio.backend.common'] = common_module
                
                # Create soundfile_backend submodule with required functions
                soundfile_module = ModuleType('torchaudio.backend.soundfile_backend')
                soundfile_module.__package__ = 'torchaudio.backend'
                
                # Add stub functions that deepfilternet might call
                def load(filepath: str, frame_offset: int = 0, num_frames: int = -1, 
                        normalize: bool = True, channels_first: bool = True):
                    """Stub load function - delegates to torchaudio.load."""
                    import torchaudio as ta
                    return ta.load(filepath, frame_offset=frame_offset, num_frames=num_frames, 
                                  normalize=normalize, channels_first=channels_first)
                
                def info(filepath: str) -> AudioMetaData:
                    """Stub info function - delegates to torchaudio.info."""
                    import torchaudio as ta
                    meta = ta.info(filepath)
                    return AudioMetaData(meta.sample_rate, meta.num_frames, meta.num_channels)
                
                soundfile_module.load = load
                soundfile_module.info = info
                sys.modules['torchaudio.backend.soundfile_backend'] = soundfile_module
                
                # Attach to torchaudio
                torchaudio.backend = backend_module
                logger.info("Created torchaudio.backend compatibility shim for deepfilternet")
            
            # Ensure soundfile is available for audio I/O
            try:
                import soundfile
                logger.info(f"soundfile available: {soundfile.__version__}")
            except ImportError:
                logger.warning("soundfile not available")
            
            # Now import DeepFilter2 components
            from df.enhance import enhance, init_df

            self.enhance = enhance
            self.init_df = init_df
            logger.info("Successfully imported DeepFilter2 (df.enhance)")
        except ImportError as e:
            # Provide detailed diagnostic information
            import sys
            import importlib.util
            import subprocess
            
            # Check if deepfilternet package is installed
            df_spec = importlib.util.find_spec("deepfilternet")
            dflib_spec = importlib.util.find_spec("deepfilterlib")
            df_module_spec = importlib.util.find_spec("df")
            
            diagnostic_info = []
            diagnostic_info.append(f"Python version: {sys.version}")
            diagnostic_info.append(f"deepfilternet package found: {df_spec is not None}")
            diagnostic_info.append(f"deepfilterlib package found: {dflib_spec is not None}")
            diagnostic_info.append(f"df module found: {df_module_spec is not None}")
            
            if df_spec:
                diagnostic_info.append(f"deepfilternet location: {df_spec.origin if df_spec.origin else 'unknown'}")
            if df_module_spec:
                diagnostic_info.append(f"df module location: {df_module_spec.origin if df_module_spec.origin else 'unknown'}")
            
            # Try to get pip list info for deepfilternet
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=freeze"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                installed_packages = result.stdout
                deepfilter_lines = [line for line in installed_packages.split('\n') if 'deepfilter' in line.lower() or 'torch' in line.lower()]
                if deepfilter_lines:
                    diagnostic_info.append(f"Installed packages (deepfilter/torch): {', '.join(deepfilter_lines)}")
            except Exception:
                pass
            
            logger.warning("\n".join(diagnostic_info))
            
            raise RuntimeError(
                "Install deepfilternet to use DeepFilter2 preprocessing. "
                "Run: pip install deepfilternet\n"
                f"Import error: {str(e)}"
            ) from e

        try:            
            # Initialize model
            # DeepFilter2 downloads models to cache on first run
            # Set cache directory to a writable location if needed
            import os
            if "HF_HOME" in os.environ:
                # On HuggingFace Spaces, ensure cache is in a writable location
                logger.info(f"Using HF_HOME for DeepFilter2 cache: {os.environ['HF_HOME']}")
            
            self.model, self.df_state, _ = self.init_df()
            logger.info("DeepFilter2 model initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize DeepFilter2 model: {e}. "
                "This may be due to missing model files or incompatible dependencies."
            ) from e

    def process(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Process audio through DeepFilter2 for noise suppression.

        Args:
            audio: Input audio array (float32, range -1 to 1).
            sample_rate: Sample rate of the audio.

        Returns:
            Cleaned audio array with same shape as input.
        """
        if not self.options.enabled or self.model is None:
            return audio

        try:
            # DeepFilter2 expects audio in shape (channels, samples)
            # If mono, add channel dimension
            if audio.ndim == 1:
                audio_input = audio[np.newaxis, :]
                was_mono = True
            else:
                audio_input = audio
                was_mono = False

            # DeepFilter2 works at 48kHz, resample if needed
            target_sr = 48000
            if sample_rate != target_sr:
                try:
                    import librosa
                    # Resample each channel
                    if audio_input.ndim == 2:
                        resampled = np.array([
                            librosa.resample(
                                audio_input[i], 
                                orig_sr=sample_rate, 
                                target_sr=target_sr
                            )
                            for i in range(audio_input.shape[0])
                        ])
                    else:
                        resampled = librosa.resample(
                            audio_input[0], 
                            orig_sr=sample_rate, 
                            target_sr=target_sr
                        )[np.newaxis, :]
                    audio_input = resampled
                    needs_resample_back = True
                except ImportError:
                    logger.warning(
                        "librosa not available, processing at original sample rate. "
                        "Install librosa for optimal DeepFilter2 performance."
                    )
                    needs_resample_back = False
            else:
                needs_resample_back = False

            # Convert numpy array to torch tensor for DeepFilter2
            import torch
            audio_tensor = torch.from_numpy(audio_input)

            # Process with DeepFilter2
            # Note: deepfilternet 0.5.6 enhance() accepts: model, df_state, audio, atten_lim_db
            # Audio must be a torch.Tensor, not numpy array
            enhanced_tensor = self.enhance(
                self.model,
                self.df_state,
                audio_tensor,
                atten_lim_db=self.options.attenuation_limit,
            )

            # Convert back to numpy
            enhanced = enhanced_tensor.cpu().numpy()

            # Resample back to original rate if needed
            if needs_resample_back:
                try:
                    import librosa
                    if enhanced.ndim == 2:
                        enhanced = np.array([
                            librosa.resample(
                                enhanced[i],
                                orig_sr=target_sr,
                                target_sr=sample_rate
                            )
                            for i in range(enhanced.shape[0])
                        ])
                    else:
                        enhanced = librosa.resample(
                            enhanced[0],
                            orig_sr=target_sr,
                            target_sr=sample_rate
                        )[np.newaxis, :]
                except ImportError:
                    pass

            # Remove channel dimension if input was mono
            if was_mono:
                enhanced = enhanced[0]

            return enhanced

        except Exception as e:
            logger.error(f"Error processing audio with DeepFilter2: {e}")
            # Return original audio on error
            return audio

    def warmup(self):
        """Warm up the DeepFilter2 model with dummy audio."""
        if not self.options.enabled or self.model is None:
            return
        
        logger.info("Warming up DeepFilter2 model...")
        # Process a few dummy audio chunks to warm up the model
        for _ in range(3):
            dummy_audio = np.zeros(48000, dtype=np.float32)  # 1 second at 48kHz
            self.process(dummy_audio, 48000)
        logger.info("DeepFilter2 model warmed up")


@lru_cache(maxsize=1)
def get_deepfilter_processor(
    enabled: bool = True,
    attenuation_limit: float = 100.0,
) -> DeepFilter2Processor:
    """Get or create a cached DeepFilter2 processor instance.

    Args:
        enabled: Whether to enable DeepFilter2 preprocessing.
        attenuation_limit: Maximum attenuation in dB (range: 0-100, default 100).

    Returns:
        DeepFilter2Processor instance.
    """
    options = DeepFilterOptions(
        enabled=enabled,
        attenuation_limit=attenuation_limit,
    )
    return DeepFilter2Processor(options)

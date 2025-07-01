"""
Audio processing utilities for CompI Phase 2.A: Audio Input Integration

This module provides comprehensive audio analysis capabilities including:
- Audio feature extraction (tempo, energy, spectral features)
- Audio preprocessing and normalization
- Audio-to-text captioning using OpenAI Whisper
- Multimodal prompt fusion combining audio features with text prompts
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    tempo: float
    energy: float  # RMS energy
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_rolloff: float
    mfcc_mean: np.ndarray
    chroma_mean: np.ndarray
    duration: float
    sample_rate: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'tempo': float(self.tempo),
            'energy': float(self.energy),
            'zero_crossing_rate': float(self.zero_crossing_rate),
            'spectral_centroid': float(self.spectral_centroid),
            'spectral_rolloff': float(self.spectral_rolloff),
            'mfcc_mean': self.mfcc_mean.tolist() if hasattr(self.mfcc_mean, 'tolist') else list(self.mfcc_mean),
            'chroma_mean': self.chroma_mean.tolist() if hasattr(self.chroma_mean, 'tolist') else list(self.chroma_mean),
            'duration': float(self.duration),
            'sample_rate': int(self.sample_rate)
        }

class AudioProcessor:
    """Comprehensive audio processing and analysis"""
    
    def __init__(self, target_sr: int = 16000, max_duration: float = 60.0):
        """
        Initialize audio processor
        
        Args:
            target_sr: Target sample rate for processing
            max_duration: Maximum audio duration to process (seconds)
        """
        self.target_sr = target_sr
        self.max_duration = max_duration
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(
                audio_path, 
                sr=self.target_sr, 
                duration=self.max_duration
            )
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            logger.info(f"Loaded audio: {audio_path}, duration: {len(audio)/sr:.2f}s")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray, sr: int) -> AudioFeatures:
        """
        Extract comprehensive audio features
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        try:
            # Basic features
            duration = len(audio) / sr
            
            # Tempo and beat tracking
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=audio)[0]
            energy = np.sqrt(np.mean(rms**2))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_centroid = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            features = AudioFeatures(
                tempo=float(tempo),
                energy=float(energy),
                zero_crossing_rate=float(zcr_mean),
                spectral_centroid=float(spectral_centroid),
                spectral_rolloff=float(spectral_rolloff_mean),
                mfcc_mean=mfcc_mean,
                chroma_mean=chroma_mean,
                duration=float(duration),
                sample_rate=int(sr)
            )
            
            logger.info(f"Extracted features: tempo={float(tempo):.1f}, energy={float(energy):.4f}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            raise
    
    def analyze_audio_file(self, audio_path: str) -> AudioFeatures:
        """
        Complete audio analysis pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioFeatures object
        """
        audio, sr = self.load_audio(audio_path)
        return self.extract_features(audio, sr)

class AudioCaptioner:
    """Audio-to-text captioning using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize audio captioner
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (auto, cpu, cuda)
        """
        self.model_size = model_size
        self.device = device
        self._model = None
        
    def _load_model(self):
        """Lazy load Whisper model"""
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(self.model_size, device=self.device)
                logger.info(f"Loaded Whisper model: {self.model_size}")
            except ImportError:
                logger.error("OpenAI Whisper not installed. Install with: pip install openai-whisper")
                raise
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
    
    def caption_audio(self, audio_path: str, language: str = "en") -> str:
        """
        Generate text caption from audio
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Text caption of the audio content
        """
        self._load_model()
        
        try:
            import whisper
            
            # Load and preprocess audio for Whisper
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Generate mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self._model.device)
            
            # Decode audio
            options = whisper.DecodingOptions(language=language, fp16=False)
            result = whisper.decode(self._model, mel, options)
            
            caption = result.text.strip()
            logger.info(f"Generated audio caption: '{caption[:50]}...'")
            
            return caption
            
        except Exception as e:
            logger.error(f"Error captioning audio: {e}")
            return ""

class MultimodalPromptFusion:
    """Intelligent fusion of text prompts with audio features and captions"""
    
    def __init__(self):
        """Initialize prompt fusion system"""
        pass
    
    def fuse_prompt_with_audio(
        self, 
        text_prompt: str,
        style: str,
        mood: str,
        audio_features: AudioFeatures,
        audio_caption: str = ""
    ) -> str:
        """
        Create enhanced prompt by fusing text with audio analysis
        
        Args:
            text_prompt: Original text prompt
            style: Art style
            mood: Mood/atmosphere
            audio_features: Extracted audio features
            audio_caption: Audio caption from Whisper
            
        Returns:
            Enhanced multimodal prompt
        """
        # Start with base prompt
        enhanced_prompt = text_prompt.strip()
        
        # Add style and mood
        if style:
            enhanced_prompt += f", {style}"
        if mood:
            enhanced_prompt += f", {mood}"
        
        # Add audio caption if available
        if audio_caption:
            enhanced_prompt += f", inspired by the sound of: {audio_caption}"
        
        # Add tempo-based descriptors
        if audio_features.tempo < 80:
            enhanced_prompt += ", slow and contemplative"
        elif audio_features.tempo > 140:
            enhanced_prompt += ", fast-paced and energetic"
        elif audio_features.tempo > 120:
            enhanced_prompt += ", upbeat and dynamic"
        
        # Add energy-based descriptors
        if audio_features.energy > 0.05:
            enhanced_prompt += ", vibrant and powerful"
        elif audio_features.energy < 0.02:
            enhanced_prompt += ", gentle and subtle"
        
        # Add rhythm-based descriptors
        if audio_features.zero_crossing_rate > 0.15:
            enhanced_prompt += ", rhythmic and percussive"
        
        # Add tonal descriptors based on spectral features
        if audio_features.spectral_centroid > 3000:
            enhanced_prompt += ", bright and crisp"
        elif audio_features.spectral_centroid < 1500:
            enhanced_prompt += ", warm and deep"
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    
    def generate_audio_tags(self, audio_features: AudioFeatures) -> List[str]:
        """
        Generate descriptive tags based on audio features
        
        Args:
            audio_features: Extracted audio features
            
        Returns:
            List of descriptive tags
        """
        tags = []
        
        # Tempo tags
        if audio_features.tempo < 60:
            tags.append("very_slow")
        elif audio_features.tempo < 90:
            tags.append("slow")
        elif audio_features.tempo < 120:
            tags.append("moderate")
        elif audio_features.tempo < 140:
            tags.append("fast")
        else:
            tags.append("very_fast")
        
        # Energy tags
        if audio_features.energy > 0.06:
            tags.append("high_energy")
        elif audio_features.energy > 0.03:
            tags.append("medium_energy")
        else:
            tags.append("low_energy")
        
        # Rhythm tags
        if audio_features.zero_crossing_rate > 0.15:
            tags.append("percussive")
        elif audio_features.zero_crossing_rate < 0.05:
            tags.append("smooth")
        
        # Spectral tags
        if audio_features.spectral_centroid > 3000:
            tags.append("bright")
        elif audio_features.spectral_centroid < 1500:
            tags.append("dark")
        
        return tags

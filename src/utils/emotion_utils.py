"""
CompI Emotion Processing Utilities

This module provides utilities for Phase 2.C: Emotional/Contextual Input Integration
- Emotion detection and sentiment analysis
- Mood mapping and emotional context processing
- Color palette generation based on emotions
- Contextual prompt enhancement
- Emoji and text-based emotion recognition
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Optional imports with fallbacks
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    emoji = None

logger = logging.getLogger(__name__)

class EmotionCategory(Enum):
    """Primary emotion categories"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    NEUTRAL = "neutral"

@dataclass
class EmotionAnalysis:
    """Container for emotion analysis results"""
    
    # Primary emotion detection
    primary_emotion: EmotionCategory
    emotion_confidence: float  # 0-1 confidence score
    
    # Sentiment analysis
    sentiment_polarity: float  # -1 to 1 (negative to positive)
    sentiment_subjectivity: float  # 0 to 1 (objective to subjective)
    
    # Detected emotions with scores
    emotion_scores: Dict[str, float]
    
    # Contextual information
    detected_emojis: List[str]
    emotion_keywords: List[str]
    intensity_level: str  # 'low', 'medium', 'high'
    
    # Generated artistic attributes
    color_palette: List[str]
    artistic_descriptors: List[str]
    mood_modifiers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'primary_emotion': self.primary_emotion.value,
            'emotion_confidence': self.emotion_confidence,
            'sentiment_polarity': self.sentiment_polarity,
            'sentiment_subjectivity': self.sentiment_subjectivity,
            'emotion_scores': self.emotion_scores,
            'detected_emojis': self.detected_emojis,
            'emotion_keywords': self.emotion_keywords,
            'intensity_level': self.intensity_level,
            'color_palette': self.color_palette,
            'artistic_descriptors': self.artistic_descriptors,
            'mood_modifiers': self.mood_modifiers
        }

class EmotionProcessor:
    """Core emotion processing and analysis functionality"""
    
    def __init__(self):
        """Initialize the emotion processor with predefined mappings"""
        
        # Predefined emotion sets
        self.preset_emotions = {
            "joyful": {"category": EmotionCategory.JOY, "intensity": "high", "emoji": "😊"},
            "happy": {"category": EmotionCategory.JOY, "intensity": "medium", "emoji": "😄"},
            "ecstatic": {"category": EmotionCategory.JOY, "intensity": "high", "emoji": "🤩"},
            "sad": {"category": EmotionCategory.SADNESS, "intensity": "medium", "emoji": "😢"},
            "melancholic": {"category": EmotionCategory.SADNESS, "intensity": "high", "emoji": "😔"},
            "depressed": {"category": EmotionCategory.SADNESS, "intensity": "high", "emoji": "😞"},
            "angry": {"category": EmotionCategory.ANGER, "intensity": "high", "emoji": "😡"},
            "frustrated": {"category": EmotionCategory.ANGER, "intensity": "medium", "emoji": "😤"},
            "furious": {"category": EmotionCategory.ANGER, "intensity": "high", "emoji": "🤬"},
            "fearful": {"category": EmotionCategory.FEAR, "intensity": "high", "emoji": "😱"},
            "anxious": {"category": EmotionCategory.FEAR, "intensity": "medium", "emoji": "😰"},
            "nervous": {"category": EmotionCategory.FEAR, "intensity": "low", "emoji": "😬"},
            "surprised": {"category": EmotionCategory.SURPRISE, "intensity": "medium", "emoji": "😲"},
            "amazed": {"category": EmotionCategory.SURPRISE, "intensity": "high", "emoji": "🤯"},
            "romantic": {"category": EmotionCategory.LOVE, "intensity": "high", "emoji": "💖"},
            "loving": {"category": EmotionCategory.LOVE, "intensity": "medium", "emoji": "❤️"},
            "peaceful": {"category": EmotionCategory.TRUST, "intensity": "medium", "emoji": "🕊️"},
            "serene": {"category": EmotionCategory.TRUST, "intensity": "high", "emoji": "🌱"},
            "mysterious": {"category": EmotionCategory.ANTICIPATION, "intensity": "medium", "emoji": "🕵️‍♂️"},
            "nostalgic": {"category": EmotionCategory.SADNESS, "intensity": "medium", "emoji": "🕰️"},
            "energetic": {"category": EmotionCategory.JOY, "intensity": "high", "emoji": "⚡"},
            "whimsical": {"category": EmotionCategory.JOY, "intensity": "medium", "emoji": "🎠"},
            "uplifting": {"category": EmotionCategory.JOY, "intensity": "high", "emoji": "🌞"},
            "dark": {"category": EmotionCategory.SADNESS, "intensity": "high", "emoji": "🌑"},
            "moody": {"category": EmotionCategory.SADNESS, "intensity": "medium", "emoji": "🌫️"}
        }
        
        # Emotion-to-color mappings
        self.emotion_colors = {
            EmotionCategory.JOY: ["#FFD700", "#FFA500", "#FF69B4", "#00CED1", "#32CD32"],
            EmotionCategory.SADNESS: ["#4169E1", "#6495ED", "#708090", "#2F4F4F", "#191970"],
            EmotionCategory.ANGER: ["#DC143C", "#B22222", "#8B0000", "#FF4500", "#FF6347"],
            EmotionCategory.FEAR: ["#800080", "#4B0082", "#2E2E2E", "#696969", "#A9A9A9"],
            EmotionCategory.SURPRISE: ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB", "#FFFF00"],
            EmotionCategory.LOVE: ["#FF69B4", "#DC143C", "#FF1493", "#C71585", "#DB7093"],
            EmotionCategory.TRUST: ["#00CED1", "#20B2AA", "#48D1CC", "#40E0D0", "#AFEEEE"],
            EmotionCategory.ANTICIPATION: ["#9370DB", "#8A2BE2", "#7B68EE", "#6A5ACD", "#483D8B"],
            EmotionCategory.NEUTRAL: ["#808080", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#DCDCDC"]
        }
        
        # Artistic descriptors for each emotion
        self.artistic_descriptors = {
            EmotionCategory.JOY: ["vibrant", "luminous", "radiant", "effervescent", "sparkling"],
            EmotionCategory.SADNESS: ["muted", "somber", "melancholic", "wistful", "contemplative"],
            EmotionCategory.ANGER: ["intense", "fiery", "bold", "dramatic", "powerful"],
            EmotionCategory.FEAR: ["shadowy", "mysterious", "ethereal", "haunting", "enigmatic"],
            EmotionCategory.SURPRISE: ["dynamic", "explosive", "unexpected", "striking", "vivid"],
            EmotionCategory.LOVE: ["warm", "tender", "passionate", "romantic", "intimate"],
            EmotionCategory.TRUST: ["serene", "peaceful", "harmonious", "balanced", "tranquil"],
            EmotionCategory.ANTICIPATION: ["electric", "suspenseful", "charged", "expectant", "tense"],
            EmotionCategory.NEUTRAL: ["balanced", "calm", "steady", "composed", "neutral"]
        }
        
        # Emoji to emotion mapping
        self.emoji_emotions = {
            "😊": EmotionCategory.JOY, "😄": EmotionCategory.JOY, "😃": EmotionCategory.JOY,
            "🤩": EmotionCategory.JOY, "😍": EmotionCategory.LOVE, "🥰": EmotionCategory.LOVE,
            "😢": EmotionCategory.SADNESS, "😭": EmotionCategory.SADNESS, "😔": EmotionCategory.SADNESS,
            "😡": EmotionCategory.ANGER, "🤬": EmotionCategory.ANGER, "😤": EmotionCategory.ANGER,
            "😱": EmotionCategory.FEAR, "😰": EmotionCategory.FEAR, "😨": EmotionCategory.FEAR,
            "😲": EmotionCategory.SURPRISE, "😮": EmotionCategory.SURPRISE, "🤯": EmotionCategory.SURPRISE,
            "❤️": EmotionCategory.LOVE, "💖": EmotionCategory.LOVE, "💕": EmotionCategory.LOVE,
            "🕊️": EmotionCategory.TRUST, "🌱": EmotionCategory.TRUST, "☮️": EmotionCategory.TRUST
        }
        
        # Keyword patterns for emotion detection
        self.emotion_keywords = {
            EmotionCategory.JOY: ["happy", "joyful", "cheerful", "delighted", "elated", "euphoric", "blissful"],
            EmotionCategory.SADNESS: ["sad", "depressed", "melancholy", "sorrowful", "gloomy", "dejected"],
            EmotionCategory.ANGER: ["angry", "furious", "rage", "irritated", "annoyed", "livid", "irate"],
            EmotionCategory.FEAR: ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "fearful"],
            EmotionCategory.SURPRISE: ["surprised", "amazed", "astonished", "shocked", "stunned", "bewildered"],
            EmotionCategory.LOVE: ["love", "romantic", "affectionate", "tender", "passionate", "adoring"],
            EmotionCategory.TRUST: ["peaceful", "serene", "calm", "tranquil", "secure", "confident"],
            EmotionCategory.ANTICIPATION: ["excited", "eager", "hopeful", "expectant", "anticipating"]
        }
    
    def analyze_emotion(self, text: str, selected_emotion: Optional[str] = None) -> EmotionAnalysis:
        """
        Comprehensive emotion analysis of input text
        
        Args:
            text: Input text to analyze
            selected_emotion: Optional pre-selected emotion
            
        Returns:
            EmotionAnalysis object with complete analysis
        """
        logger.info(f"Analyzing emotion for text: {text[:100]}...")
        
        # Initialize analysis components
        detected_emojis = self._extract_emojis(text)
        emotion_keywords = self._extract_emotion_keywords(text)
        
        # Determine primary emotion
        if selected_emotion and selected_emotion.lower() in self.preset_emotions:
            # Use selected emotion
            emotion_info = self.preset_emotions[selected_emotion.lower()]
            primary_emotion = emotion_info["category"]
            emotion_confidence = 0.9
            intensity_level = emotion_info["intensity"]
        else:
            # Analyze text for emotion
            primary_emotion, emotion_confidence, intensity_level = self._analyze_text_emotion(text, detected_emojis, emotion_keywords)
        
        # Sentiment analysis
        sentiment_polarity, sentiment_subjectivity = self._analyze_sentiment(text)
        
        # Generate emotion scores
        emotion_scores = self._generate_emotion_scores(primary_emotion, emotion_confidence)
        
        # Generate artistic attributes
        color_palette = self.emotion_colors.get(primary_emotion, self.emotion_colors[EmotionCategory.NEUTRAL])
        artistic_descriptors = self.artistic_descriptors.get(primary_emotion, ["neutral"])
        mood_modifiers = self._generate_mood_modifiers(primary_emotion, intensity_level)
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            emotion_confidence=emotion_confidence,
            sentiment_polarity=sentiment_polarity,
            sentiment_subjectivity=sentiment_subjectivity,
            emotion_scores=emotion_scores,
            detected_emojis=detected_emojis,
            emotion_keywords=emotion_keywords,
            intensity_level=intensity_level,
            color_palette=color_palette[:3],  # Top 3 colors
            artistic_descriptors=artistic_descriptors[:3],  # Top 3 descriptors
            mood_modifiers=mood_modifiers
        )

    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        if not EMOJI_AVAILABLE:
            # Simple emoji detection using Unicode ranges
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            return emoji_pattern.findall(text)
        else:
            return [char for char in text if char in emoji.UNICODE_EMOJI['en']]

    def _extract_emotion_keywords(self, text: str) -> List[str]:
        """Extract emotion-related keywords from text"""
        text_lower = text.lower()
        found_keywords = []

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

        return found_keywords

    def _analyze_text_emotion(self, text: str, emojis: List[str], keywords: List[str]) -> Tuple[EmotionCategory, float, str]:
        """Analyze emotion from text, emojis, and keywords"""

        # Check emojis first
        for emoji_char in emojis:
            if emoji_char in self.emoji_emotions:
                return self.emoji_emotions[emoji_char], 0.8, "medium"

        # Check keywords
        emotion_votes = {}
        for keyword in keywords:
            for emotion, emotion_keywords in self.emotion_keywords.items():
                if keyword in emotion_keywords:
                    emotion_votes[emotion] = emotion_votes.get(emotion, 0) + 1

        if emotion_votes:
            primary_emotion = max(emotion_votes, key=emotion_votes.get)
            confidence = min(emotion_votes[primary_emotion] * 0.3, 0.9)
            intensity = "high" if emotion_votes[primary_emotion] > 2 else "medium"
            return primary_emotion, confidence, intensity

        # Fallback to sentiment analysis
        sentiment_polarity, _ = self._analyze_sentiment(text)

        if sentiment_polarity > 0.3:
            return EmotionCategory.JOY, 0.6, "medium"
        elif sentiment_polarity < -0.3:
            return EmotionCategory.SADNESS, 0.6, "medium"
        else:
            return EmotionCategory.NEUTRAL, 0.5, "low"

    def _analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using TextBlob or fallback method"""
        if not text.strip():
            return 0.0, 0.0

        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {e}")

        # Simple fallback sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "joy"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "fear", "worried", "depressed"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0, 0.0

        polarity = (positive_count - negative_count) / max(total_words, 1)
        subjectivity = (positive_count + negative_count) / max(total_words, 1)

        return max(-1.0, min(1.0, polarity)), max(0.0, min(1.0, subjectivity))

    def _generate_emotion_scores(self, primary_emotion: EmotionCategory, confidence: float) -> Dict[str, float]:
        """Generate scores for all emotions"""
        scores = {emotion.value: 0.1 for emotion in EmotionCategory}
        scores[primary_emotion.value] = confidence

        # Add some secondary emotions based on primary
        secondary_emotions = {
            EmotionCategory.JOY: [EmotionCategory.LOVE, EmotionCategory.TRUST],
            EmotionCategory.SADNESS: [EmotionCategory.FEAR, EmotionCategory.NEUTRAL],
            EmotionCategory.ANGER: [EmotionCategory.DISGUST, EmotionCategory.FEAR],
            EmotionCategory.FEAR: [EmotionCategory.SADNESS, EmotionCategory.SURPRISE],
            EmotionCategory.LOVE: [EmotionCategory.JOY, EmotionCategory.TRUST],
            EmotionCategory.TRUST: [EmotionCategory.JOY, EmotionCategory.LOVE]
        }

        if primary_emotion in secondary_emotions:
            for secondary in secondary_emotions[primary_emotion]:
                scores[secondary.value] = min(0.4, confidence * 0.5)

        return scores

    def _generate_mood_modifiers(self, emotion: EmotionCategory, intensity: str) -> List[str]:
        """Generate mood modifiers for prompt enhancement"""
        base_modifiers = {
            EmotionCategory.JOY: ["bright", "cheerful", "uplifting", "radiant"],
            EmotionCategory.SADNESS: ["melancholic", "somber", "wistful", "contemplative"],
            EmotionCategory.ANGER: ["intense", "dramatic", "powerful", "bold"],
            EmotionCategory.FEAR: ["mysterious", "dark", "ethereal", "haunting"],
            EmotionCategory.SURPRISE: ["dynamic", "striking", "unexpected", "vivid"],
            EmotionCategory.LOVE: ["romantic", "warm", "tender", "passionate"],
            EmotionCategory.TRUST: ["peaceful", "serene", "harmonious", "tranquil"],
            EmotionCategory.ANTICIPATION: ["electric", "suspenseful", "charged", "expectant"],
            EmotionCategory.NEUTRAL: ["balanced", "calm", "neutral", "composed"]
        }

        modifiers = base_modifiers.get(emotion, ["neutral"])

        # Adjust based on intensity
        if intensity == "high":
            intensity_modifiers = ["very", "extremely", "deeply", "intensely"]
            return [f"{intensity_modifiers[0]} {mod}" for mod in modifiers[:2]]
        elif intensity == "low":
            return [f"subtly {mod}" for mod in modifiers[:2]]
        else:
            return modifiers[:3]


class EmotionalPromptEnhancer:
    """Enhance prompts with emotional context"""

    def __init__(self):
        """Initialize the prompt enhancer"""
        self.emotion_processor = EmotionProcessor()

    def enhance_prompt_with_emotion(
        self,
        base_prompt: str,
        style: str,
        emotion_analysis: EmotionAnalysis,
        enhancement_strength: float = 0.7
    ) -> str:
        """
        Enhance prompt with emotional context

        Args:
            base_prompt: Original text prompt
            style: Art style
            emotion_analysis: Emotion analysis results
            enhancement_strength: How strongly to apply emotion (0-1)

        Returns:
            Enhanced prompt with emotional context
        """
        enhanced_prompt = base_prompt.strip()

        # Add style
        if style:
            enhanced_prompt += f", {style}"

        # Add emotional descriptors based on strength
        if enhancement_strength > 0.5:
            # Strong emotional enhancement
            descriptors = emotion_analysis.artistic_descriptors[:2]
            mood_modifiers = emotion_analysis.mood_modifiers[:2]

            enhanced_prompt += f", {', '.join(descriptors)}"
            enhanced_prompt += f", with a {', '.join(mood_modifiers)} atmosphere"

            # Add intensity if high
            if emotion_analysis.intensity_level == "high":
                enhanced_prompt += f", deeply {emotion_analysis.primary_emotion.value}"

        elif enhancement_strength > 0.2:
            # Moderate emotional enhancement
            descriptor = emotion_analysis.artistic_descriptors[0]
            mood = emotion_analysis.mood_modifiers[0]

            enhanced_prompt += f", {descriptor}, {mood}"

        else:
            # Subtle emotional enhancement
            if emotion_analysis.artistic_descriptors:
                enhanced_prompt += f", {emotion_analysis.artistic_descriptors[0]}"

        return enhanced_prompt

    def generate_emotion_tags(self, emotion_analysis: EmotionAnalysis) -> List[str]:
        """Generate descriptive tags for the emotion"""
        tags = []

        # Primary emotion
        tags.append(emotion_analysis.primary_emotion.value)

        # Intensity
        tags.append(f"{emotion_analysis.intensity_level}_intensity")

        # Sentiment
        if emotion_analysis.sentiment_polarity > 0.3:
            tags.append("positive_sentiment")
        elif emotion_analysis.sentiment_polarity < -0.3:
            tags.append("negative_sentiment")
        else:
            tags.append("neutral_sentiment")

        # Artistic descriptors
        tags.extend(emotion_analysis.artistic_descriptors[:2])

        return tags

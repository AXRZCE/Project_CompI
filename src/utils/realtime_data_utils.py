"""
CompI Real-Time Data Processing Utilities

This module provides utilities for Phase 2.D: Real-Time Data Feeds Integration
- Weather data fetching from multiple APIs
- News headlines and RSS feed processing
- Social media trends and sentiment analysis
- Stock market and financial data integration
- Data summarization and context generation
- Real-time data caching and rate limiting
"""

import os
import json
import time
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataFeedType(Enum):
    """Types of real-time data feeds"""
    WEATHER = "weather"
    NEWS = "news"
    SOCIAL = "social"
    FINANCIAL = "financial"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    CUSTOM_RSS = "custom_rss"

@dataclass
class RealTimeDataPoint:
    """Container for a single real-time data point"""
    
    feed_type: DataFeedType
    source: str
    timestamp: datetime
    title: str
    content: str
    metadata: Dict[str, Any]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'feed_type': self.feed_type.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'sentiment_score': self.sentiment_score,
            'relevance_score': self.relevance_score
        }

@dataclass
class RealTimeContext:
    """Container for processed real-time context"""
    
    data_points: List[RealTimeDataPoint]
    summary: str
    mood_indicators: List[str]
    key_themes: List[str]
    temporal_context: str
    artistic_inspiration: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'data_points': [dp.to_dict() for dp in self.data_points],
            'summary': self.summary,
            'mood_indicators': self.mood_indicators,
            'key_themes': self.key_themes,
            'temporal_context': self.temporal_context,
            'artistic_inspiration': self.artistic_inspiration
        }

class DataFeedCache:
    """Simple caching system for real-time data to respect rate limits"""
    
    def __init__(self, cache_duration_minutes: int = 15):
        """
        Initialize cache
        
        Args:
            cache_duration_minutes: How long to cache data in minutes
        """
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get_cache_key(self, feed_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key from feed type and parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{feed_type}_{param_str}".encode()).hexdigest()
    
    def get(self, feed_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached data if still valid"""
        cache_key = self.get_cache_key(feed_type, params)
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.info(f"Using cached data for {feed_type}")
                return data
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    def set(self, feed_type: str, params: Dict[str, Any], data: Any):
        """Cache data with timestamp"""
        cache_key = self.get_cache_key(feed_type, params)
        self.cache[cache_key] = (data, datetime.now())
        logger.info(f"Cached data for {feed_type}")

class WeatherDataFetcher:
    """Fetch weather data from multiple sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather fetcher
        
        Args:
            api_key: OpenWeatherMap API key (optional, uses demo key if not provided)
        """
        self.api_key = api_key or "9a524f695a4940f392150142250107"  # User's API key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def fetch_weather(self, city: str, country_code: Optional[str] = None) -> RealTimeDataPoint:
        """
        Fetch current weather for a city
        
        Args:
            city: City name
            country_code: Optional country code (e.g., 'US', 'UK')
            
        Returns:
            RealTimeDataPoint with weather information
        """
        logger.info(f"Fetching weather for {city}")
        
        # Prepare query
        query = city
        if country_code:
            query += f",{country_code}"
        
        params = {
            "q": query,
            "units": "metric",
            "appid": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract weather information
            weather_main = data['weather'][0]['main']
            weather_desc = data['weather'][0]['description']
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            pressure = data['main']['pressure']
            
            # Create content summary
            content = f"Current weather in {city}: {weather_desc}, {temp:.1f}°C (feels like {feels_like:.1f}°C), humidity {humidity}%, pressure {pressure} hPa"
            
            # Determine mood based on weather
            mood_mapping = {
                'clear': 'bright and optimistic',
                'clouds': 'contemplative and soft',
                'rain': 'melancholic and reflective',
                'drizzle': 'gentle and soothing',
                'thunderstorm': 'dramatic and intense',
                'snow': 'serene and peaceful',
                'mist': 'mysterious and ethereal',
                'fog': 'mysterious and ethereal'
            }
            
            mood = mood_mapping.get(weather_main.lower(), 'neutral')
            
            return RealTimeDataPoint(
                feed_type=DataFeedType.WEATHER,
                source="OpenWeatherMap",
                timestamp=datetime.now(),
                title=f"Weather in {city}",
                content=content,
                metadata={
                    'city': city,
                    'country_code': country_code,
                    'temperature': temp,
                    'feels_like': feels_like,
                    'humidity': humidity,
                    'pressure': pressure,
                    'weather_main': weather_main,
                    'weather_description': weather_desc,
                    'mood': mood
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return RealTimeDataPoint(
                feed_type=DataFeedType.WEATHER,
                source="OpenWeatherMap",
                timestamp=datetime.now(),
                title=f"Weather in {city}",
                content=f"Unable to fetch weather data for {city}: {str(e)}",
                metadata={'error': str(e), 'city': city}
            )

class NewsDataFetcher:
    """Fetch news data from multiple sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news fetcher
        
        Args:
            api_key: NewsAPI key (optional, uses RSS feeds if not provided)
        """
        self.api_key = api_key
        self.newsapi_url = "https://newsapi.org/v2/top-headlines"
        
        # Free RSS feeds for different categories
        self.rss_feeds = {
            'general': 'https://feeds.bbci.co.uk/news/rss.xml',
            'technology': 'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'science': 'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
            'world': 'https://feeds.bbci.co.uk/news/world/rss.xml',
            'business': 'https://feeds.bbci.co.uk/news/business/rss.xml'
        }
    
    def fetch_news_headlines(self, category: str = 'general', max_headlines: int = 5) -> List[RealTimeDataPoint]:
        """
        Fetch news headlines
        
        Args:
            category: News category
            max_headlines: Maximum number of headlines to fetch
            
        Returns:
            List of RealTimeDataPoint objects with news data
        """
        logger.info(f"Fetching {max_headlines} news headlines for category: {category}")
        
        if self.api_key:
            return self._fetch_from_newsapi(category, max_headlines)
        else:
            return self._fetch_from_rss(category, max_headlines)
    
    def _fetch_from_newsapi(self, category: str, max_headlines: int) -> List[RealTimeDataPoint]:
        """Fetch news from NewsAPI (requires API key)"""
        params = {
            'apiKey': self.api_key,
            'category': category,
            'pageSize': max_headlines,
            'language': 'en'
        }
        
        try:
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            news_points = []
            for article in data.get('articles', []):
                news_point = RealTimeDataPoint(
                    feed_type=DataFeedType.NEWS,
                    source=article.get('source', {}).get('name', 'Unknown'),
                    timestamp=datetime.now(),
                    title=article.get('title', ''),
                    content=article.get('description', ''),
                    metadata={
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'category': category
                    }
                )
                news_points.append(news_point)
            
            return news_points
            
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def _fetch_from_rss(self, category: str, max_headlines: int) -> List[RealTimeDataPoint]:
        """Fetch news from RSS feeds (free, no API key required)"""
        feed_url = self.rss_feeds.get(category, self.rss_feeds['general'])
        
        try:
            feed = feedparser.parse(feed_url)
            news_points = []
            
            for entry in feed.entries[:max_headlines]:
                news_point = RealTimeDataPoint(
                    feed_type=DataFeedType.NEWS,
                    source=feed.feed.get('title', 'BBC News'),
                    timestamp=datetime.now(),
                    title=entry.get('title', ''),
                    content=entry.get('summary', ''),
                    metadata={
                        'url': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'category': category
                    }
                )
                news_points.append(news_point)
            
            return news_points
            
        except Exception as e:
            logger.error(f"Error fetching RSS news: {e}")
            return []


class FinancialDataFetcher:
    """Fetch financial and market data"""

    def __init__(self):
        """Initialize financial data fetcher"""
        # Using free APIs that don't require keys
        self.crypto_url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        self.forex_url = "https://api.exchangerate-api.com/v4/latest/USD"

    def fetch_market_summary(self) -> List[RealTimeDataPoint]:
        """
        Fetch basic market data

        Returns:
            List of RealTimeDataPoint objects with financial data
        """
        logger.info("Fetching market summary")
        data_points = []

        # Fetch Bitcoin price
        try:
            response = requests.get(self.crypto_url, timeout=10)
            response.raise_for_status()
            btc_data = response.json()

            btc_price = btc_data['bpi']['USD']['rate']
            btc_point = RealTimeDataPoint(
                feed_type=DataFeedType.FINANCIAL,
                source="CoinDesk",
                timestamp=datetime.now(),
                title="Bitcoin Price",
                content=f"Bitcoin (BTC): {btc_price}",
                metadata={
                    'currency': 'USD',
                    'asset': 'Bitcoin',
                    'symbol': 'BTC'
                }
            )
            data_points.append(btc_point)

        except Exception as e:
            logger.error(f"Error fetching Bitcoin data: {e}")

        # Fetch basic forex data
        try:
            response = requests.get(self.forex_url, timeout=10)
            response.raise_for_status()
            forex_data = response.json()

            eur_rate = forex_data['rates'].get('EUR', 'N/A')
            gbp_rate = forex_data['rates'].get('GBP', 'N/A')

            forex_point = RealTimeDataPoint(
                feed_type=DataFeedType.FINANCIAL,
                source="ExchangeRate-API",
                timestamp=datetime.now(),
                title="Currency Exchange",
                content=f"USD/EUR: {eur_rate}, USD/GBP: {gbp_rate}",
                metadata={
                    'base_currency': 'USD',
                    'eur_rate': eur_rate,
                    'gbp_rate': gbp_rate
                }
            )
            data_points.append(forex_point)

        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")

        return data_points


class RealTimeDataProcessor:
    """Process and contextualize real-time data for artistic inspiration"""

    def __init__(self):
        """Initialize the data processor"""
        self.cache = DataFeedCache()
        self.weather_fetcher = WeatherDataFetcher()
        self.news_fetcher = NewsDataFetcher()
        self.financial_fetcher = FinancialDataFetcher()

        # Mood and theme mappings
        self.mood_keywords = {
            'positive': ['sunny', 'clear', 'bright', 'growth', 'success', 'celebration', 'victory'],
            'negative': ['storm', 'rain', 'decline', 'crisis', 'conflict', 'tragedy', 'loss'],
            'neutral': ['cloudy', 'stable', 'steady', 'normal', 'routine', 'regular'],
            'dramatic': ['thunderstorm', 'breaking', 'urgent', 'major', 'significant', 'dramatic'],
            'peaceful': ['calm', 'gentle', 'quiet', 'serene', 'peaceful', 'tranquil']
        }

    def fetch_realtime_context(
        self,
        include_weather: bool = False,
        weather_city: str = "New York",
        include_news: bool = False,
        news_category: str = "general",
        max_news: int = 3,
        include_financial: bool = False,
        weather_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None
    ) -> RealTimeContext:
        """
        Fetch and process real-time data from multiple sources

        Args:
            include_weather: Whether to include weather data
            weather_city: City for weather data
            include_news: Whether to include news data
            news_category: Category of news to fetch
            max_news: Maximum number of news items
            include_financial: Whether to include financial data
            weather_api_key: Optional weather API key
            news_api_key: Optional news API key

        Returns:
            RealTimeContext with processed data
        """
        logger.info("Fetching real-time context")

        data_points = []

        # Fetch weather data
        if include_weather:
            cache_key = f"weather_{weather_city}"
            cached_weather = self.cache.get("weather", {"city": weather_city})

            if cached_weather:
                data_points.append(cached_weather)
            else:
                if weather_api_key:
                    self.weather_fetcher.api_key = weather_api_key

                weather_data = self.weather_fetcher.fetch_weather(weather_city)
                data_points.append(weather_data)
                self.cache.set("weather", {"city": weather_city}, weather_data)

        # Fetch news data
        if include_news:
            cache_key = f"news_{news_category}_{max_news}"
            cached_news = self.cache.get("news", {"category": news_category, "max": max_news})

            if cached_news:
                data_points.extend(cached_news)
            else:
                if news_api_key:
                    self.news_fetcher.api_key = news_api_key

                news_data = self.news_fetcher.fetch_news_headlines(news_category, max_news)
                data_points.extend(news_data)
                self.cache.set("news", {"category": news_category, "max": max_news}, news_data)

        # Fetch financial data
        if include_financial:
            cached_financial = self.cache.get("financial", {})

            if cached_financial:
                data_points.extend(cached_financial)
            else:
                financial_data = self.financial_fetcher.fetch_market_summary()
                data_points.extend(financial_data)
                self.cache.set("financial", {}, financial_data)

        # Process the collected data
        return self._process_data_points(data_points)

    def _process_data_points(self, data_points: List[RealTimeDataPoint]) -> RealTimeContext:
        """Process data points into artistic context"""

        if not data_points:
            return RealTimeContext(
                data_points=[],
                summary="No real-time data available",
                mood_indicators=[],
                key_themes=[],
                temporal_context="",
                artistic_inspiration=""
            )

        # Generate summary
        summaries = []
        for dp in data_points:
            summaries.append(f"{dp.title}: {dp.content}")

        summary = "; ".join(summaries)

        # Extract mood indicators
        mood_indicators = self._extract_mood_indicators(data_points)

        # Extract key themes
        key_themes = self._extract_key_themes(data_points)

        # Generate temporal context
        temporal_context = self._generate_temporal_context(data_points)

        # Generate artistic inspiration
        artistic_inspiration = self._generate_artistic_inspiration(data_points, mood_indicators, key_themes)

        return RealTimeContext(
            data_points=data_points,
            summary=summary,
            mood_indicators=mood_indicators,
            key_themes=key_themes,
            temporal_context=temporal_context,
            artistic_inspiration=artistic_inspiration
        )

    def _extract_mood_indicators(self, data_points: List[RealTimeDataPoint]) -> List[str]:
        """Extract mood indicators from data points"""
        moods = []

        for dp in data_points:
            content_lower = dp.content.lower()

            # Check weather mood
            if dp.feed_type == DataFeedType.WEATHER:
                weather_mood = dp.metadata.get('mood', '')
                if weather_mood:
                    moods.append(weather_mood)

            # Check content for mood keywords
            for mood, keywords in self.mood_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    moods.append(mood)
                    break

        return list(set(moods))  # Remove duplicates

    def _extract_key_themes(self, data_points: List[RealTimeDataPoint]) -> List[str]:
        """Extract key themes from data points"""
        themes = []

        for dp in data_points:
            if dp.feed_type == DataFeedType.WEATHER:
                themes.append("nature")
                themes.append("environment")
            elif dp.feed_type == DataFeedType.NEWS:
                themes.append("current events")
                themes.append("society")
            elif dp.feed_type == DataFeedType.FINANCIAL:
                themes.append("economy")
                themes.append("markets")

        return list(set(themes))

    def _generate_temporal_context(self, data_points: List[RealTimeDataPoint]) -> str:
        """Generate temporal context description"""
        now = datetime.now()
        time_desc = now.strftime("%A, %B %d, %Y at %I:%M %p")

        return f"Real-time context captured on {time_desc}"

    def _generate_artistic_inspiration(
        self,
        data_points: List[RealTimeDataPoint],
        mood_indicators: List[str],
        key_themes: List[str]
    ) -> str:
        """Generate artistic inspiration text from processed data"""

        inspiration_parts = []

        # Add mood-based inspiration
        if mood_indicators:
            mood_text = ", ".join(mood_indicators)
            inspiration_parts.append(f"reflecting a {mood_text} atmosphere")

        # Add theme-based inspiration
        if key_themes:
            theme_text = " and ".join(key_themes)
            inspiration_parts.append(f"inspired by {theme_text}")

        # Add specific data inspirations
        for dp in data_points:
            if dp.feed_type == DataFeedType.WEATHER:
                weather_desc = dp.metadata.get('weather_description', '')
                if weather_desc:
                    inspiration_parts.append(f"with {weather_desc} weather influences")

            elif dp.feed_type == DataFeedType.NEWS:
                inspiration_parts.append("capturing the pulse of current events")

            elif dp.feed_type == DataFeedType.FINANCIAL:
                inspiration_parts.append("reflecting market dynamics and economic energy")

        if inspiration_parts:
            return ", ".join(inspiration_parts)
        else:
            return "drawing from the current moment in time"

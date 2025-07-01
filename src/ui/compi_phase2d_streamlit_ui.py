"""
CompI Phase 2.D: Real-Time Data Feeds Streamlit UI

Interactive web interface for real-time data-driven AI art generation combining:
- Weather data integration from multiple APIs
- News headlines and RSS feed processing
- Financial market data incorporation
- Real-time context visualization
- Comprehensive generation controls
"""

import os
import sys
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.generators.compi_phase2d_realtime_to_image import CompIPhase2DRealTimeToImage
from src.utils.realtime_data_utils import RealTimeContext, DataFeedType

# ------------------------ CONFIGURATION ------------------------

st.set_page_config(
    page_title="CompI Phase 2.D: Real-Time Data-to-Art", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .data-feed-info {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .realtime-context {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4169E1;
    }
    .data-point {
        background-color: #f9f9f9;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #32CD32;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------ INITIALIZATION ------------------------

@st.cache_resource(show_spinner=True)
def load_generator():
    """Load and cache the CompI Phase 2.D generator"""
    return CompIPhase2DRealTimeToImage()

# Display header
st.markdown('<h1 class="main-header">🌎 CompI Phase 2.D: Real-Time Data-to-Art Generator</h1>', unsafe_allow_html=True)

st.markdown("""
**Welcome to CompI Phase 2.D!** Connect your art to the pulse of the world through real-time data feeds. 
Incorporate live weather, breaking news, and market data to create artwork that captures the current moment in time.
""")

# Device information
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Running on: {device.upper()}")
if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    except:
        pass

# Load generator
with st.spinner("Loading AI models... (this may take a moment on first run)"):
    generator = load_generator()

# ------------------------ INPUT SECTION ------------------------

st.header("🎨 Generation Settings")

col1, col2 = st.columns([2, 1])

with col1:
    # Text inputs
    text_prompt = st.text_input(
        "Text Prompt", 
        value="A landscape that reflects the world's current mood",
        help="Describe what you want to see in the image"
    )
    
    style = st.text_input(
        "Art Style", 
        value="impressionist digital painting",
        help="e.g., photorealistic, abstract, oil painting, cyberpunk"
    )
    
    mood = st.text_input(
        "Base Mood/Atmosphere", 
        value="contemplative, dynamic",
        help="Base mood that will be enhanced by real-time data"
    )

with col2:
    # Generation parameters
    st.subheader("Generation Parameters")
    
    num_images = st.number_input(
        "Number of Images", 
        min_value=1, max_value=4, value=1, step=1
    )
    
    num_steps = st.slider(
        "Quality Steps", 
        min_value=10, max_value=50, value=30, step=5,
        help="More steps = higher quality but slower generation"
    )
    
    guidance_scale = st.slider(
        "Prompt Adherence", 
        min_value=1.0, max_value=20.0, value=7.5, step=0.5,
        help="Higher values follow the prompt more closely"
    )
    
    context_strength = st.slider(
        "Real-Time Context Strength", 
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="How strongly to apply real-time data context"
    )

# ------------------------ REAL-TIME DATA CONFIGURATION ------------------------

st.header("🌐 Real-Time Data Sources")

# Data source selection
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌤️ Weather Data")
    include_weather = st.checkbox("Include Weather Data", value=False)
    
    if include_weather:
        weather_city = st.text_input(
            "City", 
            value="New York",
            help="City for weather data"
        )
        
        weather_api_key = st.text_input(
            "OpenWeatherMap API Key (Optional)",
            value="9a524f695a4940f392150142250107",
            type="password",
            help="Leave blank to use demo key with limited requests"
        )
        
        if st.button("Test Weather Connection", key="test_weather"):
            with st.spinner("Testing weather connection..."):
                try:
                    context = generator.fetch_realtime_context(
                        include_weather=True,
                        weather_city=weather_city,
                        weather_api_key=weather_api_key if weather_api_key else None
                    )
                    
                    if context.data_points:
                        weather_data = next((dp for dp in context.data_points if dp.feed_type == DataFeedType.WEATHER), None)
                        if weather_data:
                            st.success(f"✅ Weather: {weather_data.content}")
                        else:
                            st.error("❌ No weather data received")
                    else:
                        st.error("❌ Failed to fetch weather data")
                        
                except Exception as e:
                    st.error(f"❌ Weather connection failed: {e}")

with col2:
    st.subheader("📰 News Data")
    include_news = st.checkbox("Include News Headlines", value=False)
    
    if include_news:
        news_category = st.selectbox(
            "News Category",
            ["general", "technology", "science", "world", "business"],
            help="Category of news to fetch"
        )
        
        max_news = st.slider(
            "Max Headlines",
            min_value=1, max_value=10, value=3, step=1,
            help="Maximum number of headlines to include"
        )
        
        news_api_key = st.text_input(
            "NewsAPI Key (Optional)",
            type="password",
            help="Leave blank to use free RSS feeds"
        )
        
        if st.button("Test News Connection", key="test_news"):
            with st.spinner("Testing news connection..."):
                try:
                    context = generator.fetch_realtime_context(
                        include_news=True,
                        news_category=news_category,
                        max_news=max_news,
                        news_api_key=news_api_key if news_api_key else None
                    )
                    
                    news_data = [dp for dp in context.data_points if dp.feed_type == DataFeedType.NEWS]
                    if news_data:
                        st.success(f"✅ Fetched {len(news_data)} news headlines")
                        for i, news in enumerate(news_data[:2]):  # Show first 2
                            st.info(f"{i+1}. {news.title}")
                    else:
                        st.error("❌ No news data received")
                        
                except Exception as e:
                    st.error(f"❌ News connection failed: {e}")

with col3:
    st.subheader("💹 Financial Data")
    include_financial = st.checkbox("Include Market Data", value=False)
    
    if include_financial:
        st.info("📊 **Available Data:**\n- Bitcoin price\n- USD exchange rates\n- Market indicators")
        
        if st.button("Test Financial Connection", key="test_financial"):
            with st.spinner("Testing financial connection..."):
                try:
                    context = generator.fetch_realtime_context(include_financial=True)
                    
                    financial_data = [dp for dp in context.data_points if dp.feed_type == DataFeedType.FINANCIAL]
                    if financial_data:
                        st.success(f"✅ Fetched {len(financial_data)} financial data points")
                        for financial in financial_data:
                            st.info(f"💰 {financial.content}")
                    else:
                        st.error("❌ No financial data received")
                        
                except Exception as e:
                    st.error(f"❌ Financial connection failed: {e}")

# ------------------------ REAL-TIME CONTEXT PREVIEW ------------------------

if include_weather or include_news or include_financial:
    st.header("📊 Real-Time Context Preview")
    
    if st.button("🔄 Fetch Current Data", type="secondary"):
        with st.spinner("Fetching real-time data..."):
            try:
                context = generator.fetch_realtime_context(
                    include_weather=include_weather,
                    weather_city=weather_city if include_weather else "New York",
                    weather_api_key=weather_api_key if include_weather and weather_api_key else None,
                    include_news=include_news,
                    news_category=news_category if include_news else "general",
                    max_news=max_news if include_news else 3,
                    news_api_key=news_api_key if include_news and news_api_key else None,
                    include_financial=include_financial
                )
                
                if context.data_points:
                    st.markdown('<div class="realtime-context">', unsafe_allow_html=True)
                    st.success("✅ Real-time data fetched successfully!")
                    
                    # Display summary
                    st.write(f"**📝 Summary:** {context.summary}")
                    
                    # Display mood indicators
                    if context.mood_indicators:
                        st.write(f"**🎭 Mood Indicators:** {', '.join(context.mood_indicators)}")
                    
                    # Display key themes
                    if context.key_themes:
                        st.write(f"**🎯 Key Themes:** {', '.join(context.key_themes)}")
                    
                    # Display artistic inspiration
                    st.write(f"**🎨 Artistic Inspiration:** {context.artistic_inspiration}")
                    
                    # Display temporal context
                    st.write(f"**⏰ Temporal Context:** {context.temporal_context}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display individual data points
                    with st.expander("🔍 Detailed Data Points", expanded=False):
                        for i, dp in enumerate(context.data_points):
                            st.markdown(f'<div class="data-point">', unsafe_allow_html=True)
                            st.write(f"**{dp.feed_type.value.title()} - {dp.source}**")
                            st.write(f"📰 {dp.title}")
                            st.write(f"📄 {dp.content}")
                            st.write(f"🕐 {dp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store context in session state for generation
                    st.session_state['realtime_context'] = context
                    
                else:
                    st.warning("⚠️ No real-time data could be fetched")
                    
            except Exception as e:
                st.error(f"❌ Error fetching real-time data: {e}")
                st.exception(e)

# ------------------------ PROMPT PREVIEW ------------------------

st.header("📝 Enhanced Prompt Preview")

# Create preview of enhanced prompt
enhanced_prompt = text_prompt
if style:
    enhanced_prompt += f", {style}"
if mood:
    enhanced_prompt += f", {mood}"

# Add real-time context preview if available
if 'realtime_context' in st.session_state and context_strength > 0:
    context = st.session_state['realtime_context']
    if context_strength > 0.7:
        enhanced_prompt += f", {context.artistic_inspiration}"
        if context.mood_indicators:
            mood_text = ", ".join(context.mood_indicators[:2])
            enhanced_prompt += f", with {mood_text} influences"
    elif context_strength > 0.4:
        enhanced_prompt += f", {context.artistic_inspiration}"
    else:
        if context.key_themes:
            theme = context.key_themes[0]
            enhanced_prompt += f", inspired by {theme}"

st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

# Show data source indicators
data_sources = []
if include_weather:
    data_sources.append("🌤️ Weather")
if include_news:
    data_sources.append("📰 News")
if include_financial:
    data_sources.append("💹 Financial")

if data_sources:
    st.info(f"**📡 Active Data Sources:** {' | '.join(data_sources)}")

# ------------------------ GENERATION ------------------------

st.header("🎨 Generate Real-Time Art")

# Advanced options
with st.expander("⚙️ Advanced Options", expanded=False):
    batch_mode = st.checkbox(
        "Batch Mode",
        value=False,
        help="Generate multiple images with different data source combinations"
    )

    if batch_mode:
        st.write("**Batch Configurations:**")
        batch_configs = []

        if st.checkbox("Weather Only", key="batch_weather"):
            batch_configs.append({
                "include_weather": True,
                "weather_city": weather_city if include_weather else "New York",
                "include_news": False,
                "include_financial": False
            })

        if st.checkbox("News Only", key="batch_news"):
            batch_configs.append({
                "include_weather": False,
                "include_news": True,
                "news_category": news_category if include_news else "general",
                "max_news": max_news if include_news else 3,
                "include_financial": False
            })

        if st.checkbox("Financial Only", key="batch_financial"):
            batch_configs.append({
                "include_weather": False,
                "include_news": False,
                "include_financial": True
            })

        if st.checkbox("All Sources Combined", key="batch_all"):
            batch_configs.append({
                "include_weather": include_weather,
                "weather_city": weather_city if include_weather else "New York",
                "include_news": include_news,
                "news_category": news_category if include_news else "general",
                "max_news": max_news if include_news else 3,
                "include_financial": include_financial
            })

    temporal_series = st.checkbox(
        "Temporal Series",
        value=False,
        help="Generate images at different time intervals to show data evolution"
    )

    if temporal_series:
        intervals = st.text_input(
            "Time Intervals (minutes)",
            value="0,5,10",
            help="Comma-separated intervals in minutes (e.g., 0,5,10,15)"
        )

if st.button("🚀 Generate Real-Time Art", type="primary", use_container_width=True):
    if not text_prompt.strip():
        st.error("Please enter a text prompt!")
    elif not (include_weather or include_news or include_financial):
        st.error("Please enable at least one real-time data source!")
    else:
        try:
            if batch_mode and 'batch_configs' in locals() and batch_configs:
                # Batch processing
                with st.spinner(f"Generating art for {len(batch_configs)} data configurations... This may take several minutes."):
                    results = generator.batch_process_data_sources(
                        text_prompt=text_prompt,
                        style=style,
                        data_source_configs=batch_configs,
                        mood=mood,
                        context_strength=context_strength,
                        num_images=1,  # One per config for batch
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        weather_api_key=weather_api_key if weather_api_key else None,
                        news_api_key=news_api_key if news_api_key else None
                    )

                    st.success(f"✅ Generated {len(results)} images successfully!")

                    # Display batch results
                    cols = st.columns(min(len(results), 4))
                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            st.image(result["image"], caption=f"Config {i+1}", use_column_width=True)
                            st.caption(result["filename"])

                            # Show data sources for this result
                            sources = []
                            if result["metadata"]["data_sources"]["weather"]:
                                sources.append("🌤️")
                            if result["metadata"]["data_sources"]["news"]:
                                sources.append("📰")
                            if result["metadata"]["data_sources"]["financial"]:
                                sources.append("💹")
                            st.caption(f"Sources: {' '.join(sources)}")

            elif temporal_series:
                # Temporal series processing
                try:
                    interval_list = [int(x.strip()) for x in intervals.split(',')]

                    with st.spinner(f"Generating temporal series with {len(interval_list)} time points... This will take time."):
                        data_config = {
                            "include_weather": include_weather,
                            "weather_city": weather_city if include_weather else "New York",
                            "weather_api_key": weather_api_key if weather_api_key else None,
                            "include_news": include_news,
                            "news_category": news_category if include_news else "general",
                            "max_news": max_news if include_news else 3,
                            "news_api_key": news_api_key if news_api_key else None,
                            "include_financial": include_financial,
                            "mood": mood,
                            "context_strength": context_strength,
                            "num_images": 1,
                            "num_inference_steps": num_steps,
                            "guidance_scale": guidance_scale
                        }

                        results = generator.generate_temporal_series(
                            text_prompt=text_prompt,
                            style=style,
                            data_config=data_config,
                            time_intervals=interval_list
                        )

                        st.success(f"✅ Generated {len(results)} temporal images successfully!")

                        # Display temporal results
                        for i, result in enumerate(results):
                            st.subheader(f"Time Point {i+1} (T+{interval_list[i]} min)")
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.image(result["image"], caption=result["filename"], use_column_width=True)

                            with col2:
                                if result["realtime_context"]:
                                    st.write("**📊 Data Summary:**")
                                    st.write(result["realtime_context"].summary[:200] + "...")

                                    if result["realtime_context"].mood_indicators:
                                        st.write(f"**🎭 Mood:** {', '.join(result['realtime_context'].mood_indicators[:2])}")

                except ValueError:
                    st.error("Invalid time intervals format. Use comma-separated numbers (e.g., 0,5,10)")

            else:
                # Standard generation
                with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                    results = generator.generate_image(
                        text_prompt=text_prompt,
                        style=style,
                        mood=mood,
                        include_weather=include_weather,
                        weather_city=weather_city if include_weather else "New York",
                        weather_api_key=weather_api_key if weather_api_key else None,
                        include_news=include_news,
                        news_category=news_category if include_news else "general",
                        max_news=max_news if include_news else 3,
                        news_api_key=news_api_key if news_api_key else None,
                        include_financial=include_financial,
                        context_strength=context_strength,
                        num_images=num_images,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )

                    st.success(f"✅ Generated {len(results)} image(s) successfully!")

                    # Display results
                    if len(results) == 1:
                        result = results[0]
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.image(result["image"], caption=result["filename"], use_column_width=True)
                            st.info(f"💾 Saved as: `{result['filename']}.png`")

                        with col2:
                            if result["realtime_context"]:
                                st.markdown("**🌐 Real-Time Context:**")
                                st.write(f"📝 {result['realtime_context'].summary}")

                                if result["realtime_context"].mood_indicators:
                                    st.write(f"🎭 **Mood:** {', '.join(result['realtime_context'].mood_indicators)}")

                                if result["realtime_context"].key_themes:
                                    st.write(f"🎯 **Themes:** {', '.join(result['realtime_context'].key_themes)}")

                                st.write(f"🎨 **Inspiration:** {result['realtime_context'].artistic_inspiration}")
                    else:
                        # Multiple images
                        cols = st.columns(min(len(results), 3))
                        for i, result in enumerate(results):
                            with cols[i % len(cols)]:
                                st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                                st.caption(result["filename"])

                    # Show generation details
                    with st.expander("📋 Generation Details", expanded=False):
                        for i, result in enumerate(results):
                            st.subheader(f"Image {i+1}")
                            st.json({
                                "enhanced_prompt": result["metadata"]["enhanced_prompt"],
                                "data_sources": result["metadata"]["data_sources"],
                                "context_strength": result["metadata"]["context_strength"],
                                "generation_params": result["metadata"]["generation_params"]
                            })

                            if result["metadata"].get("realtime_context"):
                                st.write("**Real-Time Context:**")
                                context_summary = {
                                    "summary": result["metadata"]["realtime_context"]["summary"],
                                    "mood_indicators": result["metadata"]["realtime_context"]["mood_indicators"],
                                    "key_themes": result["metadata"]["realtime_context"]["key_themes"],
                                    "temporal_context": result["metadata"]["realtime_context"]["temporal_context"]
                                }
                                st.json(context_summary)

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.exception(e)

# ------------------------ FOOTER ------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>CompI Phase 2.D — Real-Time Data-to-Art</strong><br>
    Connect Art to the World's Pulse • Built with Stable Diffusion, Real-Time APIs & Streamlit<br>
    🌎 + 📡 + 🎨 = ✨ Living Art
</div>
""", unsafe_allow_html=True)

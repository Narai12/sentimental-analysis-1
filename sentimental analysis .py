import streamlit as st
import pandas as pd
import re
import base64
from datetime import datetime
from collections import Counter
import time
import numpy as np

# Try importing NLP libraries with error handling
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    nltk = None
    TextBlob = None
    SentimentIntensityAnalyzer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    VaderAnalyzer = None

# Try importing visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
    plt = None

# Page configuration
st.set_page_config(
    page_title="Ultimate Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PREMIUM THEMES COLLECTION ============
PREMIUM_THEMES = {
    "✨ Aurora Premium": {
        "bg": "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
        "card_bg": "rgba(255, 255, 255, 0.15)",
        "text": "#ffffff",
        "secondary": "#e0e0e0",
        "accent": "#ffd700"
    },
    "🌊 Ocean Pearl": {
        "bg": "linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #8e9eab 100%)",
        "card_bg": "rgba(255, 255, 255, 0.12)",
        "text": "#ffffff",
        "secondary": "#b8e1fc",
        "accent": "#00d2ff"
    },
    "🔥 Crimson Night": {
        "bg": "linear-gradient(135deg, #ff6b6b 0%, #c92a2a 50%, #862e2e 100%)",
        "card_bg": "rgba(255, 255, 255, 0.15)",
        "text": "#ffffff",
        "secondary": "#ffe0e0",
        "accent": "#ff4757"
    },
    "🌿 Emerald Forest": {
        "bg": "linear-gradient(135deg, #134e5e 0%, #71b280 50%, #2ecc71 100%)",
        "card_bg": "rgba(255, 255, 255, 0.12)",
        "text": "#ffffff",
        "secondary": "#d4edda",
        "accent": "#00b894"
    },
    "💜 Royal Velvet": {
        "bg": "linear-gradient(135deg, #2c3e50 0%, #8e44ad 50%, #3498db 100%)",
        "card_bg": "rgba(255, 255, 255, 0.12)",
        "text": "#ffffff",
        "secondary": "#e8daef",
        "accent": "#9b59b6"
    },
    "🌅 Sunset Paradise": {
        "bg": "linear-gradient(135deg, #fc4a1a 0%, #f7b733 50%, #f39c12 100%)",
        "card_bg": "rgba(255, 255, 255, 0.15)",
        "text": "#ffffff",
        "secondary": "#fff5e6",
        "accent": "#ffa502"
    },
    "🌸 Cherry Blossom": {
        "bg": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fbc2eb 100%)",
        "card_bg": "rgba(255, 255, 255, 0.85)",
        "text": "#2c3e50",
        "secondary": "#ff6b6b",
        "accent": "#ff6b6b"
    },
    "🌟 Midnight Galaxy": {
        "bg": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        "card_bg": "rgba(255, 255, 255, 0.08)",
        "text": "#ffffff",
        "secondary": "#c39bd3",
        "accent": "#e74c3c"
    },
    "🍃 Mint Breeze": {
        "bg": "linear-gradient(135deg, #00b4db 0%, #0083b0 50%, #00d2ff 100%)",
        "card_bg": "rgba(255, 255, 255, 0.14)",
        "text": "#ffffff",
        "secondary": "#d4f1f9",
        "accent": "#00a8ff"
    },
    "🎨 Candy Crush": {
        "bg": "linear-gradient(135deg, #f857a6 0%, #ff5858 50%, #fc4a1a 100%)",
        "card_bg": "rgba(255, 255, 255, 0.13)",
        "text": "#ffffff",
        "secondary": "#ffe0f0",
        "accent": "#ff6b6b"
    }
}


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data packages"""
    if not NLP_AVAILABLE:
        return False
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        return True
    except Exception:
        return False


nltk_success = download_nltk_data() if NLP_AVAILABLE else False


# Initialize VADER
@st.cache_resource
def get_vader_analyzer():
    """Initialize VADER sentiment analyzer"""
    if VADER_AVAILABLE and VaderAnalyzer:
        return VaderAnalyzer()
    elif NLP_AVAILABLE and nltk_success and SentimentIntensityAnalyzer:
        return SentimentIntensityAnalyzer()
    return None


vader = get_vader_analyzer()


# Initialize session state with enhanced features
def initialize_session_state():
    """Initialize all session state variables"""
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'total_analyses' not in st.session_state:
        st.session_state.total_analyses = 0
    if 'example_feedback' not in st.session_state:
        st.session_state.example_feedback = ""
    if 'current_theme' not in st.session_state:
        st.session_state.current_theme = "✨ Aurora Premium"
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = True
    if 'notifications' not in st.session_state:
        st.session_state.notifications = True
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'chart_animation' not in st.session_state:
        st.session_state.chart_animation = True
    if 'word_cloud_enabled' not in st.session_state:
        st.session_state.word_cloud_enabled = True


initialize_session_state()


# Apply premium theme
def apply_premium_theme(theme_name):
    """Apply selected premium theme"""
    theme = PREMIUM_THEMES.get(theme_name, PREMIUM_THEMES["✨ Aurora Premium"])

    # Escape the accent color for CSS
    accent_color = theme['accent']

    st.markdown(f"""
        <style>
        /* Premium Glass Morphism Effect */
        .stApp {{
            background: {theme['bg']};
            background-attachment: fixed;
        }}

        /* Main Header Styling */
        .main-header {{
            text-align: center;
            padding: 35px;
            background: {theme['card_bg']};
            backdrop-filter: blur(15px);
            color: {theme['text']};
            border-radius: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            animation: glowPulse 2s infinite;
        }}

        @keyframes glowPulse {{
            0% {{ box-shadow: 0 0 5px {accent_color}; }}
            50% {{ box-shadow: 0 0 20px {accent_color}; }}
            100% {{ box-shadow: 0 0 5px {accent_color}; }}
        }}

        .feedback-card {{
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            border-left: 5px solid {accent_color};
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            color: {theme['text']};
            position: relative;
            overflow: hidden;
        }}

        .feedback-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}

        .feedback-card:hover::before {{
            left: 100%;
        }}

        .feedback-card:hover {{
            transform: translateX(10px) translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }}

        .metric-card {{
            text-align: center;
            padding: 25px;
            border-radius: 20px;
            background: {theme['card_bg']};
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            color: {theme['text']};
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
        }}

        .metric-card:hover {{
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }}

        /* Stunning button animations */
        .stButton > button {{
            transition: all 0.3s ease;
            border-radius: 12px;
            font-weight: 600;
            background: linear-gradient(135deg, {accent_color}, {accent_color}dd);
            color: white;
            border: none;
            position: relative;
            overflow: hidden;
        }}

        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}

        .stButton > button:hover::before {{
            width: 300px;
            height: 300px;
        }}

        /* Animated stats */
        @keyframes countUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: {accent_color};
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {accent_color}dd;
        }}

        /* Floating animation for emojis */
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}

        .emoji-large {{
            font-size: 54px;
            animation: float 3s ease-in-out infinite;
            display: inline-block;
        }}

        /* Premium glass panel */
        .css-1d391kg, .css-1633sj2 {{
            background: {theme['card_bg']};
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        </style>
    """, unsafe_allow_html=True)


# Apply current theme
apply_premium_theme(st.session_state.current_theme)


# Enhanced sentiment analysis functions
def analyze_sentiment_keywords_enhanced(text):
    """Enhanced keyword analysis with context"""
    text_lower = text.lower()

    positive_terms = {
        'excellent': 1.0, 'amazing': 1.0, 'outstanding': 1.0, 'perfect': 0.9,
        'brilliant': 0.9, 'fantastic': 0.9, 'wonderful': 0.9, 'superb': 0.9,
        'great': 0.8, 'awesome': 0.9, 'love': 0.8, 'best': 0.8, 'good': 0.5
    }

    negative_terms = {
        'terrible': -1.0, 'horrible': -1.0, 'awful': -1.0, 'worst': -0.9,
        'hate': -0.9, 'useless': -0.9, 'pathetic': -0.9, 'bad': -0.6,
        'poor': -0.6
    }

    words = text_lower.split()
    score = 0.0
    count = 0

    for word in words:
        if word in positive_terms:
            score += positive_terms[word]
            count += 1
        elif word in negative_terms:
            score += negative_terms[word]
            count += 1

    polarity = score / count if count > 0 else 0.0
    return {'polarity': polarity}


def analyze_patterns(text):
    """Analyze emojis, punctuation, and caps for sentiment"""
    score = 0.0

    # Emoji analysis
    happy_emojis = ['😊', '😍', '🤩', '🎉', '❤️', '👍', '✨']
    sad_emojis = ['😞', '😢', '😡', '🤬', '👎', '💔']

    for emoji in happy_emojis:
        if emoji in text:
            score += 0.3
    for emoji in sad_emojis:
        if emoji in text:
            score -= 0.3

    # Excitement detection
    if text.count('!') > 1:
        score += 0.2
    if text.isupper() and len(text) > 10:
        score += 0.2

    return max(-0.5, min(0.5, score))


def analyze_sentiment_advanced(text):
    """Advanced ensemble sentiment analysis with 4 methods"""
    if not text or not text.strip():
        return {
            'sentiment': 'Neutral',
            'emoji': '😐',
            'polarity': 0.0,
            'subjectivity': 0.0,
            'confidence': 0.0,
            'intensity': 'Neutral',
            'vader_score': 0.0,
            'textblob_score': 0.0
        }

    # Method 1: VADER
    if vader:
        scores = vader.polarity_scores(text)
        vader_score = scores['compound']
    else:
        vader_score = 0.0

    # Method 2: TextBlob
    try:
        if NLP_AVAILABLE and TextBlob:
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            textblob_score = 0.0
            subjectivity = 0.5
    except Exception:
        textblob_score = 0.0
        subjectivity = 0.5

    # Method 3: Keyword analysis
    keyword_result = analyze_sentiment_keywords_enhanced(text)

    # Method 4: Pattern-based (emoji & punctuation)
    pattern_score = analyze_patterns(text)

    # Ensemble with weights
    final_polarity = (vader_score * 0.35 + textblob_score * 0.25 +
                      keyword_result['polarity'] * 0.25 + pattern_score * 0.15)

    # Calculate confidence based on agreement
    scores_list = [vader_score, textblob_score, keyword_result['polarity'], pattern_score]
    std_dev = np.std(scores_list)
    confidence = max(50.0, 100.0 - (std_dev * 100.0))
    confidence = min(confidence, 98.0)

    # Determine sentiment with intensity
    if final_polarity > 0.15:
        sentiment = "Positive"
        if final_polarity > 0.6:
            emoji = "🤩"
            intensity = "Very Strong"
        elif final_polarity > 0.4:
            emoji = "😍"
            intensity = "Strong"
        else:
            emoji = "😊"
            intensity = "Mild"
    elif final_polarity < -0.15:
        sentiment = "Negative"
        if final_polarity < -0.6:
            emoji = "🤬"
            intensity = "Very Strong"
        elif final_polarity < -0.4:
            emoji = "😡"
            intensity = "Strong"
        else:
            emoji = "😞"
            intensity = "Mild"
    else:
        sentiment = "Neutral"
        emoji = "😐"
        intensity = "Neutral"

    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'polarity': round(final_polarity, 3),
        'subjectivity': round(subjectivity, 3),
        'confidence': round(confidence, 2),
        'intensity': intensity,
        'vader_score': round(vader_score, 3),
        'textblob_score': round(textblob_score, 3)
    }


def extract_keywords_smart(text):
    """Smart keyword extraction with context"""
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}

    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(keywords).most_common(10)

    # Add context-based emojis
    context_emojis = {
        'happy': '😊', 'love': '❤️', 'great': '👍', 'amazing': '✨',
        'bad': '👎', 'terrible': '💔', 'angry': '😠', 'sad': '😢'
    }

    enhanced = []
    for word, count in word_counts:
        emoji = next((v for k, v in context_emojis.items() if k in word), '')
        display_word = f"{emoji} {word}" if emoji else word
        enhanced.append((display_word, count))

    return enhanced


def generate_smart_insights():
    """Generate AI-powered insights from feedback"""
    if not st.session_state.feedback_history:
        return []

    df = pd.DataFrame(st.session_state.feedback_history)
    insights = []

    # Overall sentiment insight
    positive_pct = (df['sentiment'] == 'Positive').mean() * 100
    if positive_pct > 70:
        insights.append("🎉 **Excellent!** Customer satisfaction is very high")
    elif positive_pct > 50:
        insights.append("📈 **Good progress** - More than half of customers are satisfied")
    elif positive_pct > 30:
        insights.append("⚠️ **Needs attention** - Consider improving customer experience")
    else:
        insights.append("🚨 **Urgent action required** - Customer satisfaction is critically low")

    # Trend insight
    if len(df) > 5:
        recent_avg = df.tail(5)['polarity'].mean()
        overall_avg = df['polarity'].mean()
        if recent_avg > overall_avg + 0.1:
            insights.append("📊 **Positive trend detected** - Recent feedback is improving!")
        elif recent_avg < overall_avg - 0.1:
            insights.append("📉 **Negative trend detected** - Recent feedback is declining")
        else:
            insights.append("📊 **Stable trend** - Sentiment consistency maintained")

    # Confidence insight
    avg_confidence = df['confidence'].mean()
    if avg_confidence > 80:
        insights.append("🎯 **High confidence** - Analysis results are very reliable")
    elif avg_confidence < 60:
        insights.append("🔍 **Consider reviewing** - Some analyses have low confidence")

    return insights


# Apply theme and show dashboard
st.markdown(f"""
    <div class="main-header">
        <h1>🎯 Ultimate Sentiment Analysis Dashboard</h1>
        <p>Advanced AI-Powered Multi-Model Feedback Classification System</p>
        <p style="font-size: 14px; margin-top: 10px;">✨ 4 Analysis Methods • Real-time Insights • Premium Themes ✨</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with premium features
with st.sidebar:
    st.markdown("### 🎨 Premium Theme Gallery")

    # Theme selector with preview
    theme_options = list(PREMIUM_THEMES.keys())

    for i, theme_name in enumerate(theme_options):
        if st.button(f"{theme_name}", key=f"theme_{i}", use_container_width=True):
            st.session_state.current_theme = theme_name
            apply_premium_theme(theme_name)
            st.rerun()

    st.markdown("---")

    # Advanced settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        auto_analyze = st.toggle("🚀 Auto-Analyze Mode", st.session_state.auto_analyze)
        if auto_analyze != st.session_state.auto_analyze:
            st.session_state.auto_analyze = auto_analyze

        notifications = st.toggle("🔔 Smart Notifications", st.session_state.notifications)
        if notifications != st.session_state.notifications:
            st.session_state.notifications = notifications

        chart_animation = st.toggle("🎬 Chart Animations", st.session_state.chart_animation)
        if chart_animation != st.session_state.chart_animation:
            st.session_state.chart_animation = chart_animation

        word_cloud_enabled = st.toggle("☁️ Word Cloud", st.session_state.word_cloud_enabled)
        if word_cloud_enabled != st.session_state.word_cloud_enabled:
            st.session_state.word_cloud_enabled = word_cloud_enabled

    st.markdown("---")

    # Mode selection
    analysis_mode = st.radio(
        "📋 Select Mode",
        ["🎯 Smart Analysis", "📊 Batch Analysis", "📈 Analytics Hub", "💡 Insights Engine"],
        index=0
    )

    st.markdown("---")

    # Live statistics
    st.markdown("### 📊 Live Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Analyses", st.session_state.total_analyses)
    with col2:
        if st.session_state.feedback_history:
            positive_count = sum(1 for fb in st.session_state.feedback_history if fb['sentiment'] == 'Positive')
            satisfaction = (positive_count / len(st.session_state.feedback_history) * 100)
            st.metric("Satisfaction", f"{satisfaction:.0f}%")
        else:
            st.metric("Satisfaction", "0%")

    # Export feature
    if st.session_state.feedback_history:
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        df_export = pd.DataFrame(st.session_state.feedback_history)
        csv = df_export.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis.csv">📥 Download CSV Report</a>',
                    unsafe_allow_html=True)

# Main content based on mode
if analysis_mode == "🎯 Smart Analysis":
    st.header("🎯 Smart Sentiment Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        feedback_text = st.text_area(
            "✍️ Enter Customer Feedback",
            placeholder="Example: I absolutely love this product! The quality is amazing and customer service is outstanding! 🎉",
            height=120,
            key="smart_input",
            value=st.session_state.get('example_feedback', '')
        )

        customer_info = st.text_input("👤 Customer Name (Optional)", placeholder="Enter customer name...")

        if st.button("🚀 Analyze Now", type="primary", use_container_width=True):
            if feedback_text and feedback_text.strip():
                with st.spinner("🧠 Analyzing with 4 AI models..."):
                    result = analyze_sentiment_advanced(feedback_text)
                    keywords = extract_keywords_smart(feedback_text)

                    # Save to history
                    feedback_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'customer': customer_info if customer_info else "Anonymous",
                        'text': feedback_text,
                        'sentiment': result['sentiment'],
                        'emoji': result['emoji'],
                        'polarity': result['polarity'],
                        'subjectivity': result['subjectivity'],
                        'confidence': result['confidence'],
                        'intensity': result.get('intensity', 'Medium')
                    }
                    st.session_state.feedback_history.append(feedback_record)
                    st.session_state.total_analyses += 1

                    # Display results with animations
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")

                    # Animated metrics
                    col_a, col_b, col_c, col_d = st.columns(4)

                    with col_a:
                        sentiment_color = "#28a745" if result['sentiment'] == "Positive" else "#dc3545" if result[
                                                                                                               'sentiment'] == "Negative" else "#ffc107"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="emoji-large">{result['emoji']}</div>
                                <h3 style="color: {sentiment_color};">{result['sentiment']}</h3>
                                <p>Sentiment</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 48px;">🎯</div>
                                <h2>{result['confidence']:.1f}%</h2>
                                <p>Confidence</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_c:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 48px;">📊</div>
                                <h2>{result['polarity']:.2f}</h2>
                                <p>Polarity Score</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_d:
                        if result.get('intensity') == "Very Strong":
                            intensity_icon = "🔥"
                        elif result.get('intensity') == "Strong":
                            intensity_icon = "⚡"
                        else:
                            intensity_icon = "💧"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 48px;">{intensity_icon}</div>
                                <h3>{result.get('intensity', 'Medium')}</h3>
                                <p>Intensity</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Model comparison chart
                    if PLOTLY_AVAILABLE and go and 'vader_score' in result:
                        st.markdown("---")
                        st.subheader("🔬 Model Performance Comparison")

                        models = ['VADER', 'TextBlob', 'Keyword', 'Pattern']
                        scores = [
                            result.get('vader_score', 0),
                            result.get('textblob_score', 0),
                            analyze_sentiment_keywords_enhanced(feedback_text)['polarity'],
                            analyze_patterns(feedback_text)
                        ]

                        fig = go.Figure(data=[
                            go.Bar(name='Scores', x=models, y=scores,
                                   marker_color=['#667eea', '#764ba2', '#f093fb', '#4a90e2'],
                                   text=[f'{s:.3f}' for s in scores], textposition='auto')
                        ])
                        fig.update_layout(
                            title="Multi-Model Analysis Results",
                            yaxis_title="Sentiment Score",
                            yaxis_range=[-1, 1],
                            showlegend=False,
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Keywords
                    if keywords:
                        st.markdown("---")
                        st.subheader("🏷️ Key Insights & Keywords")
                        cols = st.columns(5)
                        for i, (word, count) in enumerate(keywords[:10]):
                            with cols[i % 5]:
                                st.info(f"**{word}**\n\n*{count} occurrences*")

                    st.success("✅ Analysis complete with 4 AI models!")
                    st.balloons()
            else:
                st.warning("⚠️ Please enter feedback to analyze")

    with col2:
        st.info("""
        ### 🚀 Premium Features

        **4 AI Analysis Models:**
        1. 🎯 **VADER** - Social media optimized
        2. 📚 **TextBlob** - Linguistic analysis
        3. 🔤 **Keyword** - Context-aware
        4. 🎨 **Pattern** - Emoji & punctuation

        ### ✨ Smart Features
        - **Real-time analysis**
        - **Sentiment intensity detection**
        - **Emotion recognition**
        - **Context understanding**
        - **Multi-model consensus**

        ### 💡 Pro Tips
        - Add emojis for better accuracy
        - Use natural language
        - Include intensity words (very, extremely)
        - Get detailed model breakdowns
        """)

        # Quick test cards
        st.markdown("---")
        st.markdown("### 🧪 Quick Test Examples")
        examples = [
            ("🤩 Amazing! Best product ever!", "Very Positive"),
            ("😡 Terrible service, very disappointed", "Very Negative"),
            ("😐 It's okay, nothing special", "Neutral"),
            ("❤️ Love this! Excellent quality!", "Very Positive")
        ]

        for text, _ in examples:
            if st.button(f"{text}", key=f"example_{text[:20]}", use_container_width=True):
                st.session_state.example_feedback = text
                st.rerun()

elif analysis_mode == "📊 Batch Analysis":
    st.header("📊 Batch Analysis Engine")

    col1, col2 = st.columns([2, 1])

    with col1:
        batch_data = st.text_area(
            "📝 Paste Multiple Feedbacks (One per line)",
            placeholder="Product is amazing! 🎉\nService needs improvement\nLove the quality! ❤️\nCould be better\nExcellent customer support!",
            height=250
        )

        if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
            if batch_data and batch_data.strip():
                feedbacks = [line.strip() for line in batch_data.split('\n') if line.strip()]

                if feedbacks:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for idx, feedback in enumerate(feedbacks):
                        status_text.text(f"Analyzing {idx + 1}/{len(feedbacks)}: {feedback[:50]}...")
                        result = analyze_sentiment_advanced(feedback)

                        results.append({
                            'Feedback': feedback[:80] + "..." if len(feedback) > 80 else feedback,
                            'Sentiment': f"{result['emoji']} {result['sentiment']}",
                            'Polarity': result['polarity'],
                            'Confidence': f"{result['confidence']:.0f}%",
                            'Intensity': result.get('intensity', 'Medium')
                        })

                        # Save to history
                        st.session_state.feedback_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'customer': 'Batch',
                            'text': feedback,
                            'sentiment': result['sentiment'],
                            'emoji': result['emoji'],
                            'polarity': result['polarity'],
                            'subjectivity': result['subjectivity'],
                            'confidence': result['confidence']
                        })
                        st.session_state.total_analyses += 1

                        progress_bar.progress((idx + 1) / len(feedbacks))
                        time.sleep(0.05)

                    status_text.text("✅ Analysis complete!")

                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Batch Results Summary")

                    df_results = pd.DataFrame(results)

                    # Statistics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Total Feedbacks", len(results))
                    with col_s2:
                        positive_count = sum(1 for r in results if 'Positive' in r['Sentiment'])
                        st.metric("Positive Feedbacks", positive_count)
                    with col_s3:
                        confidence_values = [float(r['Confidence'].replace('%', '')) for r in results]
                        avg_confidence = sum(confidence_values) / len(results) if confidence_values else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

                    # Results table
                    st.dataframe(df_results, use_container_width=True)

                    # Visualization
                    if PLOTLY_AVAILABLE and px:
                        sentiment_counts = pd.Series([r['Sentiment'].split()[1] for r in results]).value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'},
                            hole=0.4
                        )
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)

                    st.success(f"✅ Successfully analyzed {len(feedbacks)} feedbacks!")
                    st.balloons()

    with col2:
        st.info("""
        ### ⚡ Batch Features

        - **Bulk processing** - Analyze 100+ feedbacks
        - **Progress tracking** - Real-time updates
        - **Summary statistics** - Instant insights
        - **Export ready** - CSV format included

        ### 📈 Best Practices
        - One feedback per line
        - Include emojis for context
        - Vary length for accuracy
        - Review low-confidence items
        """)

elif analysis_mode == "📈 Analytics Hub":
    st.header("📈 Advanced Analytics Hub")

    if st.session_state.feedback_history:
        df = pd.DataFrame(st.session_state.feedback_history)

        # KPI Dashboard
        st.markdown("### 🎯 Key Performance Indicators")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.metric("Total Feedback", len(df))
        with kpi2:
            positive_pct = (df['sentiment'] == 'Positive').mean() * 100
            st.metric("Customer Satisfaction", f"{positive_pct:.1f}%",
                      delta="Good" if positive_pct > 60 else "Needs Work")
        with kpi3:
            avg_polarity = df['polarity'].mean()
            st.metric("Avg Polarity", f"{avg_polarity:.3f}")
        with kpi4:
            high_conf = (df['confidence'] > 80).mean() * 100
            st.metric("High Confidence", f"{high_conf:.1f}%")

        # Advanced visualizations
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎨 Distribution", "🔬 Correlation"])

        with tab1:
            if PLOTLY_AVAILABLE and px:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date

                # Daily trend
                daily_trend = df.groupby('date')['polarity'].mean().reset_index()
                fig = px.line(daily_trend, x='date', y='polarity',
                              title="Sentiment Trend Over Time",
                              markers=True)
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

                # Cumulative
                df['cumulative'] = df['polarity'].cumsum()
                fig2 = px.area(df, x='timestamp', y='cumulative',
                               title="Cumulative Sentiment Score",
                               color_discrete_sequence=['#667eea'])
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                if PLOTLY_AVAILABLE and px:
                    fig = px.histogram(df, x='polarity', nbins=30,
                                       title="Polarity Distribution",
                                       color_discrete_sequence=['#764ba2'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if PLOTLY_AVAILABLE and px:
                    sentiment_counts = df['sentiment'].value_counts()
                    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                                 title="Sentiment Breakdown",
                                 color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107',
                                                     'Negative': '#dc3545'})
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if PLOTLY_AVAILABLE and px and go:
                # Correlation matrix
                corr = df[['polarity', 'subjectivity', 'confidence']].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Feature Correlation Matrix",
                                color_continuous_scale='RdBu')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        # Word Cloud
        if WORDCLOUD_AVAILABLE and WordCloud and plt and st.session_state.word_cloud_enabled:
            st.markdown("---")
            st.subheader("☁️ Feedback Word Cloud")
            all_text = ' '.join(df['text'])
            wordcloud = WordCloud(width=1000, height=500, background_color='white',
                                  colormap='viridis', max_words=100).generate(all_text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

    else:
        st.info("📭 No data available. Start analyzing feedback to see analytics!")

else:  # Insights Engine
    st.header("💡 AI-Powered Insights Engine")

    if st.session_state.feedback_history:
        insights = generate_smart_insights()

        st.markdown("### 🤖 Artificial Intelligence Insights")

        for insight in insights:
            st.success(insight)

        st.markdown("---")

        # Recommendation engine
        st.markdown("### 📋 Smart Recommendations")
        df = pd.DataFrame(st.session_state.feedback_history)

        # Find common issues
        negative_feedbacks = df[df['sentiment'] == 'Negative']
        if len(negative_feedbacks) > 0:
            st.markdown("#### 🔍 Common Issues Detected")
            negative_text = ' '.join(negative_feedbacks['text'].tolist())
            words = re.findall(r'\b\w+\b', negative_text.lower())
            common_words = Counter(words).most_common(10)
            displayed = 0
            for word, count in common_words[:5]:
                if word not in ['the', 'and', 'for', 'that', 'this', 'was', 'with']:
                    st.warning(f"• Issue: **{word}** mentioned {count} times")
                    displayed += 1
            if displayed == 0:
                st.info("No specific issues identified")

        # Positive trends
        positive_feedbacks = df[df['sentiment'] == 'Positive']
        if len(positive_feedbacks) > 0:
            st.markdown("#### 🌟 What's Working Well")
            st.info("Customers appreciate the positive aspects of your service!")

            # Top positive words
            positive_text = ' '.join(positive_feedbacks['text'].tolist())
            words = re.findall(r'\b\w+\b', positive_text.lower())
            common_words = Counter(words).most_common(10)
            for word, count in common_words[:3]:
                if word in ['love', 'great', 'amazing', 'excellent', 'perfect', 'good']:
                    st.success(f"• **{word}** is resonating well with customers")

        # Actionable insights
        st.markdown("---")
        st.markdown("### 🚀 Actionable Recommendations")

        if len(df) > 0:
            avg_confidence = df['confidence'].mean()
            if avg_confidence < 70:
                st.warning("📝 **Improve data quality**: Provide more detailed feedback for better analysis")

            positive_rate = (df['sentiment'] == 'Positive').mean()
            if positive_rate < 0.5:
                st.error("🎯 **Urgent**: Customer satisfaction is below 50%. Immediate action recommended")
            elif positive_rate < 0.7:
                st.info("📈 **Opportunity**: Focus on improving customer experience to boost satisfaction")

            # Performance trends
            if len(df) > 5:
                recent = df.tail(5)['polarity'].mean()
                previous = df.head(-5)['polarity'].mean() if len(df) > 10 else df['polarity'].mean()
                if recent > previous:
                    st.success("📈 **Positive trend**: Recent improvements are working well!")
                elif recent < previous:
                    st.warning("📉 **Declining trend**: Recent changes may need review")

        # Export insights
        st.markdown("---")
        if st.button("📊 Generate Full Report", use_container_width=True):
            report_lines = [
                f"SENTIMENT ANALYSIS REPORT",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"TOTAL FEEDBACKS: {len(df)}",
                f"POSITIVE RATE: {(df['sentiment'] == 'Positive').mean() * 100:.1f}%",
                f"AVG CONFIDENCE: {df['confidence'].mean():.1f}%",
                f"AVG POLARITY: {df['polarity'].mean():.3f}",
                "",
                "KEY INSIGHTS:",
            ]
            report_lines.extend(insights)
            report = "\n".join(report_lines)
            st.download_button("💾 Download Report", report, file_name="insights_report.txt")
    else:
        st.info("📭 Insufficient data for insights. Please add more feedback first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>🚀 Ultimate Sentiment Analysis Dashboard | Powered by 4 AI Models</p>
        <p>✨ VADER • TextBlob • Keyword Analysis • Pattern Recognition ✨</p>
        <p>🎨 10 Premium Themes • Real-time Analytics • Smart Insights • Enterprise Ready 🎨</p>
    </div>
""", unsafe_allow_html=True)
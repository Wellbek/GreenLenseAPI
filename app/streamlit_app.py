import streamlit as st
import requests
import json
import time
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="GreenLense - Product Insights",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern AI/UX platform with enhanced visuals
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Enhanced headers with gradient text */
    h1, h2, h3 {
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600 !important;
    }
    
    /* Modern glassmorphism cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(52, 211, 153, 0.2);
    }
    
    /* Metric containers with gradient backgrounds */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        font-size: 0.875rem !important;
        letter-spacing: 0.05em;
    }
    
    /* Modern button design */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    
    /* Enhanced input fields */
    .stTextInput>div>div>input {
        background: rgba(15, 23, 42, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    /* Enhanced score badges with glow effect */
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .score-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .score-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .score-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3);
    }
    
    /* Modern tabs design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 0.25rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    /* Enhanced expander */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.7);
        color: #60a5fa !important;
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(15, 23, 42, 0.9);
        border-color: rgba(52, 211, 153, 0.2);
    }
    
    /* Gradient progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 4px;
    }
    
    /* Floating mini badge with pulse animation */
    .mini-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(16px);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem;
        cursor: pointer;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
        max-width: 280px;
        z-index: 1000;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.5); }
        100% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3); }
    }
    
    .mini-badge:hover {
        transform: translateY(-4px);
        border-color: #34d399;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.4);
    }
    
    /* Hero section styling */
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInUp 1s ease;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.25rem;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Enhanced loading spinner */
    .stSpinner {
        color: #10b981 !important;
    }
    
    /* Product card enhancements */
    .product-showcase {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .product-showcase::before {load_mock_data
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    
    /* Notification styles */
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #34d399;
    }
    
    .error-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #f87171;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .mini-badge { bottom: 10px; right: 10px; padding: 1rem; max-width: 220px; }
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = os.getenv("API_URL", "http://api-gateway:8000")

# Cache for API responses
if 'api_cache' not in st.session_state:
    st.session_state['api_cache'] = {}

def get_score_color(score, max_score=5.0):
    """Return color class based on score"""
    normalized = score / max_score
    if normalized >= 0.7:
        return "score-high"
    elif normalized >= 0.4:
        return "score-medium"
    else:
        return "score-low"

def get_risk_color(risk):
    """Return color class based on risk level"""
    risk_map = {
        "low": "score-high",
        "medium": "score-medium",
        "high": "score-low"
    }
    return risk_map.get(risk.lower(), "score-medium")

def render_score_card(title, score, confidence, max_score=5.0, is_risk=False):
    """Render a modern score card with confidence"""
    if is_risk:
        color_class = get_risk_color(score)
        display_score = score.upper()
    else:
        color_class = get_score_color(score, max_score)
        display_score = f"{score}/{max_score}"
    
    confidence_pct = int(confidence * 100)
    
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color: #94a3b8; margin-bottom: 1rem; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">{title}</h3>
        <div class="score-badge {color_class}" style="font-size: 1.25rem; margin-bottom: 1rem;">
            {display_score}
        </div>
        <div style="color: #64748b; font-size: 0.875rem; font-weight: 500;">
            <div style="background: rgba(148, 163, 184, 0.1); height: 4px; border-radius: 2px; margin-bottom: 0.5rem;">
                <div style="background: linear-gradient(90deg, #10b981 0%, #34d399 100%); height: 100%; width: {confidence_pct}%; border-radius: 2px;"></div>
            </div>
            Confidence: {confidence_pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard(data):
    """Render the main analysis dashboard with enhanced visuals"""
    st.markdown('<h1 class="hero-title">🌿 GreenLense Analysis</h1>', unsafe_allow_html=True)
    
    # Overview metrics with modern layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_score_card("Durability", data["durability_score"], data["durability_confidence"])
    
    with col2:
        render_score_card("Quality", data["quality_score"], data["quality_confidence"])
    
    with col3:
        render_score_card("Sustainability", data["sustainability_score"], data["sustainability_confidence"])
    
    with col4:
        render_score_card("Allergen Risk", data["allergen_risk"], data["allergen_confidence"], is_risk=True)
    
    # Overall confidence with enhanced visualization
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Overall Confidence")
        st.progress(data["overall_confidence"])
    with col2:
        st.metric("Confidence Score", f"{int(data['overall_confidence'] * 100)}%")
    
    # Detailed explanations
    st.markdown("## 📊 Detailed Analysis")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for i, explanation in enumerate(data["explanation"], 1):
        st.markdown(f"**{i}.** {explanation}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Product summary with enhanced layout
    st.markdown("## 🏷️ Product Information")
    summary = data["product_summary"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Product Name:** {summary['name']}")
        st.markdown(f"**Brand:** {summary['brand']}")
        st.markdown(f"**Category:** {summary['category']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if summary.get("materials"):
            with st.expander("🧵 Materials & Composition"):
                for material in summary["materials"]:
                    st.markdown(f"• {material}")
        
        if summary.get("durability_indicators"):
            with st.expander("💪 Durability Indicators"):
                for indicator in summary["durability_indicators"]:
                    st.markdown(f"• {indicator}")
    
    with col2:
        if summary.get("safety_information"):
            with st.expander("🛡️ Safety Information"):
                for info in summary["safety_information"]:
                    st.markdown(f"• {info}")
        
        if summary.get("manufacturing_details"):
            with st.expander("🏭 Manufacturing Details"):
                for detail in summary["manufacturing_details"]:
                    st.markdown(f"• {detail}")
        
        if summary.get("allergen_warnings"):
            with st.expander("⚠️ Allergen Warnings"):
                if summary["allergen_warnings"]:
                    for warning in summary["allergen_warnings"]:
                        st.markdown(f"• {warning}")
                else:
                    st.success("No allergen warnings detected")
    
    # Data sources
    st.markdown("## 📈 Analysis Overview")
    st.info(f"🔍 Analysis based on **{data['total_items_identified']}** identified data points")

def load_mock_data():
    """Load mock data from file"""
    mock_file = Path("./data/coupang/mock_api.json")
    if mock_file.exists():
        with open(mock_file, 'r', encoding='utf-8') as f:
            mock_data = json.load(f)
            
        # Extract product information from mock data
        product_data = mock_data["data"]
        item = product_data["items"][0]
        reviews = item["reviews"]
        
        # Calculate scores based on mock data
        review_summary = item["reviewSummary"]
        avg_rating = review_summary["averageRating"]
        
        # Extract materials from notices
        materials = []
        for notice in item["notices"]:
            if notice["noticeCategoryDetailName"] == "제품소재":
                materials.append(notice["content"])
            elif notice["noticeCategoryDetailName"] == "색상":
                materials.append(f"색상: {notice['content']}")
        
        # Extract durability indicators from reviews
        durability_indicators = []
        quality_indicators = []
        for review in reviews:
            if "품질" in review["content"] or "quality" in review.get("attributes", {}):
                quality_indicators.append(f"Review rating: {review['rating']}/5")
            if "형태" in review["content"] or "durability" in review.get("attributes", {}):
                durability_indicators.append("형태 유지 우수")
        
        # Manufacturing details from notices
        manufacturing_details = []
        safety_info = []
        for notice in item["notices"]:
            if notice["noticeCategoryDetailName"] == "제조국":
                manufacturing_details.append(f"제조국: {notice['content']}")
            elif notice["noticeCategoryDetailName"] == "제조연월":
                manufacturing_details.append(f"제조일: {notice['content']}")
            elif notice["noticeCategoryDetailName"] == "취급시 주의사항":
                safety_info.append(notice["content"])
        
        return {
            "durability_score": min(5.0, avg_rating + 0.3),
            "durability_confidence": 0.9,
            "allergen_risk": "low",
            "allergen_confidence": 0.95,
            "quality_score": avg_rating,
            "quality_confidence": 0.85,
            "sustainability_score": 3.5,
            "sustainability_confidence": 0.7,
            "explanation": [
                f"Durability scored {min(5.0, avg_rating + 0.3)}/5 based on {review_summary['totalReviews']} customer reviews and material analysis",
                f"Quality rating of {avg_rating}/5 derived from verified purchase reviews",
                "Allergen risk assessed as low - no allergen warnings found in product notices",
                "Sustainability score moderate due to cotton-polyester blend and domestic manufacturing"
            ],
            "overall_confidence": 0.87,
            "total_items_identified": len(item["notices"]) + len(reviews) + len(item["attributes"]),
            "product_summary": {
                "name": product_data["displayProductName"],
                "brand": product_data["brand"],
                "category": product_data["productGroup"],
                "materials": materials,
                "durability_indicators": durability_indicators or ["고중량 원단", "면혼방 소재"],
                "allergen_warnings": [],
                "safety_information": safety_info,
                "manufacturing_details": manufacturing_details
            }
        }
    
    return None

def make_api_request(url, use_mock=False):
    """Make API request with caching"""
    cache_key = f"{url}_{use_mock}"
    
    if cache_key in st.session_state['api_cache']:
        return st.session_state['api_cache'][cache_key]
    
    try:
        endpoint = "/analyze-product-mock" if use_mock else "/analyze-product"
        response = requests.post(
            f"{API_URL}{endpoint}",
            json={"url": url},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state['api_cache'][cache_key] = result
            return result
        else:
            return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Main app
st.markdown('<div class="hero-title">🌿 GreenLense</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Sustainable Product Analysis Platform</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyze Product", "🛒 Product Showcase"])

with tab1:
    st.markdown("## Enter Product URL for Analysis")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input(
            "Product URL", 
            placeholder="https://www.example.com/product/...",
            help="Enter any product URL for comprehensive AI analysis"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze", use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        use_mock = st.checkbox("🧪 Use mock data (for testing)")
    with col4:
        if st.button("🗑️ Clear Cache"):
            st.session_state['api_cache'] = {}
            st.success("Cache cleared!")
    
    if analyze_btn and url:
        with st.spinner("🔍 Analyzing product with AI..."):
            if use_mock:
                mock_data = load_mock_data()
                if mock_data:
                    st.session_state['analysis_data'] = mock_data
                    st.markdown('<div class="success-message">✅ Analysis complete using mock data!</div>', unsafe_allow_html=True)
                else:
                    st.error("Mock data file not found")
            else:
                result = make_api_request(url, use_mock)
                if result:
                    st.session_state['analysis_data'] = result
                    st.markdown('<div class="success-message">✅ AI analysis complete!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ Analysis failed. Please try again or use mock data.</div>', unsafe_allow_html=True)
    
    if 'analysis_data' in st.session_state:
        st.markdown("---")
        render_dashboard(st.session_state['analysis_data'])

with tab2:
    st.markdown("## 🛒 Featured Product")
    
    mock_data = load_mock_data()
    if mock_data:
        summary = mock_data["product_summary"]
        
        # Enhanced product showcase
        st.markdown('<div class="product-showcase">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%); 
                        border: 2px dashed rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 2rem; 
                        height: 300px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 5rem; filter: drop-shadow(0 4px 8px rgba(16, 185, 129, 0.3));">👕</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {summary['name']}")
            st.markdown(f"**Brand:** {summary['brand']}")
            st.markdown(f"**Category:** {summary['category']}")
            
            # Enhanced pricing
            col_price1, col_price2 = st.columns([1, 1])
            with col_price1:
                st.markdown("#### ₩25,330")
                st.markdown("~~₩29,800~~ 15% OFF", help="Limited time offer")
            
            # AI-powered scores
            st.markdown("#### 🤖 AI Quality Scores")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Durability", f"{mock_data['durability_score']:.1f}/5", 
                         delta=f"{int(mock_data['durability_confidence']*100)}% confidence")
            with col_b:
                st.metric("Quality", f"{mock_data['quality_score']:.1f}/5",
                         delta=f"{int(mock_data['quality_confidence']*100)}% confidence")
            with col_c:
                st.metric("Sustainability", f"{mock_data['sustainability_score']:.1f}/5",
                         delta=f"{int(mock_data['sustainability_confidence']*100)}% confidence")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.button("🛒 Add to Cart", use_container_width=True)
            with col_btn2:
                if st.button("💚 View AI Analysis", use_container_width=True):
                    st.session_state['show_full'] = not st.session_state.get('show_full', False)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.get('show_full', False):
            st.markdown("---")
            render_dashboard(mock_data)
        
        # Enhanced product information
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📦 Product Details"):
                st.markdown("Premium quality cotton blend sweatshirt with exceptional comfort and durability. Perfect for casual wear in spring and fall seasons.")
                
            with st.expander("🧵 Materials & Composition"):
                if summary.get("materials"):
                    for material in summary["materials"]:
                        st.markdown(f"• {material}")
        
        with col2:
            with st.expander("⭐ Customer Reviews (4.5/5 Stars)"):
                st.markdown("**⭐⭐⭐⭐⭐** *Great quality!* - The fabric feels premium and fits perfectly.")
                st.markdown("**⭐⭐⭐⭐⭐** *Very comfortable* - Perfect for daily wear, exactly as described.")
                st.markdown("**⭐⭐⭐⭐** *Runs large* - Order one size down for better fit, but great quality.")
                
            with st.expander("🚚 Shipping & Returns"):
                st.markdown("• **Free shipping** on orders over ₩19,800")
                st.markdown("• **1-day shipping** available")
                st.markdown("• **30-day returns** with ₩5,000 return shipping")

# Floating mini sustainability badge
if st.session_state.get('show_full', False):
    st.markdown("""
    <div class="mini-badge" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="font-size: 1.5rem;">🌿</div>
            <div>
                <div style="font-weight: 600; color: #34d399;">GreenLense</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">AI Analysis Active</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

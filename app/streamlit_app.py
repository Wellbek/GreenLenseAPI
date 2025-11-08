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

# Custom CSS for Docker-style dark theme with green accents
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #7ee787 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid #2ea043;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #3fb950;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    /* Cards/Containers */
    .product-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .score-high {
        background-color: #238636;
        color: white;
    }
    
    .score-medium {
        background-color: #d29922;
        color: white;
    }
    
    .score-low {
        background-color: #da3633;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #161b22;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0d1117;
        color: #8b949e;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #58a6ff !important;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #238636;
    }
    
    /* Mini product badge */
    .mini-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #161b22;
        border: 2px solid #238636;
        border-radius: 8px;
        padding: 1rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);
        max-width: 250px;
        z-index: 1000;
    }
    
    .mini-badge:hover {
        border-color: #2ea043;
        box-shadow: 0 6px 16px rgba(35, 134, 54, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
    """Render a score card with confidence"""
    if is_risk:
        color_class = get_risk_color(score)
        display_score = score.upper()
    else:
        color_class = get_score_color(score, max_score)
        display_score = f"{score}/{max_score}"
    
    confidence_pct = int(confidence * 100)
    
    st.markdown(f"""
    <div class="product-card">
        <h3 style="color: #8b949e; margin-bottom: 0.5rem;">{title}</h3>
        <div class="score-badge {color_class}" style="font-size: 1.5rem;">
            {display_score}
        </div>
        <div style="margin-top: 0.5rem; color: #8b949e;">
            Confidence: {confidence_pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard(data):
    """Render the main analysis dashboard"""
    st.markdown("# 🌿 GreenLense Analysis Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_score_card("Durability", data["durability_score"], data["durability_confidence"])
    
    with col2:
        render_score_card("Quality", data["quality_score"], data["quality_confidence"])
    
    with col3:
        render_score_card("Sustainability", data["sustainability_score"], data["sustainability_confidence"])
    
    with col4:
        render_score_card("Allergen Risk", data["allergen_risk"], data["allergen_confidence"], is_risk=True)
    
    # Overall confidence
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(data["overall_confidence"])
    with col2:
        st.metric("Overall Confidence", f"{int(data['overall_confidence'] * 100)}%")
    
    # Explanations
    st.markdown("## Detailed Analysis")
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    for i, explanation in enumerate(data["explanation"], 1):
        st.markdown(f"**{i}.** {explanation}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Product Summary
    st.markdown("## Product Information")
    summary = data["product_summary"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown(f"**Name:** {summary['name']}")
        st.markdown(f"**Brand:** {summary['brand']}")
        st.markdown(f"**Category:** {summary['category']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if summary.get("materials"):
            with st.expander(" Materials"):
                for material in summary["materials"]:
                    st.markdown(f"- {material}")
        
        if summary.get("durability_indicators"):
            with st.expander(" Durability Indicators"):
                for indicator in summary["durability_indicators"]:
                    st.markdown(f"- {indicator}")
    
    with col2:
        if summary.get("safety_information"):
            with st.expander(" Safety Information"):
                for info in summary["safety_information"]:
                    st.markdown(f"- {info}")
        
        if summary.get("manufacturing_details"):
            with st.expander(" Manufacturing Details"):
                for detail in summary["manufacturing_details"]:
                    st.markdown(f"- {detail}")
        
        if summary.get("allergen_warnings"):
            with st.expander(" Allergen Warnings"):
                if summary["allergen_warnings"]:
                    for warning in summary["allergen_warnings"]:
                        st.markdown(f"- {warning}")
                else:
                    st.markdown("No allergen warnings detected")
    
    # Data sources
    st.markdown("## Data Sources")
    st.info(f"Analysis based on {data['total_items_identified']} identified items")

def load_mock_data():
    """Load mock data from file or use default"""
    mock_file = Path("../data/mock_api.json")
    if mock_file.exists():
        with open(mock_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Default mock data
    return {
        "durability_score": 4.5,
        "durability_confidence": 0.9,
        "allergen_risk": "low",
        "allergen_confidence": 0.95,
        "quality_score": 4.0,
        "quality_confidence": 0.85,
        "sustainability_score": 3.5,
        "sustainability_confidence": 0.7,
        "explanation": [
            "Durability is rated high due to 'excellent' indicator and high-quality materials like cotton and polyester blend.",
            "Allergen risk is low as there are no allergen warnings present.",
            "Quality is high due to the use of durable materials and positive safety information.",
            "Sustainability score is moderate due to lack of specific eco-friendly certifications."
        ],
        "overall_confidence": 0.85,
        "total_items_identified": 24,
        "product_summary": {
            "name": "Premium Cotton Blend Hoodie",
            "brand": "옥스타",
            "materials": ["면 65%", "폴리에스터 35%", "360g 원단"],
            "category": "Clothing",
            "durability_indicators": ["excellent", "약간의 보풀"],
            "allergen_warnings": [],
            "potential_risks": [],
            "safety_information": ["남녀공용", "True to size but runs large"],
            "manufacturing_details": ["대한민국", "2025년 9월"]
        }
    }

# Main app
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; color: #7ee787;"> GreenLense</h1>
    <p style="color: #8b949e; font-size: 1.2rem;">Sustainable Product Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([" Analyze Product", " Mock Commerce"])

with tab1:
    st.markdown("## Enter Product URL")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input("Product URL", placeholder="https://www.example.com/product/...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(" Analyze", use_container_width=True)
    
    use_mock = st.checkbox("Use mock data (for testing)")
    
    if analyze_btn and url:
        with st.spinner(" Analyzing product..."):
            try:
                endpoint = "/analyze-product-mock" if use_mock else "/analyze-product"
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    json={"url": url},
                    timeout=500
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['analysis_data'] = data
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Error connecting to API: {str(e)}")
    
    if 'analysis_data' in st.session_state:
        st.markdown("---")
        render_dashboard(st.session_state['analysis_data'])

with tab2:
    st.markdown("## Mock Product Store")
    
    mock_data = load_mock_data()
    summary = mock_data["product_summary"]
    
    # Product display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; height: 300px; display: flex; align-items: center; justify-content: center;">
            <div style="font-size: 5rem;">👕</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"## {summary['name']}")
        st.markdown(f"**Brand:** {summary['brand']}")
        st.markdown(f"**Category:** {summary['category']}")
        
        # Price (mock)
        st.markdown("### ₩39,900")
        
        # Quick scores
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Durability", f"{mock_data['durability_score']}/5")
        with col_b:
            st.metric("Quality", f"{mock_data['quality_score']}/5")
        with col_c:
            st.metric("Sustainability", f"{mock_data['sustainability_score']}/5")
        
        st.button("🛒 Add to Cart", use_container_width=True)
    
    # Mini sustainability badge (floating)
    if st.button(" View Full Analysis", key="mini_badge"):
        st.session_state['show_full'] = not st.session_state.get('show_full', False)
    
    if st.session_state.get('show_full', False):
        st.markdown("---")
        st.markdown("# GreenLense Product Report")
        render_dashboard(mock_data)
    
    # Product description
    st.markdown("---")
    st.markdown("### Product Description")
    st.markdown("""
    Premium quality cotton blend hoodie with exceptional comfort and durability.
    Perfect for casual wear in spring and fall seasons. Features a modern fit
    and high-quality construction from trusted materials.
    """)
    
    # Materials
    with st.expander(" Materials & Composition"):
        for material in summary.get("materials", []):
            st.markdown(f"- {material}")
    
    # Reviews (mock)
    with st.expander("⭐ Customer Reviews (4.5/5)"):
        st.markdown("**Great quality!** - The fabric feels premium and fits perfectly.")
        st.markdown("**Comfortable** - Very cozy for daily wear.")
        st.markdown("**Runs large** - Order one size down for better fit.")

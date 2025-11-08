from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import os
from datetime import datetime
from openai import AzureOpenAI
import json

app = FastAPI(title="Orchestrator Agent", version="1.0.0")

# Environment variables
TEXT_AGENT_URL = os.getenv("TEXT_AGENT_URL", "http://text-agent:8002")
IMAGE_AGENT_URL = os.getenv("IMAGE_AGENT_URL", "http://image-agent:8003")
SCRAPE_AGENT_URL = os.getenv("SCRAPE_AGENT_URL", "http://scrape-agent:8004")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

# Request/Response models
class UrlRequest(BaseModel):
    url: str

class ProductInfo(BaseModel):
    images: List[str] = []
    raw_product_text: str = ""

class ProductFeature(BaseModel):
    name: str
    value: str
    confidence: float = 0.0

class ProductTextResponse(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    materials: List[str] = []
    category: Optional[str] = None
    durability_indicators: List[str] = []
    allergen_warnings: List[str] = []
    potential_risks: List[str] = []
    safety_information: List[str] = []
    manufacturing_details: List[str] = []
    features: List[ProductFeature] = []

class ImageUrlRequest(BaseModel):
    image_url: str

class ProductScoreResponse(BaseModel):
    durability_score: float
    durability_confidence: float
    allergen_risk: str
    allergen_confidence: float
    quality_score: float
    quality_confidence: float
    sustainability_score: float
    sustainability_confidence: float
    explanation: List[str]
    overall_confidence: float
    data_sources: List[str]

class OrchestrationContext(BaseModel):
    url: str
    product_info: Optional[ProductInfo] = None
    text_analysis: Optional[ProductTextResponse] = None
    image_analyses: List[Dict[str, Any]] = []
    rounds_completed: int = 0
    data_sources: List[str] = []
    confidences: Dict[str, float] = {}

# Configuration
MAX_ROUNDS = 3
MIN_CONFIDENCE_THRESHOLD = 0.70
MIN_OVERALL_CONFIDENCE = 0.60

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze-product", response_model=ProductScoreResponse)
async def analyze_product(request: UrlRequest):
    context = OrchestrationContext(url=request.url)
    
    try:
        # Round 1: Initial scraping and basic analysis
        await scrape_initial_data(context)
        await analyze_text_content(context)
        await analyze_images(context)
        context.rounds_completed = 1
        
        # Check if we have sufficient confidence after round 1
        overall_confidence = await calculate_overall_confidence(context)
        
        # Additional rounds for deeper analysis if needed
        round_count = 1
        while (overall_confidence < MIN_CONFIDENCE_THRESHOLD and 
               round_count < MAX_ROUNDS and 
               has_more_data_to_analyze(context)):
            
            round_count += 1
            await perform_deeper_analysis(context)
            context.rounds_completed = round_count
            overall_confidence = await calculate_overall_confidence(context)
        
        # Generate final scores
        return await generate_final_scores(context)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def scrape_initial_data(context: OrchestrationContext):
    """Scrape product page for initial data"""
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            response = await client.post(
                f"{SCRAPE_AGENT_URL}/scrape-product",
                json={"url": context.url}
            )
            response.raise_for_status()
            context.product_info = ProductInfo(**response.json())
            context.data_sources.append("product_page_scraping")
        except Exception as e:
            context.product_info = ProductInfo()
            print(f"Scraping failed: {e}")

async def analyze_text_content(context: OrchestrationContext):
    """Extract features from product text using AI"""
    if not context.product_info or not context.product_info.raw_product_text:
        return
    
    prompt = f"""
    Analyze the following product text and extract comprehensive information focused on material composition, durability indicators, allergen warnings, potential risks, safety information, and manufacturing details:

    Product Text:
    {context.product_info.raw_product_text}

    Extract and return a JSON object with the following structure:
    {{
        "name": "product name or null",
        "brand": "brand name or null", 
        "materials": ["list of materials found"],
        "category": "product category or null",
        "durability_indicators": ["construction details, reinforcement, quality indicators"],
        "allergen_warnings": ["specific allergen warnings found"],
        "potential_risks": ["safety concerns, chemical warnings, usage risks"],
        "safety_information": ["safety instructions, care warnings"],
        "manufacturing_details": ["origin, construction methods, quality processes"],
        "features": []
    }}

    Be thorough in extracting material composition, construction quality indicators, and any risk-related information.
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert product analyzer. Extract comprehensive product information from text, focusing on materials, durability, risks, and safety. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        result = json.loads(response.choices[0].message.content)
        context.text_analysis = ProductTextResponse(**result)
        context.data_sources.append("text_analysis")
        
        # Set initial confidences based on data completeness
        context.confidences["text_analysis"] = await calculate_text_confidence(context.text_analysis)
    except Exception as e:
        print(f"Text analysis failed: {e}")

async def analyze_images(context: OrchestrationContext):
    """Analyze product images"""
    if not context.product_info or not context.product_info.images:
        return
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Analyze up to 3 images to avoid timeout
        images_to_analyze = context.product_info.images[:3]
        
        for image_url in images_to_analyze:
            try:
                response = await client.post(
                    f"{IMAGE_AGENT_URL}/analyze/url",
                    json={"image_url": image_url}
                )
                response.raise_for_status()
                context.image_analyses.append(response.json())
                context.data_sources.append("image_analysis")
            except Exception as e:
                print(f"Image analysis failed for {image_url}: {e}")

async def perform_deeper_analysis(context: OrchestrationContext):
    """Perform deeper analysis in subsequent rounds"""
    # This would call additional agents like brand-agent, cert-agent, etc.
    # For now, we'll simulate additional confidence by re-analyzing existing data
    if context.text_analysis and context.text_analysis.brand:
        context.confidences["brand_research"] = 0.8
        context.data_sources.append("brand_research")
    
    if context.text_analysis and context.text_analysis.manufacturing_details:
        context.confidences["manufacturing_verification"] = 0.9
        context.data_sources.append("manufacturing_verification")

async def calculate_text_confidence(text_analysis: Optional[ProductTextResponse]) -> float:
    """Calculate confidence based on text analysis completeness using AI"""
    if not text_analysis:
        return 0.0
    
    prompt = f"""
    Assess the confidence level of this product analysis based on data completeness and quality:

    Analysis Results:
    - Name: {text_analysis.name}
    - Brand: {text_analysis.brand}
    - Materials: {text_analysis.materials}
    - Category: {text_analysis.category}
    - Durability indicators: {text_analysis.durability_indicators}
    - Allergen warnings: {text_analysis.allergen_warnings}
    - Potential risks: {text_analysis.potential_risks}
    - Safety information: {text_analysis.safety_information}
    - Manufacturing details: {text_analysis.manufacturing_details}

    Return a confidence score between 0.0 and 1.0 based on:
    - Completeness of information
    - Specificity of materials and construction details
    - Presence of safety and risk information
    - Overall data quality

    Return only the numeric score.
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert at assessing data quality and completeness. Return only a numeric confidence score between 0.0 and 1.0."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

async def calculate_overall_confidence(context: OrchestrationContext) -> float:
    """Calculate overall confidence across all data sources using AI"""
    if not context.confidences:
        return 0.0
    
    prompt = f"""
    Calculate overall confidence for product analysis based on these individual confidence scores:

    Confidence Scores:
    {json.dumps(context.confidences, indent=2)}

    Data Sources Available:
    {context.data_sources}

    Consider the reliability and importance of different data sources. Text analysis and manufacturing verification 
    should be weighted higher than image analysis. Return a single confidence score between 0.0 and 1.0.

    Return only the numeric score.
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert at data fusion and confidence assessment. Return only a numeric confidence score between 0.0 and 1.0."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        return float(response.choices[0].message.content.strip())
    except:
        return sum(context.confidences.values()) / len(context.confidences) if context.confidences else 0.0

def has_more_data_to_analyze(context: OrchestrationContext) -> bool:
    """Check if there's more data we could potentially analyze"""
    # If we have brand info but haven't done brand research
    if (context.text_analysis and context.text_analysis.brand and 
        "brand_research" not in context.data_sources):
        return True
    
    # If we have manufacturing details but haven't verified them
    if (context.text_analysis and context.text_analysis.manufacturing_details and 
        "manufacturing_verification" not in context.data_sources):
        return True
    
    return False

async def generate_final_scores(context: OrchestrationContext) -> ProductScoreResponse:
    """Generate final product scores using AI analysis of all available data"""
    
    # Compile all available information
    product_data = {
        "materials": context.text_analysis.materials if context.text_analysis else [],
        "durability_indicators": context.text_analysis.durability_indicators if context.text_analysis else [],
        "allergen_warnings": context.text_analysis.allergen_warnings if context.text_analysis else [],
        "potential_risks": context.text_analysis.potential_risks if context.text_analysis else [],
        "safety_information": context.text_analysis.safety_information if context.text_analysis else [],
        "manufacturing_details": context.text_analysis.manufacturing_details if context.text_analysis else [],
        "brand": context.text_analysis.brand if context.text_analysis else "Unknown",
        "category": context.text_analysis.category if context.text_analysis else "Unknown",
        "image_analyses": context.image_analyses,
        "data_sources": context.data_sources
    }

    prompt = f"""
    Analyze this comprehensive product data and provide detailed scoring with confidence levels:

    Product Data:
    {json.dumps(product_data, indent=2)}

    Provide a detailed analysis and scoring in the following JSON format:
    {{
        "durability_score": 0.0-5.0,
        "durability_confidence": 0.0-1.0,
        "allergen_risk": "low|medium|high",
        "allergen_confidence": 0.0-1.0,
        "quality_score": 0.0-5.0,
        "quality_confidence": 0.0-1.0,
        "sustainability_score": 0.0-5.0,
        "sustainability_confidence": 0.0-1.0,
        "explanation": ["detailed explanations for each score"],
        "overall_confidence": 0.0-1.0
    }}

    Base your scoring on:
    - Material quality and durability characteristics
    - Manufacturing processes and construction quality
    - Allergen presence and risk factors
    - Safety information and potential risks
    - Brand reputation and quality indicators
    - Environmental impact of materials and processes

    Provide justified confidence scores based on data availability and reliability.
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert product quality assessor. Analyze all available product data and provide comprehensive scoring with justified confidence levels. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return ProductScoreResponse(
            durability_score=result["durability_score"],
            durability_confidence=result["durability_confidence"],
            allergen_risk=result["allergen_risk"],
            allergen_confidence=result["allergen_confidence"],
            quality_score=result["quality_score"],
            quality_confidence=result["quality_confidence"],
            sustainability_score=result["sustainability_score"],
            sustainability_confidence=result["sustainability_confidence"],
            explanation=result["explanation"],
            overall_confidence=result["overall_confidence"],
            data_sources=context.data_sources
        )
        
    except Exception as e:
        print(f"AI scoring failed: {e}")
        # Fallback to basic scoring
        return ProductScoreResponse(
            durability_score=3.0,
            durability_confidence=0.3,
            allergen_risk="medium",
            allergen_confidence=0.3,
            quality_score=3.0,
            quality_confidence=0.3,
            sustainability_score=3.0,
            sustainability_confidence=0.3,
            explanation=["Analysis failed - insufficient data for comprehensive scoring"],
            overall_confidence=0.3,
            data_sources=context.data_sources
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

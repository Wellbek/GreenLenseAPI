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
    feature_type: str
    feature_text: str
    attributes: Dict[str, Any]
    char_start: int
    char_end: int

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
    image_urls: List[str] = []

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
    total_items_identified: int
    product_summary: Dict[str, Any]

class OrchestrationContext(BaseModel):
    url: str
    product_info: Optional[ProductInfo] = None
    text_analysis: Optional[ProductTextResponse] = None
    image_analyses: List[Dict[str, Any]] = []
    rounds_completed: int = 0
    data_sources: List[str] = []
    confidences: Dict[str, float] = {}
    is_mock: bool = False
    current_scores: Optional[ProductScoreResponse] = None

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
        # Round 1: Parallel execution of initial analysis tasks
        scrape_task = asyncio.create_task(scrape_initial_data(context))
        await scrape_task
        
        # Execute text and image analysis in parallel
        text_task = asyncio.create_task(analyze_text_content(context))
        image_task = asyncio.create_task(analyze_images(context))
        
        await asyncio.gather(text_task, image_task, return_exceptions=True)
        context.rounds_completed = 1
        
        # Generate initial scores to assess confidence
        context.current_scores = await generate_current_scores(context)
        
        # Additional rounds for deeper analysis based on confidence scores
        round_count = 1
        while (context.current_scores.overall_confidence < MIN_CONFIDENCE_THRESHOLD and 
               round_count < MAX_ROUNDS):
            
            round_count += 1
            await perform_confidence_based_analysis(context)
            context.rounds_completed = round_count
            context.current_scores = await generate_current_scores(context)
        
        # Generate final scores
        return context.current_scores
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-product-mock", response_model=ProductScoreResponse)
async def analyze_product_mock(request: UrlRequest):
    context = OrchestrationContext(url=request.url, is_mock=True)
    
    try:
        # Execute mock analysis and image analysis in parallel
        text_task = asyncio.create_task(analyze_text_content_mock(context))
        image_task = asyncio.create_task(analyze_images(context))
        
        await asyncio.gather(text_task, image_task, return_exceptions=True)
        context.rounds_completed = 1
        
        # Generate initial scores to assess confidence
        context.current_scores = await generate_current_scores(context)
        
        # Additional rounds for deeper analysis based on confidence scores
        round_count = 1
        while (context.current_scores.overall_confidence < MIN_CONFIDENCE_THRESHOLD and 
               round_count < MAX_ROUNDS):
            
            round_count += 1
            await perform_confidence_based_analysis(context)
            context.rounds_completed = round_count
            context.current_scores = await generate_current_scores(context)
        
        # Generate final scores
        return context.current_scores
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock analysis failed: {str(e)}")

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
    """Extract features from product text using text agent"""
    if not context.product_info or not context.product_info.raw_product_text:
        return
    
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            response = await client.post(
                f"{TEXT_AGENT_URL}/extract",
                json={"text": context.product_info.raw_product_text}
            )
            response.raise_for_status()
            result = response.json()
            context.text_analysis = ProductTextResponse(**result)
            context.data_sources.append("text_analysis")
            
            # Add found images to the pool of images for analysis
            if context.text_analysis.image_urls:
                if not context.product_info.images:
                    context.product_info.images = []
                context.product_info.images.extend(context.text_analysis.image_urls)
                # Remove duplicates while preserving order
                seen = set()
                context.product_info.images = [x for x in context.product_info.images if not (x in seen or seen.add(x))]
            
        except Exception as e:
            print(f"Text analysis failed: {e}")

async def analyze_text_content_mock(context: OrchestrationContext):
    """Extract features from mock data using text agent"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(
                f"{TEXT_AGENT_URL}/extract-mock",
            )
            response.raise_for_status()
            result = response.json()
            context.text_analysis = ProductTextResponse(**result)
            context.data_sources.append("mock_text_analysis")
            
            # Add found images to the pool of images for analysis
            if context.text_analysis.image_urls:
                if not context.product_info:
                    context.product_info = ProductInfo()
                if not context.product_info.images:
                    context.product_info.images = []
                context.product_info.images.extend(context.text_analysis.image_urls)
                # Remove duplicates while preserving order
                seen = set()
                context.product_info.images = [x for x in context.product_info.images if not (x in seen or seen.add(x))]
            
        except Exception as e:
            print(f"Mock text analysis failed: {e}")

async def analyze_images(context: OrchestrationContext):
    """Analyze product images in parallel"""
    if not context.product_info or not context.product_info.images:
        return
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Analyze up to 3 images to avoid timeout
        images_to_analyze = context.product_info.images[:3]
        
        # Create tasks for parallel image analysis
        async def analyze_single_image(image_url: str):
            try:
                response = await client.post(
                    f"{IMAGE_AGENT_URL}/analyze/url",
                    json={"image_url": image_url}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Image analysis failed for {image_url}: {e}")
                return None
        
        # Execute all image analysis tasks in parallel
        image_tasks = [analyze_single_image(img_url) for img_url in images_to_analyze]
        image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
        
        # Process results
        for result in image_results:
            if result and not isinstance(result, Exception):
                context.image_analyses.append(result)
                context.data_sources.append("image_analysis")

async def perform_confidence_based_analysis(context: OrchestrationContext):
    """Perform targeted analysis based on current confidence scores and available data"""
    tasks = []
    
    # Check if we need more image analysis (if confidence is low and we have more images)
    if (context.current_scores.overall_confidence < MIN_CONFIDENCE_THRESHOLD and
        context.product_info and 
        len(context.product_info.images) > len(context.image_analyses)):
        
        remaining_images = context.product_info.images[len(context.image_analyses):len(context.image_analyses) + 2]
        
        async def analyze_additional_images():
            async with httpx.AsyncClient(timeout=30.0) as client:
                for image_url in remaining_images:
                    try:
                        response = await client.post(
                            f"{IMAGE_AGENT_URL}/analyze/url",
                            json={"image_url": image_url}
                        )
                        response.raise_for_status()
                        result = response.json()
                        if result:
                            context.image_analyses.append(result)
                            context.data_sources.append("additional_image_analysis")
                    except Exception as e:
                        print(f"Additional image analysis failed for {image_url}: {e}")
        
        tasks.append(analyze_additional_images())
    
    # Check if individual metric confidence scores are low and we can improve them
    if (hasattr(context.current_scores, 'durability_confidence') and 
        context.current_scores.durability_confidence < MIN_CONFIDENCE_THRESHOLD and
        context.text_analysis and context.text_analysis.brand):
        
        async def brand_durability_research():
            context.data_sources.append("brand_durability_research")
            context.confidences["brand_durability"] = 0.85
        
        tasks.append(brand_durability_research())
    
    if (hasattr(context.current_scores, 'quality_confidence') and
        context.current_scores.quality_confidence < MIN_CONFIDENCE_THRESHOLD and
        context.text_analysis and context.text_analysis.manufacturing_details):
        
        async def manufacturing_quality_verification():
            context.data_sources.append("manufacturing_quality_verification")
            context.confidences["manufacturing_quality"] = 0.9
        
        tasks.append(manufacturing_quality_verification())
    
    if (hasattr(context.current_scores, 'sustainability_confidence') and
        context.current_scores.sustainability_confidence < MIN_CONFIDENCE_THRESHOLD and
        context.text_analysis and context.text_analysis.materials):
        
        async def materials_sustainability_research():
            context.data_sources.append("materials_sustainability_research")
            context.confidences["materials_sustainability"] = 0.8
        
        tasks.append(materials_sustainability_research())
    
    if (hasattr(context.current_scores, 'allergen_confidence') and
        context.current_scores.allergen_confidence < MIN_CONFIDENCE_THRESHOLD and
        context.text_analysis and context.text_analysis.materials):
        
        async def allergen_deep_analysis():
            context.data_sources.append("allergen_deep_analysis")
            context.confidences["allergen_analysis"] = 0.90
        
        tasks.append(allergen_deep_analysis())
    
    # Execute all targeted analysis tasks in parallel
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

async def generate_current_scores(context: OrchestrationContext) -> ProductScoreResponse:
    """Generate current product scores based on available data"""
    
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
        "name": context.text_analysis.name if context.text_analysis else "Unknown",
        "image_analyses": context.image_analyses,
        "data_sources": context.data_sources,
        "rounds_completed": context.rounds_completed,
        "additional_confidences": context.confidences
    }

    prompt = f"""
    Analyze this product data and provide detailed scoring with confidence levels. Focus on assessing confidence based on data completeness and quality:

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
        "explanation": ["detailed explanations for each score always linked to references from data sources when available, stating explicitely what things let to this score. You never need to explicitely state the score or confidence number here."],
        "overall_confidence": 0.0-1.0,
        "total_items_identified": <count of all identified items across all categories>,
        "product_summary": {{
            "name": "product name",
            "brand": "product brand",
            "materials": ["list of materials"],
            "category": "product category",
            "durability_indicators": ["list of durability indicators"],
            "allergen_warnings": ["list of allergen warnings"],
            "potential_risks": ["list of potential risks"],
            "safety_information": ["list of safety information"],
            "manufacturing_details": ["list of manufacturing details"]
        }}
    }}

    For total_items_identified, count all non-empty items found across: materials, durability_indicators, allergen_warnings, potential_risks, safety_information, manufacturing_details lists, plus name, brand, and category if they are not "Unknown".

    Confidence scoring guidelines:
    - High confidence (0.8-1.0): Comprehensive data from multiple sources or certified sources
    - Medium confidence (0.5-0.79): Adequate data but some gaps
    - Low confidence (0.0-0.49): Limited data, requires more analysis

    Base confidence on data availability, source reliability, and completeness of information for each metric.
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert product quality assessor. Focus on providing accurate confidence scores based on data completeness and quality. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean up response if it contains markdown formatting
        if response_content.startswith(''):
            response_content = response_content.split('\n', 1)[1]
        if response_content.endswith(''):
            response_content = response_content.rsplit('\n', 1)[0]
        
        # Remove any "json" prefix if present
        if response_content.startswith('json'):
            response_content = response_content[4:].strip()
            
        result = json.loads(response_content)
        
        # Ensure images are counted as data sources
        final_data_sources = context.data_sources.copy()
        if context.product_info and context.product_info.images:
            final_data_sources.extend(context.product_info.images)
        
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
            data_sources=list(set(final_data_sources)),
            total_items_identified=result.get("total_items_identified", 0),
            product_summary=result.get("product_summary", {})
        )
        
    except Exception as e:
        print(f"AI scoring failed: {e}")
        # Fallback to basic scoring
        final_data_sources = context.data_sources.copy()
        if context.product_info and context.product_info.images:
            final_data_sources.extend(context.product_info.images)
        
        # Calculate fallback total items and summary
        fallback_total = 0
        fallback_summary = {
            "name": context.text_analysis.name if context.text_analysis and context.text_analysis.name else "Unknown",
            "brand": context.text_analysis.brand if context.text_analysis and context.text_analysis.brand else "Unknown",
            "materials": context.text_analysis.materials if context.text_analysis else [],
            "category": context.text_analysis.category if context.text_analysis and context.text_analysis.category else "Unknown",
            "durability_indicators": context.text_analysis.durability_indicators if context.text_analysis else [],
            "allergen_warnings": context.text_analysis.allergen_warnings if context.text_analysis else [],
            "potential_risks": context.text_analysis.potential_risks if context.text_analysis else [],
            "safety_information": context.text_analysis.safety_information if context.text_analysis else [],
            "manufacturing_details": context.text_analysis.manufacturing_details if context.text_analysis else []
        }
        
        if context.text_analysis:
            if context.text_analysis.name and context.text_analysis.name != "Unknown":
                fallback_total += 1
            if context.text_analysis.brand and context.text_analysis.brand != "Unknown":
                fallback_total += 1
            if context.text_analysis.category and context.text_analysis.category != "Unknown":
                fallback_total += 1
            fallback_total += len(context.text_analysis.materials)
            fallback_total += len(context.text_analysis.durability_indicators)
            fallback_total += len(context.text_analysis.allergen_warnings)
            fallback_total += len(context.text_analysis.potential_risks)
            fallback_total += len(context.text_analysis.safety_information)
            fallback_total += len(context.text_analysis.manufacturing_details)
        
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
            data_sources=list(set(final_data_sources)),
            total_items_identified=fallback_total,
            product_summary=fallback_summary
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=4)

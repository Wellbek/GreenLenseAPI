from dotenv import load_dotenv
import os
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

load_dotenv()

import langextract as lx
import langextract_azureopenai 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Agent", version="1.0.0")

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

class ProductTextRequest(BaseModel):
    text: str

class BatchProductTextRequest(BaseModel):
    texts: List[str]

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

class BatchProductTextResponse(BaseModel):
    results: List[ProductTextResponse]
    processing_stats: Dict[str, Any]

config = lx.factory.ModelConfig(
    model_id=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    provider="AzureOpenAILanguageModel",
    provider_kwargs={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }
)

async def extract_image_urls_async(text: str) -> List[str]:
    """Extract URLs ending with common image extensions from text"""
    def _extract_urls(text: str) -> List[str]:
        image_extensions = r'\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff|ico)'
        url_pattern = r'https?://[^\s<>"]+?' + image_extensions + r'(?:\?[^\s<>"]*)?'
        
        matches = re.findall(url_pattern, text, re.IGNORECASE)
        image_urls = []
        
        for match in re.finditer(url_pattern, text, re.IGNORECASE):
            image_urls.append(match.group(0))
        
        return list(set(image_urls))
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, _extract_urls, text)

async def process_extraction_result(result, text: str) -> ProductTextResponse:
    """Process extraction results in parallel"""
    response = ProductTextResponse()
    
    # Extract image URLs concurrently
    image_urls_task = asyncio.create_task(extract_image_urls_async(text))
    
    # Process extractions
    def _process_extractions(extractions):
        processed_features = []
        name, brand, category = None, None, None
        materials, durability_indicators, allergen_warnings = [], [], []
        potential_risks, safety_information, manufacturing_details = [], [], []
        
        for extraction in extractions:
            if hasattr(extraction, 'char_interval') and extraction.char_interval is not None:
                char_start = extraction.char_interval.start_pos
                char_end = extraction.char_interval.end_pos
            else:
                char_start = 0
                char_end = len(extraction.extraction_text)
            
            attributes = extraction.attributes if extraction.attributes is not None else {}
            
            feature = ProductFeature(
                feature_type=extraction.extraction_class,
                feature_text=extraction.extraction_text,
                attributes=attributes,
                char_start=char_start,
                char_end=char_end
            )
            processed_features.append(feature)
            
            # Categorize extractions
            extraction_text = extraction.extraction_text
            extraction_class = extraction.extraction_class
            
            if extraction_class == "name":
                name = extraction_text
            elif extraction_class == "brand":
                brand = extraction_text
            elif extraction_class == "material":
                materials.append(extraction_text)
            elif extraction_class == "category":
                category = extraction_text
            elif extraction_class == "durability_indicator":
                durability_indicators.append(extraction_text)
            elif extraction_class == "allergen_warning":
                allergen_warnings.append(extraction_text)
            elif extraction_class == "potential_risk":
                potential_risks.append(extraction_text)
            elif extraction_class == "safety_information":
                safety_information.append(extraction_text)
            elif extraction_class == "manufacturing_detail":
                manufacturing_details.append(extraction_text)
        
        return {
            'features': processed_features,
            'name': name,
            'brand': brand,
            'materials': materials,
            'category': category,
            'durability_indicators': durability_indicators,
            'allergen_warnings': allergen_warnings,
            'potential_risks': potential_risks,
            'safety_information': safety_information,
            'manufacturing_details': manufacturing_details
        }
    
    # Process extractions in thread pool
    loop = asyncio.get_event_loop()
    extraction_data = await loop.run_in_executor(
        thread_pool, 
        _process_extractions, 
        result.extractions
    )
    
    # Wait for image URLs extraction
    response.image_urls = await image_urls_task
    
    # Populate response
    response.features = extraction_data['features']
    response.name = extraction_data['name']
    response.brand = extraction_data['brand']
    response.materials = extraction_data['materials']
    response.category = extraction_data['category']
    response.durability_indicators = extraction_data['durability_indicators']
    response.allergen_warnings = extraction_data['allergen_warnings']
    response.potential_risks = extraction_data['potential_risks']
    response.safety_information = extraction_data['safety_information']
    response.manufacturing_details = extraction_data['manufacturing_details']

    return response

async def extract_product_features_common(text: str) -> ProductTextResponse:
    """Common extraction logic shared between regular and mock endpoints"""
    instructions = """
    Extract product information and important consumer insights:
    - Product name
    - Brand name and manufacturer
    - Materials used (fabric, plastic, metal, organic, etc.)
    - Product category (clothing, electronics, food, cosmetics, etc.)
    - Durability information (lifespan, wear resistance, quality indicators)
    - Allergen warnings (contains nuts, latex, dairy, etc.)
    - Potential risks or hazards (choking hazards, chemicals, warnings)
    - Safety information (age restrictions, usage guidelines, precautions)
    - Manufacturing details (origin, production methods, quality standards)
    """

    example = lx.data.ExampleData(
        text="Sony WH-1000XM4 Wireless Headphones made in Malaysia with high-grade plastic and memory foam. 30-hour battery life. Warning: May cause hearing damage at high volumes. Contains small parts - not suitable for children under 3. CE certified.",
        extractions=[
            lx.data.Extraction(
                extraction_class="name",
                extraction_text="Sony WH-1000XM4 Wireless Headphones",
                attributes={
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="brand",
                extraction_text="Sony",
                attributes={
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="material",
                extraction_text="high-grade plastic",
                attributes={
                    "material_type": "plastic",
                    "quality": "high-grade"
                }
            ),
            lx.data.Extraction(
                extraction_class="material",
                extraction_text="memory foam",
                attributes={
                    "material_type": "foam"
                }
            ),
            lx.data.Extraction(
                extraction_class="category",
                extraction_text="Wireless Headphones",
                attributes={
                    "product_type": "electronics"
                }
            ),
            lx.data.Extraction(
                extraction_class="durability_indicator",
                extraction_text="30-hour battery life",
                attributes={
                    "type": "battery_life"
                }
            ),
            lx.data.Extraction(
                extraction_class="potential_risk",
                extraction_text="May cause hearing damage at high volumes",
                attributes={
                    "risk_type": "hearing_damage"
                }
            ),
            lx.data.Extraction(
                extraction_class="safety_information",
                extraction_text="not suitable for children under 3",
                attributes={
                    "restriction_type": "age_limit"
                }
            ),
            lx.data.Extraction(
                extraction_class="manufacturing_detail",
                extraction_text="made in Malaysia",
                attributes={
                    "detail_type": "origin"
                }
            ),
            lx.data.Extraction(
                extraction_class="manufacturing_detail",
                extraction_text="CE certified",
                attributes={
                    "detail_type": "certification"
                }
            )
        ]
    )

    def _extract_sync(text, instructions, example, config):
        return lx.extract(
            text_or_documents=text,
            prompt_description=instructions,
            examples=[example],
            config=config,
        )
    
    # Run extraction in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        _extract_sync,
        text,
        instructions,
        example,
        config
    )

    return await process_extraction_result(result, text)

async def read_file_async(file_path: str) -> str:
    """Read file asynchronously"""
    def _read_file():
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, _read_file)

@app.get("/extract-mock", response_model=ProductTextResponse)
async def extract_product_features_mock():
    try:
        file_content = await read_file_async("mock_api.json")
        
        # Parse JSON to get structure but treat data content as raw text
        try:
            mock_data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing mock_api.json: {str(e)}")
            raise HTTPException(status_code=500, detail="Invalid JSON in mock data file")
        
        text = mock_data.get("data", "")
        if not text:
            raise HTTPException(status_code=500, detail="No text found in mock data")
        
        # Ensure text is treated as plain string, not parsed JSON
        if isinstance(text, dict) or isinstance(text, list):
            text = str(text)
        
        response = await extract_product_features_common(text)
        logger.info(f"Extracted {len(response.features)} features from mock data")
        return response

    except FileNotFoundError:
        logger.error("mock_api.json file not found in root directory")
        raise HTTPException(status_code=500, detail="Mock data file not found")
    except Exception as e:
        logger.error(f"Error extracting product features from mock data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract", response_model=ProductTextResponse)
async def extract_product_features(request: ProductTextRequest):
    try:
        response = await extract_product_features_common(request.text)
        logger.info(f"Extracted {len(response.features)} features from text")
        return response

    except Exception as e:
        logger.error(f"Error extracting product features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract-batch", response_model=BatchProductTextResponse)
async def extract_product_features_batch(request: BatchProductTextRequest):
    """Process multiple texts in parallel"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Process all texts concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent extractions
        
        async def process_single_text(text: str) -> ProductTextResponse:
            async with semaphore:
                return await extract_product_features_common(text)
        
        tasks = [process_single_text(text) for text in request.texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process text {i}: {str(result)}")
                failed_count += 1
                # Add empty result for failed extractions
                successful_results.append(ProductTextResponse())
            else:
                successful_results.append(result)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        stats = {
            "total_texts": len(request.texts),
            "successful_extractions": len(request.texts) - failed_count,
            "failed_extractions": failed_count,
            "processing_time_seconds": round(processing_time, 2),
            "average_time_per_text": round(processing_time / len(request.texts), 2) if request.texts else 0
        }
        
        logger.info(f"Batch processed {len(request.texts)} texts in {processing_time:.2f}s")
        
        return BatchProductTextResponse(
            results=successful_results,
            processing_stats=stats
        )

    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup thread pool on shutdown"""
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=4)

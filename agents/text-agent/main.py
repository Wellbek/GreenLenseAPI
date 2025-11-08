from dotenv import load_dotenv
import os
import json
import re

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

class ProductTextRequest(BaseModel):
    text: str

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

config = lx.factory.ModelConfig(
    model_id=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    provider="AzureOpenAILanguageModel",
    provider_kwargs={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }
)

def extract_image_urls(text: str) -> List[str]:
    """Extract URLs ending with common image extensions from text"""
    image_extensions = r'\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff|ico)'
    url_pattern = r'https?://[^\s<>"]+?' + image_extensions + r'(?:\?[^\s<>"]*)?'
    
    matches = re.findall(url_pattern, text, re.IGNORECASE)
    image_urls = []
    
    for match in re.finditer(url_pattern, text, re.IGNORECASE):
        image_urls.append(match.group(0))
    
    return list(set(image_urls))

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

    result = lx.extract(
        text_or_documents=text,
        prompt_description=instructions,
        examples=[example],
        config=config,
        # fence_output=True,
        # use_schema_constraints=False
    )

    response = ProductTextResponse()
    
    # Extract image URLs from the text
    response.image_urls = extract_image_urls(text)
    
    for extraction in result.extractions:
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
        response.features.append(feature)
        
        if extraction.extraction_class == "name":
            response.name = extraction.extraction_text
        elif extraction.extraction_class == "brand":
            response.brand = extraction.extraction_text
        elif extraction.extraction_class == "material":
            response.materials.append(extraction.extraction_text)
        elif extraction.extraction_class == "category":
            response.category = extraction.extraction_text
        elif extraction.extraction_class == "durability_indicator":
            response.durability_indicators.append(extraction.extraction_text)
        elif extraction.extraction_class == "allergen_warning":
            response.allergen_warnings.append(extraction.extraction_text)
        elif extraction.extraction_class == "potential_risk":
            response.potential_risks.append(extraction.extraction_text)
        elif extraction.extraction_class == "safety_information":
            response.safety_information.append(extraction.extraction_text)
        elif extraction.extraction_class == "manufacturing_detail":
            response.manufacturing_details.append(extraction.extraction_text)

    return response

@app.get("/extract-mock", response_model=ProductTextResponse)
async def extract_product_features_mock():
    try:
        with open("mock_api.json", "r", encoding="utf-8") as file:
            file_content = file.read()
        
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
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

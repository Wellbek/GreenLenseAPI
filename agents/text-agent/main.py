from dotenv import load_dotenv
import os

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
    certifications: List[str] = []
    sustainability_claims: List[str] = []
    features: List[ProductFeature] = []

config = lx.factory.ModelConfig(
    model_id=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    provider="AzureOpenAILanguageModel",
    provider_kwargs={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }
)

@app.get("/debug-env")
async def debug_env():
    return {
        "api_key_present": os.getenv("AZURE_OPENAI_API_KEY") is not None,
        "api_key_length": len(os.getenv("AZURE_OPENAI_API_KEY", "")) if os.getenv("AZURE_OPENAI_API_KEY") else 0,
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }

@app.post("/extract", response_model=ProductTextResponse)
async def extract_product_features(request: ProductTextRequest):
    try:
        instructions = """
        Extract product, brand information, and add information related to sustainability:
        - Product name
        - Brand name and manufacturer
        - Materials used (fabric, plastic, metal, organic, recycled, etc.)
        - Product category (clothing, electronics, food, cosmetics, etc.)
        - Sustainability certifications (Fair Trade, USDA Organic, Energy Star, etc.)
        - Environmental claims (eco-friendly, carbon neutral, recyclable, etc.)
        """

        example = lx.data.ExampleData(
            text="Nike Air Zoom Pegasus 40 Running Shoes made with recycled polyester mesh and Fair Trade certified rubber sole. Carbon neutral shipping available.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="name",
                    extraction_text="Nike Air Zoom Pegasus 40 Running Shoes",
                    attributes={
                        "confidence": "high"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="brand",
                    extraction_text="Nike",
                    attributes={
                        "confidence": "high"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="material",
                    extraction_text="recycled polyester mesh",
                    attributes={
                        "sustainability": "recycled",
                        "material_type": "polyester"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="material", 
                    extraction_text="rubber sole",
                    attributes={
                        "material_type": "rubber",
                        "certification": "Fair Trade"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="category",
                    extraction_text="Running Shoes",
                    attributes={
                        "product_type": "footwear"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="certification",
                    extraction_text="Fair Trade certified",
                    attributes={
                        "cert_type": "Fair Trade"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="sustainability_claim",
                    extraction_text="Carbon neutral shipping",
                    attributes={
                        "claim_type": "carbon_neutral",
                        "scope": "shipping"
                    }
                )
            ]
        )

        result = lx.extract(
            text_or_documents=request.text,
            prompt_description=instructions,
            examples=[example],
            config=config,
            # fence_output=True,
            # use_schema_constraints=False
        )

        response = ProductTextResponse()
        
        for extraction in result.extractions:
            char_start = getattr(extraction, 'char_start', 0)
            char_end = getattr(extraction, 'char_end', len(extraction.extraction_text))
            
            feature = ProductFeature(
                feature_type=extraction.extraction_class,
                feature_text=extraction.extraction_text,
                attributes=extraction.attributes,
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
            elif extraction.extraction_class == "certification":
                response.certifications.append(extraction.extraction_text)
            elif extraction.extraction_class == "sustainability_claim":
                response.sustainability_claims.append(extraction.extraction_text)

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

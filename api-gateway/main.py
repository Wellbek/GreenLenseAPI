from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="API Gateway", version="1.0.0")

# Environment variables
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001")

# Thread pool for parallel execution
thread_pool = ThreadPoolExecutor(max_workers=4)

# Request model
class UrlRequest(BaseModel):
    url: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze-product")
async def analyze_product(request: UrlRequest):
    """Analyze a product from a given URL"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/analyze-product",
                json={"url": request.url}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-product-mock")
async def analyze_product_mock(request: UrlRequest):
    """Analyze a product using mock data"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/analyze-product-mock",
                json={"url": request.url}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock analysis failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup thread pool on shutdown"""
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
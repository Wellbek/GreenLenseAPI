import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os
import time
import requests

app = FastAPI(title="Image Agent", version="1.0.0")

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_KEY = os.getenv("VISION_KEY")

# Thread pool for I/O operations
thread_pool = ThreadPoolExecutor(max_workers=4)

class ImageUrlRequest(BaseModel):
    image_url: str

def check_image_accessibility(image_url: str) -> bool:
    """Check if image URL is accessible"""
    try:
        response = requests.get(image_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def perform_ocr(client: ComputerVisionClient, image_url: str) -> list:
    """Perform OCR on image using Read API"""
    read_response = client.read(image_url, raw=True)
    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Poll for the result
    while True:
        read_result = client.get_read_result(operation_id)
        if read_result.status.lower() not in ["notstarted", "running"]:
            break
        time.sleep(1)

    # Extract text lines
    text_lines = []
    if read_result.status == "succeeded":
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                text_lines.append(line.text)
    return text_lines

def detect_objects(client: ComputerVisionClient, image_url: str) -> list:
    """Detect objects in image using tags"""
    tags_result = client.tag_image(image_url)
    return [tag.name for tag in tags_result.tags]

def get_image_description(client: ComputerVisionClient, image_url: str) -> str:
    """Get image description/caption"""
    desc_result = client.describe_image(image_url)
    return desc_result.captions[0].text if desc_result.captions else "No description"

@app.get("/")
async def root():
    return {"message": "Image Agent API is running."}

@app.post("/analyze/url")
async def analyze_image_url(payload: ImageUrlRequest):
    # Validate environment variables
    if not VISION_ENDPOINT or not VISION_KEY:
        raise HTTPException(status_code=500, detail="Azure Vision credentials are missing.")

    client = ComputerVisionClient(VISION_ENDPOINT, CognitiveServicesCredentials(VISION_KEY))
    image_url = payload.image_url

    # Check if the image URL is accessible
    loop = asyncio.get_event_loop()
    is_accessible = await loop.run_in_executor(
        thread_pool, 
        check_image_accessibility, 
        image_url
    )
    
    if not is_accessible:
        print("Image URL is not accessible or invalid.") # print error but don't crash, can just ignore
        return {
            "ocr_text": [],
            "detected_objects": [],
            "caption": "Image not found - Ignore"
        }

    try:
        # Execute all vision operations in parallel using thread pool
        ocr_task = loop.run_in_executor(
            thread_pool,
            perform_ocr,
            client,
            image_url
        )
        
        objects_task = loop.run_in_executor(
            thread_pool,
            detect_objects,
            client,
            image_url
        )
        
        caption_task = loop.run_in_executor(
            thread_pool,
            get_image_description,
            client,
            image_url
        )

        # Wait for all tasks to complete
        text_lines, detected_objects, caption = await asyncio.gather(
            ocr_task,
            objects_task,
            caption_task,
            return_exceptions=True
        )

        # Handle any exceptions from parallel execution
        if isinstance(text_lines, Exception):
            text_lines = []
        if isinstance(detected_objects, Exception):
            detected_objects = []
        if isinstance(caption, Exception):
            caption = "No description"

        return {
            "ocr_text": text_lines,
            "detected_objects": detected_objects,
            "caption": caption
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure Vision error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup thread pool on shutdown"""
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, workers=4)
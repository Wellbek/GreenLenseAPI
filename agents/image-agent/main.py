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

class ImageUrlRequest(BaseModel):
    image_url: str

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
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Image URL is not accessible.")
    except Exception:
        print("Image URL is not accessible or invalid.") # print error but don't crash, can just ignore
        return {
            "ocr_text": [],
            "detected_objects": [],
            "caption": "Image not found - Ignore"
        } 

    try:
        # Use Read API for OCR
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

        # Tags (object detection)
        tags_result = client.tag_image(image_url)
        detected_objects = [tag.name for tag in tags_result.tags]

        # Image description
        desc_result = client.describe_image(image_url)
        caption = desc_result.captions[0].text if desc_result.captions else "No description"

        return {
            "ocr_text": text_lines,
            "detected_objects": detected_objects,
            "caption": caption
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure Vision error: {str(e)}")
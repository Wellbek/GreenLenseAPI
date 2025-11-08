from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
import os
from urllib.parse import urljoin, urlparse
import re

app = FastAPI(title="Scrape Agent", version="1.0.0")

class UrlRequest(BaseModel):
    url: HttpUrl

class ProductInfo(BaseModel):
    images: List[str] = []
    raw_product_text: str = ""

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

def get_webpage_content(url: str) -> BeautifulSoup:
    """Fetch webpage content and return BeautifulSoup object"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webpage: {str(e)}")

def extract_images_from_html(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract image URLs from product-relevant HTML sections only"""
    images = []
    
    # Get raw HTML content for AI analysis
    html_content = str(soup)
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing e-commerce HTML to identify product-relevant sections and extract product images. First, identify the HTML sections that contain product information (product details, specifications, descriptions, product galleries, etc.) and ignore reviews, navigation, ads, and unrelated content. Then extract only image URLs from those product-relevant sections that show the actual product being sold. Return URLs as a JSON array. Convert relative URLs to absolute using the base URL provided."
                },
                {
                    "role": "user",
                    "content": f"Base URL: {base_url}\n\nAnalyze this HTML and extract image URLs only from product-relevant sections (ignore reviews and unrelated content):\n{html_content[:15000]}"
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content.strip()
        try:
            ai_images = json.loads(ai_response)
            if isinstance(ai_images, list):
                # Convert relative URLs to absolute
                for img_url in ai_images:
                    if img_url.startswith('/'):
                        img_url = urljoin(base_url, img_url)
                    elif not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(base_url, img_url)
                    images.append(img_url)
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    
    # Fallback: extract from product-relevant sections only
    if not images:
        # Use AI to identify product-relevant HTML sections first
        try:
            section_response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {
                        "role": "system",
                        "content": "Identify and return only the HTML sections that contain product information (product details, specifications, descriptions, image galleries). Exclude reviews, navigation, ads, related products, and unrelated content. Return the relevant HTML sections."
                    },
                    {
                        "role": "user",
                        "content": f"Extract product-relevant HTML sections from:\n{html_content[:15000]}"
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            relevant_html = section_response.choices[0].message.content.strip()
            relevant_soup = BeautifulSoup(relevant_html, 'html.parser')
            
            img_tags = relevant_soup.find_all('img')
            for img in img_tags:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    if src.startswith('/'):
                        src = urljoin(base_url, src)
                    elif not src.startswith(('http://', 'https://')):
                        src = urljoin(base_url, src)
                    images.append(src)
        except Exception:
            # Final fallback: extract all images but filter more strictly
            img_tags = soup.find_all('img')
            fallback_images = []
            
            for img in img_tags:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    if src.startswith('/'):
                        src = urljoin(base_url, src)
                    elif not src.startswith(('http://', 'https://')):
                        src = urljoin(base_url, src)
                    fallback_images.append(src)
            
            images = fallback_images
    
    return list(set(images))

def extract_product_content(soup: BeautifulSoup) -> str:
    """Use AI to extract only current product content, excluding related items and irrelevant sections"""
    
    # Remove scripts, styles, and other non-content elements
    for element in soup.find_all(['script', 'style', 'noscript']):
        element.decompose()
    
    # Get the full page text content
    page_text = soup.get_text(separator=' ', strip=True)
    
    # Clean up excessive whitespace
    page_text = re.sub(r'\s+', ' ', page_text).strip()
    
    # Use AI to filter for current product content only
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting product information from e-commerce websites. Your task is to extract ONLY the ORIGINAL raw text content that relates to the current product being viewed. Include all relevant product information such as descriptions, specifications, features, materials, care instructions, specifications, etc. EXCLUDE: navigation menus, headers, footers, related products, recommended items, advertisements, site-wide information, other products, reviews, and any content not directly about the main product. Return the filtered text as plain text, preserving the ORIGINAL text and structure of the product information. Do not add any information not within the page and make no summaries."
                },
                {
                    "role": "user",
                    "content": f"Extract only the current product information from this webpage text:\n\n{page_text[:20000]}"
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        filtered_content = response.choices[0].message.content.strip()
        return filtered_content
        
    except Exception as e:
        # Fallback: return the original text if AI filtering fails
        return page_text

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/scrape-product", response_model=ProductInfo)
async def scrape_product(request: UrlRequest):
    """Scrape product images and raw text content"""
    url = str(request.url)
    
    # Get webpage content
    soup = get_webpage_content(url)
    
    # Extract images using AI
    images = extract_images_from_html(soup, url)
    
    # Extract product content using AI
    raw_product_text = extract_product_content(soup)
    
    return ProductInfo(
        images=images,
        raw_product_text=raw_product_text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

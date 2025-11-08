from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from bs4 import BeautifulSoup, SoupStrainer
import json
from typing import List, Optional, Dict, Any
from openai import AsyncAzureOpenAI
import os
from urllib.parse import urljoin, urlparse
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn

app = FastAPI(title="Scrape Agent", version="1.0.0")

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Dataset of site-specific CSS classes and URL keywords to limit scraping scope for improved performance
SITE_CLASS_FILTERS = {
    "coupang": {
        "classes": [
            "product-btf-container",
            "prod-atf"
        ],
        "url_keywords": []
    },
    "amazon": {
        "classes": [],
        "url_keywords": ["aplus-media-library-service-media"]
    }
}

class UrlRequest(BaseModel):
    url: HttpUrl

class ProductInfo(BaseModel):
    images: List[str] = []
    raw_product_text: str = ""

# Initialize Azure OpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

def get_site_classes(url: str) -> Optional[List[str]]:
    """Get CSS classes to filter for a specific site if URL matches any configured site"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    for site_key, config in SITE_CLASS_FILTERS.items():
        if site_key in domain:
            return config.get("classes", [])
    return None

def get_site_url_keywords(url: str) -> Optional[List[str]]:
    """Get URL keywords to filter for a specific site if URL matches any configured site"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    for site_key, config in SITE_CLASS_FILTERS.items():
        if site_key in domain:
            return config.get("url_keywords", [])
    return None

def _parse_html_content(content: bytes, url: str) -> BeautifulSoup:
    """Parse HTML content in thread pool"""
    site_classes = get_site_classes(url)
    if site_classes:
        strainer = SoupStrainer(class_=lambda x: x and any(cls in x.split() for cls in site_classes))
        soup = BeautifulSoup(content, 'html.parser', parse_only=strainer)
    else:
        soup = BeautifulSoup(content, 'html.parser')
    return soup

async def get_webpage_content(url: str) -> BeautifulSoup:
    """Fetch webpage content and return BeautifulSoup object"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client_http:
            response = await client_http.get(url, headers=headers)
            response.raise_for_status()

        # Parse HTML in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        soup = await loop.run_in_executor(thread_pool, _parse_html_content, response.content, url)
        
        return soup
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webpage: {str(e)}")

def filter_and_deduplicate_images(images: List[str], url: str) -> List[str]:
    """Filter images to only include valid image extensions and remove duplicates, keeping smallest subset"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tiff', '.tif'}
    
    # Get site-specific URL keywords for filtering
    url_keywords = get_site_url_keywords(url)
    
    # First filter: only keep URLs that end with image extensions
    image_urls = []
    for img_url in images:
        parsed_url = urlparse(img_url)
        path = parsed_url.path.lower()
        if any(path.endswith(ext) for ext in valid_extensions):
            # Apply URL keyword filtering if configured for this site
            if url_keywords:
                if any(keyword in img_url for keyword in url_keywords):
                    image_urls.append(img_url)
            else:
                image_urls.append(img_url)
    
    # Second filter: remove duplicates and keep smallest subset
    # Group URLs by their base (everything before any extension)
    url_groups = {}
    for img_url in image_urls:
        # Find the base URL by removing everything after the image extension
        for ext in valid_extensions:
            if img_url.lower().endswith(ext):
                base_url = img_url[:img_url.lower().rfind(ext) + len(ext)]
                if base_url not in url_groups:
                    url_groups[base_url] = []
                url_groups[base_url].append(img_url)
                break
    
    # Keep only the shortest URL from each group (smallest subset)
    filtered_urls = []
    for base_url, url_list in url_groups.items():
        shortest_url = min(url_list, key=len)
        filtered_urls.append(shortest_url)
    
    return filtered_urls

def extract_images_fallback(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract image URLs using BeautifulSoup as fallback"""
    images = []
    
    # Find all img tags
    img_tags = soup.find_all('img')
    
    for img in img_tags:
        # Check src attribute
        src = img.get('src')
        if src:
            absolute_url = urljoin(base_url, src)
            if absolute_url not in images:
                images.append(absolute_url)
        
        # Check data-src for lazy loaded images
        data_src = img.get('data-src')
        if data_src:
            absolute_url = urljoin(base_url, data_src)
            if absolute_url not in images:
                images.append(absolute_url)
        
        # Check data-lazy-src, data-original, and other common lazy loading attributes
        for attr in ['data-lazy-src', 'data-original', 'data-lazy', 'data-image', 'data-img-src']:
            lazy_src = img.get(attr)
            if lazy_src:
                absolute_url = urljoin(base_url, lazy_src)
                if absolute_url not in images:
                    images.append(absolute_url)
    
    return images

async def extract_images_from_html(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract image URLs from filtered HTML sections"""
    images = []
    
    # First try BeautifulSoup extraction
    fallback_images = extract_images_fallback(soup, base_url)
    
    # Get raw HTML content for AI analysis
    html_content = str(soup)
    
    try:
        response = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing e-commerce HTML to extract product related images. Extract image URLs from img src, data-src, srcset, and other image attributes. Return ONLY a JSON array of complete URLs - no explanations. Focus on actual product images, ignore logos, icons, ads."
                },
                {
                    "role": "user",
                    "content": f"Base URL: {base_url}\n\nExtract all image URLs from this HTML:\n{html_content[:15000]}"
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        ai_response = response.choices[0].message.content.strip()
        try:
            # Clean up the response
            if ai_response.startswith(''):
                ai_response = ai_response[7:-3]
            elif ai_response.startswith(''):
                ai_response = ai_response[3:-3]
            
            ai_images = json.loads(ai_response)
            if isinstance(ai_images, list):
                # Convert relative URLs to absolute
                for img_url in ai_images:
                    if isinstance(img_url, str) and img_url.strip():
                        clean_url = img_url.strip()
                        if clean_url.startswith('//'):
                            # Protocol-relative URLs
                            parsed_base = urlparse(base_url)
                            clean_url = f"{parsed_base.scheme}:{clean_url}"
                        elif clean_url.startswith('/'):
                            clean_url = urljoin(base_url, clean_url)
                        elif not clean_url.startswith(('http://', 'https://')):
                            clean_url = urljoin(base_url, clean_url)
                        
                        if clean_url not in images:
                            images.append(clean_url)
        except json.JSONDecodeError:
            # If AI response is not valid JSON, use fallback
            images = fallback_images
    except Exception:
        # If AI fails, use fallback
        images = fallback_images
    
    # Combine AI results with fallback results, removing duplicates
    all_images = images + [img for img in fallback_images if img not in images]
    
    # Filter out common non-product images
    filtered_images = []
    for img_url in all_images:
        lower_url = img_url.lower()
        # Skip obvious non-product images
        if any(skip in lower_url for skip in ['logo', 'icon', 'favicon', 'sprite', 'button', 'arrow', 'social']):
            continue
        filtered_images.append(img_url)
    
    # Apply image filtering and deduplication with URL keyword filtering
    final_images = filter_and_deduplicate_images(filtered_images, base_url)
    
    return final_images

def _extract_text_content(soup: BeautifulSoup) -> str:
    """Extract text content in thread pool"""
    # Remove scripts, styles, and other non-content elements
    for element in soup.find_all(['script', 'style', 'noscript']):
        element.decompose()
    
    # Get the full page text content
    page_text = soup.get_text(separator=' ', strip=True)
    
    # Clean up excessive whitespace
    page_text = re.sub(r'\s+', ' ', page_text).strip()
    return page_text

async def extract_product_content(soup: BeautifulSoup) -> str:
    """Use AI to extract only current product content, excluding related items and irrelevant sections"""
    
    # Extract text in thread pool
    loop = asyncio.get_event_loop()
    page_text = await loop.run_in_executor(thread_pool, _extract_text_content, soup)
    
    try:
        response = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "Extract only current product information from this pre-filtered text. Include descriptions, specifications, features, materials, care instructions. For reviews, summarize highlighting quantity of feedback about quality, durability, allergens, warnings. Preserve original text structure and content."
                },
                {
                    "role": "user",
                    "content": f"Extract current product information from: {page_text[:15000]}"
                }
            ],
            temperature=0.1,
            max_tokens=5000
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

    # Get webpage content (automatically filtered by site classes if applicable)
    soup = await get_webpage_content(url)
    
    # Extract images and content concurrently
    images_task = extract_images_from_html(soup, url)
    content_task = extract_product_content(soup)
    
    images, raw_product_text = await asyncio.gather(images_task, content_task)
    
    return ProductInfo(
        images=images,
        raw_product_text=raw_product_text
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup thread pool on shutdown"""
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=4, loop="asyncio")

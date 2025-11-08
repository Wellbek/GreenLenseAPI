from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup, SoupStrainer
import json
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
import os
from urllib.parse import urljoin, urlparse
import re

app = FastAPI(title="Scrape Agent", version="1.0.0")

# Dataset of site-specific CSS classes to limit scraping scope for improved performance
SITE_CLASS_FILTERS = {
    "coupang.com": [
        "product-btf-container",
        "prod-atf"
    ]
}

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

def get_site_classes(url: str) -> Optional[List[str]]:
    """Get CSS classes to filter for a specific site if URL matches any configured site"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    for site_key, classes in SITE_CLASS_FILTERS.items():
        if site_key in domain:
            return classes
    return None

def get_webpage_content(url: str) -> BeautifulSoup:
    """Fetch webpage content and return BeautifulSoup object"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Filter by site-specific classes during parsing if available
        site_classes = get_site_classes(url)
        if site_classes:
            # Parse only elements with specified classes
            strainer = SoupStrainer(class_=lambda x: x and any(cls in x.split() for cls in site_classes))
            soup = BeautifulSoup(response.content, 'html.parser', parse_only=strainer)
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            
        return soup
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webpage: {str(e)}")

def extract_images_from_html(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract image URLs from filtered HTML sections"""
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
                        "content": "Extract product image URLs from this filtered HTML. Return as JSON array of strings. Focus on main product images, avoid thumbnails, recommended and related products, advertisements."
                    },
                    {
                        "role": "user",
                        "content": f"Base URL: {base_url}\n\nHTML: {html_content}"
                    }
                ],
                temperature=0,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith(''):
                content = content[7:-3]
            elif content.startswith(''):
                content = content[3:-3]
            
            ai_images = json.loads(content)
            for url in ai_images:
                if isinstance(url, str) and url.strip():
                    absolute_url = urljoin(base_url, url.strip())
                    if absolute_url not in images:
                        images.append(absolute_url)
        except Exception:
            pass  # Continue with filtered results
    
    return images

def extract_product_content(soup: BeautifulSoup) -> str:
    """Use AI to extract only current product content, excluding related items and irrelevant sections"""
    
    # Remove scripts, styles, and other non-content elements
    for element in soup.find_all(['script', 'style', 'noscript']):
        element.decompose()
    
    # Get the full page text content
    page_text = soup.get_text(separator=' ', strip=True)
    
    # Clean up excessive whitespace
    page_text = re.sub(r'\s+', ' ', page_text).strip()
    
    try:
        response = client.chat.completions.create(
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

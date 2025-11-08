from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial

app = FastAPI(title="Allergen Agent", version="1.0.0")

# Load allergen data at startup
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "allergens.csv")
try:
    allergen_df = pd.read_csv(CSV_PATH)
    allergen_df = allergen_df.dropna()
except FileNotFoundError:
    print(f"Warning: Allergen data file not found at {CSV_PATH}")
    allergen_df = None
except Exception as e:
    print(f"Error loading allergen data: {e}")
    allergen_df = None

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=8)

class TextileDescriptionRequest(BaseModel):
    description: str

def process_allergen_row(row, desc):
    """Process a single allergen row for matching"""
    if pd.isna(row.get('Allergen / Source')):
        return None
        
    allergen_source = str(row['Allergen / Source'])
    keywords = [k.strip().lower() for k in allergen_source.split(',') if k.strip()]
    
    if any(k in desc for k in keywords if k):
        return {
            "allergen": allergen_source,
            "common_symptoms": str(row.get('Common Symptoms', 'N/A')),
            "notes": str(row.get('Less Common Symptoms / Notes', 'N/A'))
        }
    return None

def analyze_allergens_sync(description, df):
    """Synchronous allergen analysis for thread pool execution"""
    desc = description.lower().strip()
    
    # Process rows in parallel using thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_allergen_row, row, desc) 
                  for _, row in df.iterrows()]
        
        matches = []
        for future in futures:
            result = future.result()
            if result is not None:
                matches.append(result)
    
    return matches

@app.get("/")
async def root():
    return {"message": "Allergen Agent API is running."}

@app.post("/analyze/textile")
async def analyze_textile(payload: TextileDescriptionRequest):
    if allergen_df is None:
        raise HTTPException(status_code=500, detail="Allergen data not loaded.")
    
    if not payload.description or not payload.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")

    # Run allergen analysis in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    matches = await loop.run_in_executor(
        thread_pool, 
        analyze_allergens_sync, 
        payload.description, 
        allergen_df
    )

    if not matches:
        return {
            "result": "No common textile allergens identified.",
            "disclaimer": "This information is based on common allergen data and may vary by individual. It is not medical advice."
        }

    return {
        "matches": matches,
        "disclaimer": "This information is based on common allergen data and may vary by individual. It is not medical advice."
    }

@app.on_event("shutdown")
async def shutdown_event():
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, workers=4)

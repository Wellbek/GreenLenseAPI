from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(title="Allergen Agent", version="1.0.0")

# Load allergen data at startup
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "allergens.csv")
try:
    allergen_df = pd.read_csv(CSV_PATH)
except Exception as e:
    allergen_df = None

class TextileDescriptionRequest(BaseModel):
    description: str

@app.get("/")
async def root():
    return {"message": "Allergen Agent API is running."}

@app.post("/analyze/textile")
async def analyze_textile(payload: TextileDescriptionRequest):
    if allergen_df is None:
        raise HTTPException(status_code=500, detail="Allergen data not loaded.")

    desc = payload.description.lower()
    matches = []
    for _, row in allergen_df.iterrows():
        keywords = [k.strip().lower() for k in row['Allergen / Source'].split(',')]
        if any(k in desc for k in keywords):
            matches.append({
                "allergen": row['Allergen / Source'],
                "common_symptoms": row['Common Symptoms'],
                "notes": row['Less Common Symptoms / Notes']
            })

    if not matches:
        return {
            "result": "No common textile allergens identified.",
            "disclaimer": "This information is based on common allergen data and may vary by individual. It is not medical advice."
        }

    return {
        "matches": matches,
        "disclaimer": "This information is based on common allergen data and may vary by individual. It is not medical advice."
    }

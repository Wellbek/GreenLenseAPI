# GreenLense Agentic AI
## Overview
The Greenlense API evaluates a product’s eco-friendliness, ethical sourcing, and social impact based on product data. It returns a Green Score along with an optional sustainability report for transparency.<br/>
This API is designed to power:<br/>

Browser Extensions – Highlight sustainability scores while browsing platforms like Coupang.<br/>
E-commerce Integrations – Partners query the API directly for product scoring.<br/>
Standalone Apps – Enable sustainability search, comparison, and insights.<br/>

# Core Concept
The challenge is to estimate a traceable sustainability score (eco, ethical, social) from minimal or unstructured data by enriching it with additional context about:

Brand: Reputation, certifications, controversies.<br/>
Product Category & Materials.<br/>
Implicit Semantics: Extracted from descriptions and images.<br/>

# Solution: Multi-Layer Scoring
## Text Layer

### Keyword Extractor: 
Extract brand, materials, category.<br/>
### Image Analyzer:
OCR for text.<br/>
Logo detection.<br/>
Image context description.<br/>


### Feature Combiner: 
Merge text and image features.<br/>

## Research Layer

### Brand Research Agent:
Live web search.<br/>
Wikipedia summaries.<br/>
Sustainability reports.<br/>

### Certification Lookup Agent:
Extract certifications from metadata.<br/>
Query certification databases.<br/>

### Footprint Agent:
Estimate environmental footprint.<br/>

### Public Sentiment Agent:
Fetch recent mentions to assess reputation.<br/>

## Reasoning & Scoring Layer
Aggregate all results into a structured context object.<br/>
Pass data to LLM Reasoning Engine for scoring.<br/>

## Example Output

{
  "product": "Nike Air Zoom Pegasus 40 Running Shoes",
  "eco_score": 3.2,
  "ethical_score": 2.8,
  "reputation_score": 3.9,
  "sources": [
    "Nike Official Website",
    "XYZ API",
    "XYZ Index",
    "XYZ Database",
    "Product Image (Packaging Text)"
  ],
  "explanation": "The Nike Air Zoom Pegasus 40 uses at least 20% recycled content by weight and features partially recycled mesh. However, the brand has faced criticism for limited supply-chain transparency and past labor-related issues in Southeast Asia. Public reputation remains moderate due to strong sustainability marketing and global brand trust."
}

# Architecture

<img width="767" height="498" alt="Screenshot 2025-11-09 at 10 19 56 AM" src="https://github.com/user-attachments/assets/c1a54625-4d0d-4fed-b545-929a8cc85d44" />


## Microservices

Each agent runs as an independent service for scalability and fault isolation.<br/>
Multiple workers per service for process-level parallelism.<br/>
Async/Await for high concurrency during I/O operations.<br/>

## Execution Flow

User sends request to API Gateway (port 8000).<br/>
Gateway forwards to Orchestrator (port 8002).<br/>
Orchestrator calls agents in parallel.<br/>
Aggregates responses → LLM Reasoning Engine → Returns final scores.<br/>

## Tech Stack

FastAPI for API endpoints.<br/>
Redis for caching frequently accessed data.<br/>
Azure for deployment.<br/>
LangExtract for text extraction.<br/>
Azure Vision for image analysis.<br/>
Docker for containerization.<br/>


## File Structure
```Greenlense-api/
├── docker-compose.yml
├── .env
├── requirements.txt
│
├── api-gateway/
│   ├── Dockerfile
│   └── main.py
│
├── orchestrator/
│   ├── Dockerfile
│   └── main.py
│
├── agents/
│   ├── text-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── image-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── allegen-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── brand-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── cert-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── footprint-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── sentiment-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── reasoning-agent/
│       ├── Dockerfile
│       └── main.py
│
├── shared/
│   └── utils.py
│
└── data/
    ├── [data files]
    └── redis.conf
```
# Getting Started
## Prerequisites

Python 3.10+<br/>
Docker & Docker Compose<br/>
Redis<br/>

## Setup
git clone [https://github.com/your-org/greenlense-api.git](https://github.com/Wellbek/GreenLenseAPI.git) <br/>
cd greenlense-api <br/>
docker-compose up --build <br/>

## Environment Variables
Create a .env file in the root directory with the following variables:<br/>

`AZURE_OPENAI_API_KEY=your_azure_openai_api_key`<br/>
`AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint/`<br/>
`AZURE_OPENAI_API_VERSION=your_api_version`<br/>
``AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name``<br/>
`VISION_KEY=your_azure_vision_key`<br/>

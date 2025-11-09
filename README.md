# GreenLense Agentic AI
## Overview
The Greenlense API evaluates a product’s eco-friendliness, ethical sourcing, and social impact based on product data. It returns a Green Score along with an optional sustainability report for transparency.
This API is designed to power:

Browser Extensions – Highlight sustainability scores while browsing platforms like Coupang.
E-commerce Integrations – Partners query the API directly for product scoring.
Standalone Apps – Enable sustainability search, comparison, and insights.

# Core Concept
The challenge is to estimate a traceable sustainability score (eco, ethical, social) from minimal or unstructured data by enriching it with additional context about:

Brand: Reputation, certifications, controversies.
Product Category & Materials.
Implicit Semantics: Extracted from descriptions and images.

# Solution: Multi-Layer Scoring
## Text Layer

### Keyword Extractor: 
Extract brand, materials, category.
### Image Analyzer:
OCR for text.
Logo detection.
Image context description.


### Feature Combiner: 
Merge text and image features.

## Research Layer

### Brand Research Agent:
Live web search.
Wikipedia summaries.
Sustainability reports.

### Certification Lookup Agent:
Extract certifications from metadata.
Query certification databases.

### Footprint Agent:
Estimate environmental footprint.

### Public Sentiment Agent:
Fetch recent mentions to assess reputation.

## Reasoning & Scoring Layer
Aggregate all results into a structured context object.
Pass data to LLM Reasoning Engine for scoring.

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

![alt text](image.png)

## Microservices

Each agent runs as an independent service for scalability and fault isolation.
Multiple workers per service for process-level parallelism.
Async/Await for high concurrency during I/O operations.

## Execution Flow

User sends request to API Gateway (port 8000).
Gateway forwards to Orchestrator (port 8002).
Orchestrator calls agents in parallel.
Aggregates responses → LLM Reasoning Engine → Returns final scores.

## Tech Stack

FastAPI for API endpoints.
Redis for caching frequently accessed data.
Azure for deployment.
LangExtract for text extraction.
Azure Vision for image analysis.
Docker for containerization.


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

Python 3.10+
Docker & Docker Compose
Redis

## Setup
git clone https://github.com/your-org/greenlense-api.git
cd greenlense-api
docker-compose up --build

## Environment Variables
Create a .env file in the root directory with the following variables:

`AZURE_OPENAI_API_KEY=your_azure_openai_api_key`
`AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint/`
`AZURE_OPENAI_API_VERSION=your_api_version`
``AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name``
`VISION_KEY=your_azure_vision_key`
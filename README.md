# GreenLense Agentic AI API

## Product Insight Intelligence API

GreenLense is an agentic product-insight system that generates comprehensive evaluations of consumer products across multiple dimensions: quality, durability, sustainability, materials, certifications, brand reputation, public sentiment, and more.

The system uses a coordinated orchestra of specialized research agents to transform minimal product data into a thorough comparative insight report. This enables:

* Instant product comparison and ranking
* Reduced uncertainty and increased purchase confidence
* Better customer experience through clarity and transparency
* Improved product matching between user needs and product attributes
* Higher conversions with fewer returns

GreenLense was originally designed as a sustainability-scoring API, but has evolved into a full multi-dimensional product-intelligence platform after identifying a larger market.

## Core Concept

The goal is to compute reliable and traceable product insights from minimal or unstructured data by enriching input with external research and contextual signals.

GreenLense aims to extract and analyze:

* Brand reputation and corporate track record
* Materials and product construction
* Durability and long-term quality indicators
* Certifications, compliance, and safety data
* Sustainability and ethical considerations
* Public sentiment and historical reliability
* Category-specific reference knowledge

All extracted and researched signals are consolidated into a unified evaluation.

## System Overview

GreenLense operates through a multi-layered pipeline composed of extraction, research, and reasoning stages.

### 1. Extraction Layer

**Text Agent**

* Brand extraction
* Material and component identification
* Product category inference
* Semantic signal extraction from descriptions

**Image Agent**

* OCR from packaging or product images
* Logo detection
* Visual material indicators
* Additional contextual cues

A unified feature combiner merges all extracted text and image information.

### 2. Research Layer (Agent Orchestra)

This layer enriches the product data through multiple specialized agents. For example:

* **Brand Research Agent**
  Corporate reputation, controversies, trust indexes, sustainability reports

* **Certification Agent**
  Identification and validation of certifications from text, metadata, or images

* **Durability & Quality Agent**
  Category benchmarks, common failure modes, long-term reliability indicators

* **Footprint & Sustainability Agent**
  Material and supply-chain environmental impact estimation

* **Sentiment Agent**
  Public sentiment, common complaints, long-term satisfaction indicators

* **Allergen / Material Risk Agent**
  Safety, allergen exposure, and hazardous material detection

These agents operate independently and in parallel.

### 3. Reasoning & Aggregation Layer

The orchestrator compiles all extracted and researched data into a structured context object.
A reasoning agent (LLM-based) generates:

* Quality score
* Durability score
* Sustainability score
* Ethical score
* Reputation score
* Key insights
* Risk flags
* Full narrative explanation
* Source list

# Architecture

<img width="1001" height="664" alt="image" src="https://github.com/user-attachments/assets/6190a08d-b846-4862-b968-e3131001a0fa" />

### Microservices

* Each agent is an independent microservice
* Multi-worker configurations for parallelism
* Asynchronous orchestrator for high-concurrency I/O
* Plan for redis caching for frequently accessed research data

### Execution Flow

1. Request sent to API Gateway (port 8000)
2. Gateway forwards to Orchestrator (port 8002)
3. Orchestrator invokes all agents in parallel
4. Responses are aggregated and passed to the Reasoning Agent
5. Final structured insight report is returned

# Tech Stack

* FastAPI for all internal and external service endpoints
* Redis for caching
* Azure Vision for image processing
* Azure OpenAI for reasoning and structured scoring
* Docker + Docker Compose for containerized microservices
* LangExtract for advanced text parsing

# File Structure

```
Greenlense-api/
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
│   ├── allergen-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── scrape-agent/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── orchestrator/
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

---

# Getting Started

## Prerequisites

* Python 3.10+
* Docker & Docker Compose
* Redis

## Setup

```bash
git clone https://github.com/Wellbek/GreenLenseAPI.git
cd greenlense-api
docker-compose up --build
```

## Environment Variables

Create a `.env` file:

```
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT_NAME=
VISION_KEY=
```

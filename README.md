# PharmaAI Assistant

A multi-layered pharmaceutical intelligence platform running entirely on local infrastructure with an NVIDIA H200 GPU. No external inference APIs — all models run locally.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Interfaces                           │
│           Rasa Bot (NLU + Dialogue)   FastAPI REST               │
└─────────────────────────┬────────────────────┬───────────────────┘
                          │                    │
┌─────────────────────────▼────────────────────▼───────────────────┐
│                     RAG Pipeline                                  │
│  1. Domain Classify → 2. Query Expand → 3. Hybrid Search         │
│  4. Build Prompt    → 5. SLM Generate → 6. Cache + Metrics       │
└──────────┬──────────────────────────────────────┬────────────────┘
           │                                      │
┌──────────▼───────────┐              ┌───────────▼──────────────┐
│  Embedding Layer (H200 GPU)         │  Storage Layer            │
│  BioBERT  (768-dim)                 │  PostgreSQL  (metadata)   │
│  ClinicalBERT (768-dim)             │  FAISS IVF   (2304-dim)   │
│  ChemBERTa (768-dim)                │  Redis       (cache)      │
│  Combined: 2304-dim                 │  RabbitMQ    (queue)      │
│  Fine-tuned Phi-3 (generation)      └──────────────────────────┘
└──────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                            │
│  PubMed · OpenFDA/FAERS · ClinicalTrials · NewsAPI · BioRxiv     │
│  Google CSE · EventRegistry · Local datasets (SIDER, CORD-19)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

```bash
# On the H200 server
nvidia-smi          # Verify GPU
nvcc --version      # CUDA toolkit
python --version    # Python 3.11+
```

### 2. Clone & Install

```bash
git clone <repo-url> pharma_ai && cd pharma_ai
pip install poetry
poetry install
```

### 3. Configure

```bash
cp .env.template .env
# Edit .env with your API keys, DB passwords, notification credentials
```

### 4. Download Models (internet machine)

```bash
python scripts/download_models.py --models all --token YOUR_HF_TOKEN
rsync -av ./models/ user@h200-server:/path/to/pharma_ai/models/
```

### 5. Start Infrastructure

```bash
docker-compose up -d postgres redis rabbitmq prometheus grafana
```

### 6. Initialise Database & Index

```bash
python scripts/init_db.py
```

### 7. Ingest Data

```bash
python scripts/run_ingestion.py
```

### 8. Fine-tune the SLM (optional but recommended)

```bash
# Generate training data from ingested docs
python training/data_generator.py

# Fine-tune with QLoRA on H200
python training/train_slm.py \
  --model_path ./models/slm \
  --data_path  ./data/processed/train.jsonl \
  --output_dir ./models/slm_finetuned \
  --epochs 3
```

### 9. Start the API

```bash
uvicorn pharmaai.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 10. Start the Rasa Bot

```bash
bash scripts/run_bot.sh
```

### 11. Or: Full Docker Deployment

```bash
docker-compose up -d
```

---

## API Reference

| Method | Endpoint       | Description                               |
| ------ | -------------- | ----------------------------------------- |
| POST   | `/v1/ask`      | Ask a pharmaceutical question (RAG)       |
| POST   | `/v1/search`   | Semantic + keyword hybrid document search |
| POST   | `/v1/classify` | Classify text into domain + content type  |
| GET    | `/health`      | Health check                              |
| GET    | `/metrics`     | Prometheus metrics                        |
| GET    | `/docs`        | Swagger UI                                |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the most common adverse reactions for metformin?",
    "top_k": 5
  }'
```

### Example: Search documents

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phase III clinical trial diabetes",
    "domain": "r&d",
    "top_k": 10
  }'
```

---

## Project Structure

```
pharma_ai/
├── pharmaai/
│   ├── core/           # Config, database, schemas, cache, metrics
│   ├── embeddings/     # BioBERT/ClinicalBERT/ChemBERTa + FAISS index
│   ├── processing/     # Domain classifier, text formatter, summariser
│   ├── ingestion/      # PubMed, OpenFDA, ClinicalTrials, NewsAPI workers
│   ├── retrieval/      # Hybrid search (dense + BM25), RRF fusion
│   ├── inference/      # SLM, RAG pipeline, query generator, notifications
│   └── api/            # FastAPI app, routes, middleware
├── rasa_bot/           # NLU, stories, rules, custom actions
├── training/           # SLM fine-tuning, classifier training, evaluation
├── scripts/            # init_db, backfill, ingestion runner, benchmarks
├── configs/            # settings.yaml, logging.yaml, prometheus, grafana
├── tests/              # pytest suite
├── docker-compose.yml
└── Dockerfile
```

---

## Domains

| Domain            | Data Sources                        |
| ----------------- | ----------------------------------- |
| Pharmacovigilance | OpenFDA FAERS, SIDER                |
| R&D               | PubMed, ClinicalTrials.gov, BioRxiv |
| Regulation        | FDA/EMA guidelines, ISO standards   |
| Formulas          | PubChem, ChEMBL, DrugBank           |
| Internal          | SOPs, inventory logs, HR docs       |

---

## GPU Memory Layout (H200 80GB)

| Component       | Approx. VRAM |
| --------------- | ------------ |
| BioBERT         | ~1.5 GB      |
| ClinicalBERT    | ~1.5 GB      |
| ChemBERTa       | ~0.5 GB      |
| Phi-3 (4-bit)   | ~2.5 GB      |
| FAISS GPU index | ~2–8 GB      |
| **Total**       | **~8–14 GB** |

Remaining ~66 GB available for fine-tuning or additional models.

---

## Monitoring

- **Grafana**: http://localhost:3000 (admin / see .env)
- **Prometheus**: http://localhost:9090
- **RabbitMQ UI**: http://localhost:15672

---

## Running Tests

```bash
pytest tests/ -v --cov=pharmaai
```

## Benchmark

```bash
python scripts/benchmark.py
```

## Evaluate

```bash
python training/eval.py \
  --eval_data ./data/processed/eval.jsonl \
  --output ./data/eval_results.json
```

## Backfill index after model upgrade

```bash
python scripts/backfill.py --batch-size 128
```

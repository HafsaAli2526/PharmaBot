# PharmaAI .gitignore Strategy Guide

## Overview

This `.gitignore` file is configured to:
✅ Track all **source code** (pharmaai/, rasa_bot/, scripts/, training/, tests/)
✅ Track **configuration files** (pyproject.toml, alembic.ini, docker-compose.yml, Dockerfile)
✅ Track **documentation** (README.md, configs/ with YAML files)
❌ Exclude **secrets** (.env file with API keys)
❌ Exclude **build artifacts** (Python cache, eggs, wheels)
❌ Exclude **generated data** (logs, processed data, FAISS indices)
❌ Exclude **large model files** (downloaded pretrained models)

---

## What Gets Ignored & Why

### 🔐 **SECRETS & CREDENTIALS** (Critical for Security)

- `.env` - Contains database passwords, API keys, Twilio tokens, etc.
- Never commit these files!
- Users should create their own `.env` by copying the template in the codebase

### 📦 **PYTHON BUILD ARTIFACTS**

- `__pycache__/` - Python bytecode cache
- `*.pyc`, `*.pyo` - Compiled Python files
- `*.egg-info/`, `build/`, `dist/` - Build outputs
- `.venv/`, `venv/` - Virtual environments (should be recreated by users)

### 📊 **GENERATED DATA & INDICES** (Too Large)

- `data/logs/` - Runtime log files
- `data/raw/` - Downloaded datasets from PubMed, FDA, ClinicalTrials, etc.
- `data/processed/` - Normalized data after ETL
- `*.faiss` - FAISS vector indices (can be 1GB+)
- `index_id_map.json` - Mapping files for indices

### 🤖 **PRETRAINED MODELS** (Users Download Separately)

- `models/biobert/` - BioBERT embeddings
- `models/chemberta/` - ChemBERTa models
- `models/clinicalberta/` - ClinicalBERT models
- `models/slm/` - Phi-3 LLM
- `models/slm_finetuned/` - Fine-tuned version

Users run: `python scripts/download_models.py` to populate this directory.

### 🧪 **TESTING & CODE QUALITY**

- `.pytest_cache/` - Pytest cache
- `.mypy_cache/`, `.ruff_cache/` - Linter caches
- `.coverage` - Coverage reports

### 🏗️ **IDE & EDITOR FILES**

- `.vscode/` - VS Code workspace settings
- `.idea/` - PyCharm settings
- `*.swp`, `*.swo` - Vim swap files
- `.DS_Store` - macOS metadata

### 📦 **RASA BOT ARTIFACTS**

- `rasa_bot/models/` - Trained NLU models (can be regenerated)
- `rasa_bot/logs/`, `rasa_bot/events.db` - Runtime data

---

## How to Deploy to HuggingFace

Since you plan to push to HuggingFace Hub, follow this workflow:

### 1. **Initial Setup**

```bash
# Create .env file locally (not committed)
cp .env.example .env
# Fill in your credentials

# Create local directories
python scripts/init_db.py
python scripts/download_models.py --models all
```

### 2. **Initialize Git & Push**

```bash
git init
git add .
git commit -m "Initial commit: PharmaAI source code"
git branch -M main
git remote add origin https://huggingface.co/spaces/your-org/pharmaai
git push -u origin main
```

### 3. **What Gets Pushed**

- ✅ All `.py` code files
- ✅ Configuration templates (like `.env` without secrets)
- ✅ README, Docker files
- ✅ Tests and scripts (so users can run them)
- ❌ Data, logs, indices, downloaded models
- ❌ Personal `.env` with real credentials

### 4. **Setup Instructions for Users**

Include in `README.md`:

````markdown
## Getting Started

### Download Required Models

```bash
python scripts/download_models.py --models all
```
````

### Initialize Database

```bash
python scripts/init_db.py
```

### Create Environment File

```bash
cp .env .env.local
# Edit with your API keys and database credentials
```

````

---

## .gitkeep Files

Four `.gitkeep` files were added to preserve directory structure:
- `models/.gitkeep` - Placeholder for downloaded models
- `data/logs/.gitkeep` - For runtime logs
- `data/raw/.gitkeep` - For ingested raw data
- `data/processed/.gitkeep` - For processed datasets

These ensure directories exist even when empty, but the actual content is gitignored.

---

## Best Practices

1. **Before First Commit:**
   - [ ] Verify `.env` is not tracked: `git status`
   - [ ] Test that tests can run: `pytest tests/`
   - [ ] Verify `.gitignore` catches large files: `git add -n . | wc -l`

2. **In CI/CD (GitHub Actions, Huggingface Spaces):**
   - Run `poetry install` to fetch dependencies
   - Run `python scripts/download_models.py` to load models in CI environment
   - Secrets should be managed via GitHub Secrets or Huggingface Secrets

3. **For Large Files (Models, Data):**
   - Consider using Git LFS (Large File Storage) if pushing models < 5GB
   - Or provide download scripts (already done with `download_models.py`)
   - Or reference data from external sources (PubMed, FDA APIs, etc.)

4. **Collaborators:**
   - Everyone should have their own `.env` (never committed)
   - `.gitkeep` files ensure they get the directory structure
   - Logs and data directories start empty

---

## Verifying Setup

```bash
# Check what will be committed
git status

# Verify secrets are ignored
git ls-files | grep -E "\.env|secrets|password" || echo "✓ No secrets found"

# Check size of commits
git ls-files | xargs wc -l  # Should be ~5-10K lines of code only
````

---

## Questions?

- **Large datasets?** Modify `data/raw/.gitkeep` to point to S3/cloud storage
- **Models versioning?** Use HuggingFace Hub with proper versioning
- **Want to track some logs?** Update `.gitignore` to allow specific log patterns
- **Database migrations?** `alembic/` and `alembic.ini` ARE tracked for schema migration

_THIS DOC WILL BE HELPFULLTO NEW COMMERS FOR NOT ONLY SETTING THERE FIRST FINE-TUNED MODEL AND MAKE IT LOCALLY AVAILIBLE FOR USE._

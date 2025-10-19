# Environment Setup Guide

This guide explains how to set up environment variables for the AEC Compliance Agent.

## Quick Setup

### 1. Create .env file

Create a `.env` file in the project root with your OpenAI API key:

```bash
# Create .env file
touch .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=your-actual-api-key-here" >> .env
```

### 2. Required Environment Variables

#### Essential for RAG System (Pilar 3)
```bash
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Customize OpenAI settings
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=1000
```

#### RAG System Configuration
```bash
# Embeddings model
EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Vectorstore settings
VECTORSTORE_DIR=vectorstore/normativa_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3
```

#### Data Paths
```bash
# Document directories
NORMATIVA_DIR=data/normativa
BLUEPRINTS_DIR=data/blueprints
EXTRACTED_DIR=data/extracted

# Output directories
OUTPUTS_DIR=outputs
LOGS_DIR=outputs/logs
REPORTS_DIR=outputs/reports
VISUALIZATIONS_DIR=outputs/visualizations
```

## Complete .env Template

Copy this content to your `.env` file and customize as needed:

```bash
# =============================================================================
# API KEYS
# =============================================================================

# OpenAI API Key (Required for RAG system - Pilar 3)
OPENAI_API_KEY=your-openai-api-key-here

# Google AI API Key (Optional - for alternative LLM)
GOOGLE_API_KEY=your-google-api-key-here

# =============================================================================
# RAG SYSTEM CONFIGURATION
# =============================================================================

# OpenAI Model Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=1000

# Embeddings Configuration
EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu

# Vectorstore Configuration
VECTORSTORE_DIR=vectorstore/normativa_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3

# =============================================================================
# DATA PATHS
# =============================================================================

# Document Directories
NORMATIVA_DIR=data/normativa
BLUEPRINTS_DIR=data/blueprints
EXTRACTED_DIR=data/extracted

# Output Directories
OUTPUTS_DIR=outputs
LOGS_DIR=outputs/logs
REPORTS_DIR=outputs/reports
VISUALIZATIONS_DIR=outputs/visualizations

# =============================================================================
# EXTRACTION CONFIGURATION
# =============================================================================

# DWG/DXF Processing
DXF_TOLERANCE=0.001
MIN_ROOM_AREA=1.0
MIN_DOOR_WIDTH=0.6

# CAD Layer Names (case-insensitive)
LAYER_ROOMS=habitaciones,rooms,espacios
LAYER_DOORS=puertas,doors,accesos
LAYER_WALLS=muros,walls,paramentos
LAYER_WINDOWS=ventanas,windows

# =============================================================================
# CALCULATION PARAMETERS
# =============================================================================

# Geometry Calculations
GEOMETRY_PRECISION=6
GRAPH_TOLERANCE=0.01

# Compliance Thresholds
MIN_DOOR_WIDTH_CM=80
MAX_EVACUATION_DISTANCE_M=30
MIN_CORRIDOR_WIDTH_CM=120

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=extraction.log

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Environment: development, testing, production
ENVIRONMENT=development

# Debug Mode
DEBUG=True

# Test Configuration
TEST_DATA_DIR=tests/fixtures
TEST_OUTPUT_DIR=test_outputs

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Parallel Processing
MAX_WORKERS=4
BATCH_SIZE=32

# Memory Management
MAX_MEMORY_USAGE=0.8
CACHE_SIZE=1000

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# API Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Data Privacy
ENABLE_DATA_LOGGING=False
ANONYMIZE_DATA=True

# =============================================================================
# INTEGRATION SETTINGS
# =============================================================================

# Database (if using external DB)
DATABASE_URL=sqlite:///aec_compliance.db

# Webhook URLs (for notifications)
WEBHOOK_URL=
SLACK_WEBHOOK_URL=

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable/Disable Features
ENABLE_RAG=True
ENABLE_VISUALIZATION=True
ENABLE_EXPORT=True
ENABLE_BATCH_PROCESSING=True

# Advanced Features
ENABLE_ML_PREDICTIONS=False
ENABLE_REAL_TIME_PROCESSING=False
```

## Alternative: Export Environment Variables

Instead of using a `.env` file, you can export variables directly:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-actual-api-key-here"

# Set other variables as needed
export OPENAI_MODEL="gpt-3.5-turbo"
export OPENAI_TEMPERATURE="0.1"
export VECTORSTORE_DIR="vectorstore/normativa_db"
```

## Verification

Test that your environment is set up correctly:

```bash
# Check if OpenAI API key is set
echo $OPENAI_API_KEY

# Run the RAG test script
python scripts/test_rag_system.py
```

## Security Notes

- **Never commit** your `.env` file to version control
- The `.env` file is already in `.gitignore`
- Keep your API keys secure and don't share them
- Use different keys for development and production

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Make sure you've created the `.env` file
   - Verify the variable name is exactly `OPENAI_API_KEY`
   - Check there are no spaces around the `=` sign

2. **"Module not found"**
   - Install dependencies: `pip install -r requirements.txt`
   - Make sure you're in the project root directory

3. **"Permission denied"**
   - Make the test script executable: `chmod +x scripts/test_rag_system.py`

## Next Steps

Once your environment is set up:

1. **Test the RAG system**: `python scripts/test_rag_system.py`
2. **Run unit tests**: `pytest tests/unit/rag/ -v`
3. **Start using the system** for building code queries

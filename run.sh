#!/bin/bash
# run.sh — GlamScan pipeline orchestrator
# Usage:
#   bash run.sh all       # Full pipeline: scrape → build → embed → index
#   bash run.sh scrape    # Scrape only
#   bash run.sh build     # Build dataset only
#   bash run.sh embed     # Embed only
#   bash run.sh index     # Build FAISS index only
#   bash run.sh api       # Start FastAPI server
#   bash run.sh frontend  # Start Streamlit UI

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies (one-time)
if [ ! -f "venv/bin/pip_installed" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    touch venv/bin/pip_installed
fi

# Function to print section headers
section() {
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo ""
}

case "${1:-all}" in
    scrape)
        section "SCRAPING: Amazon (500 per category)"
        python -m src.data.scrape_amazon --max 500
        ;;

    build)
        section "BUILDING: Merging & downloading images"
        python -m src.data.build_dataset
        ;;

    embed)
        section "EMBEDDING: Generating CLIP vectors"
        python -m src.models.infer_embedder
        ;;

    index)
        section "INDEXING: Building FAISS index"
        python -m src.retrieval.build_index
        ;;

    all)
        section "FULL PIPELINE"
        echo -e "${YELLOW}Step 1/4: Scraping${NC}"
        python -m src.data.scrape_amazon --max 500

        echo ""
        echo -e "${YELLOW}Step 2/4: Building${NC}"
        python -m src.data.build_dataset

        echo ""
        echo -e "${YELLOW}Step 3/4: Embedding${NC}"
        python -m src.models.infer_embedder

        echo ""
        echo -e "${YELLOW}Step 4/4: Indexing${NC}"
        python -m src.retrieval.build_index

        section "PIPELINE COMPLETE! ✅"
        echo "Start the API with:  bash run.sh api"
        echo "Start the UI with:   bash run.sh frontend"
        ;;

    api)
        section "STARTING: FastAPI Server"
        echo "📡 API running at http://localhost:8000"
        echo "📚 Docs available at http://localhost:8000/docs"
        echo ""
        uvicorn src.api.app:app --host 0.0.0.0 --port 8000
        ;;

    frontend)
        section "STARTING: Streamlit Frontend"
        echo "🎨 UI running at http://localhost:8501"
        echo ""
        streamlit run frontend/app.py
        ;;

    *)
        echo "Usage: bash run.sh {scrape|build|embed|index|all|api|frontend}"
        echo ""
        echo "Commands:"
        echo "  scrape      Scrape Sephora & Amazon"
        echo "  build       Merge data & download images"
        echo "  embed       Generate CLIP embeddings"
        echo "  index       Build FAISS index"
        echo "  all         Run full pipeline (1-4)"
        echo "  api         Start FastAPI backend"
        echo "  frontend    Start Streamlit UI"
        exit 1
        ;;
esac

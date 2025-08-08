#!/bin/bash

# =============================================================================
# Standalone Deep Search Environment Setup Script
# =============================================================================

set -e  # Exit on any error

echo "🚀 Setting up Standalone Deep Search Testing Environment"
echo "=" $(printf '=%.0s' {1..60})

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if running in git repository
if [ ! -d ".git" ]; then
    print_error "This script must be run from the root of the accelerated-discovery repository"
    echo "Please run: cd /path/to/accelerated-discovery && bash scripts/setup_standalone.sh"
    exit 1
fi

# Check for required tools
echo ""
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi
print_status "Python 3 found"

# Check UV
if ! command -v uv &> /dev/null; then
    print_warning "UV package manager not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install UV package manager"
        exit 1
    fi
fi
print_status "UV package manager found"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_warning "Docker not found - SearxNG will not be available"
    print_info "Install Docker from https://docs.docker.com/get-docker/"
    DOCKER_AVAILABLE=false
else
    print_status "Docker found"
    DOCKER_AVAILABLE=true
fi

# Check Docker Compose
if [ "$DOCKER_AVAILABLE" = true ] && ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose not found - trying docker compose plugin"
    if ! docker compose version &> /dev/null; then
        print_warning "Docker Compose plugin not available - SearxNG setup may fail"
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        DOCKER_COMPOSE_CMD="docker compose"
    fi
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

if [ "$DOCKER_AVAILABLE" = true ]; then
    print_status "Docker Compose available"
fi

# Set up Python environment
echo ""
print_info "Setting up Python environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    uv venv --python 3.12
fi
print_status "Virtual environment ready"

# Activate virtual environment
source .venv/bin/activate
print_status "Virtual environment activated"

# Install dependencies
print_info "Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
    uv sync
    print_status "Project dependencies installed via uv sync"
else
    print_info "No pyproject.toml found, installing basic dependencies..."
    uv add \
        pandas \
        matplotlib \
        seaborn \
        numpy \
        pydantic \
        pydantic-settings \
        aiohttp \
        beautifulsoup4 \
        lxml \
        requests \
        openai \
        anthropic \
        langchain \
        langchain-openai \
        langchain-anthropic \
        instructor \
        jupyterlab
    print_status "Basic dependencies installed"
fi

# Set up environment file
echo ""
print_info "Setting up environment configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Environment file created from template"
    else
        cat > .env << 'EOF'
# API Keys - Add your keys here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Semantic Scholar API key for academic papers
# SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

# SearxNG Configuration
SEARXNG_URL=http://localhost:8080

# Model Configuration
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.0
EOF
        print_status "Environment file created with template"
    fi
else
    print_status "Environment file already exists"
fi

print_warning "Please edit .env file and add your API keys before running the notebook"

# Set up SearxNG
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo ""
    print_info "Setting up SearxNG search engine..."
    
    # Create SearxNG directory and files
    mkdir -p searxng
    cd searxng
    
    # Copy configuration files
    cp ../config/searxng/docker-compose.yml .
    cp ../config/searxng/settings.yml searxng_settings.yml
    
    print_status "SearxNG configuration files created"
    
    # Check if SearxNG is already running
    if docker ps --format "table {{.Names}}" | grep -q "searxng_standalone"; then
        print_status "SearxNG container already running"
    else
        print_info "Starting SearxNG container..."
        if $DOCKER_COMPOSE_CMD up -d; then
            print_status "SearxNG container started"
            
            # Wait for SearxNG to be ready
            print_info "Waiting for SearxNG to start (up to 30 seconds)..."
            for i in {1..30}; do
                if curl -s "http://localhost:8080/search?q=test&format=json" > /dev/null 2>&1; then
                    print_status "SearxNG is ready and responding"
                    break
                fi
                sleep 1
                if [ $i -eq 30 ]; then
                    print_warning "SearxNG may still be starting up - check with 'docker logs searxng_standalone'"
                fi
            done
        else
            print_error "Failed to start SearxNG container"
            print_info "The notebook will work with mock data instead"
        fi
    fi
    
    cd ..
else
    print_warning "Docker not available - SearxNG will not be set up"
    print_info "The notebook will work with mock search results"
fi

# Set up Jupyter
echo ""
print_info "Setting up Jupyter Lab..."

# Ensure Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    uv add jupyterlab
fi

print_status "Jupyter Lab ready"

# Final setup verification
echo ""
print_info "Verifying installation..."

# Test Python imports
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
print('✅ Core packages import successfully')
" 2>/dev/null && print_status "Python packages verified" || print_warning "Some Python packages may have import issues"

# Test SearxNG if available
if [ "$DOCKER_AVAILABLE" = true ]; then
    if curl -s "http://localhost:8080/search?q=test&format=json" > /dev/null 2>&1; then
        print_status "SearxNG is responding correctly"
    else
        print_warning "SearxNG may not be fully ready yet"
    fi
fi

# Print summary
echo ""
echo "🎉 Setup complete!"
echo "=" $(printf '=%.0s' {1..60})
echo ""
print_info "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - OpenAI API key: https://platform.openai.com/api-keys"
echo "   - OR Anthropic API key: https://console.anthropic.com/"
echo ""
echo "2. Start Jupyter Lab:"
echo "   source .venv/bin/activate"
echo "   jupyter lab notebooks/deep_search_standalone.ipynb"
echo ""
echo "3. Run all cells in the notebook to start testing!"
echo ""

if [ "$DOCKER_AVAILABLE" = true ]; then
    print_info "SearxNG is available at: http://localhost:8080"
    echo "   - Test it: curl \"http://localhost:8080/search?q=machine+learning&format=json\""
    echo "   - Stop it: cd searxng && docker-compose down"
    echo ""
fi

print_info "Environment summary:"
echo "   - Python virtual environment: .venv/"
echo "   - Environment config: .env"
echo "   - SearxNG config: searxng/"
echo "   - Main notebook: notebooks/deep_search_standalone.ipynb"
echo ""

print_status "Ready to go! 🚀"
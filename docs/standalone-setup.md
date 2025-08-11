# Standalone Deep Search Testing Environment Setup

This guide provides complete instructions for setting up a standalone deep search testing environment, including local SearxNG installation.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git
- UV package manager

## Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery

# 2. Run the automated setup script
bash scripts/setup_standalone.sh

# 3. Start Jupyter and open the notebook
uv run jupyter lab notebooks/deep_search_testing.ipynb
```

## Manual Setup Instructions

### 1. Repository Setup

```bash
# Clone repository
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
# Optional: SEMANTIC_SCHOLAR_API_KEY for academic paper searches
```

### 3. Local SearxNG Installation

#### Option A: Docker Compose (Recommended)

```bash
# Create SearxNG directory
mkdir -p searxng
cd searxng

# Download custom configuration
curl -o docker-compose.yml https://raw.githubusercontent.com/NASA-IMPACT/accelerated-discovery/main/config/searxng/docker-compose.yml
curl -o settings.yml https://raw.githubusercontent.com/NASA-IMPACT/accelerated-discovery/main/config/searxng/settings.yml

# Start SearxNG
docker-compose up -d

# Verify installation
curl "http://localhost:8080/search?q=test&format=json"
```

#### Option B: Manual Installation

```bash
# Install SearxNG
pip install searxng

# Create configuration directory
mkdir -p ~/.config/searxng

# Copy custom configuration
cp config/searxng/settings.yml ~/.config/searxng/

# Start SearxNG
searxng-run
```

### 4. Install Jupyter and Extensions

```bash
# Install Jupyter Lab
uv add jupyterlab matplotlib seaborn

# Install optional extensions
uv add ipywidgets plotly
```

### 5. Verify Installation

```bash
# Test Python imports
python -c "
from akd.agents.search.deep_search import DeepLitSearchAgent
print('✅ AKD imports working')
"

# Test SearxNG
curl "http://localhost:8080/search?q=machine+learning&format=json" | head -5

# Start Jupyter
uv run jupyter lab notebooks/deep_search_testing.ipynb
```

## Configuration

### SearxNG Settings

The custom SearxNG configuration includes:

- **JSON Output Support**: Enables `format=json` parameter
- **Scientific Search Engines**: Academic databases, arXiv, PubMed
- **Rate Limiting**: Configured for research usage
- **Privacy**: No logging, no tracking

### Environment Variables

Required in `.env`:
```bash
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
SEARXNG_URL=http://localhost:8080  # Local SearxNG instance
```

### AKD Configuration

The notebook uses these default configurations that can be customized:

```python
# Research Parameters
MAX_RESEARCH_ITERATIONS = 3
QUALITY_THRESHOLD = 0.7
MIN_RELEVANCY_SCORE = 0.3
FULL_CONTENT_THRESHOLD = 0.7

# Search Behavior  
USE_SEMANTIC_SCHOLAR = True
ENABLE_PER_LINK_ASSESSMENT = True
ENABLE_FULL_CONTENT_SCRAPING = True
```

## Troubleshooting

### SearxNG Issues

**Port 8080 in use:**
```bash
# Check what's using port 8080
lsof -i :8080

# Use different port in docker-compose.yml
ports:
  - "8081:8080"

# Update SEARXNG_URL in .env
SEARXNG_URL=http://localhost:8081
```

**No JSON output:**
```bash
# Verify settings.yml has:
search:
  formats:
    - html
    - json
    - rss
```

### Python Environment Issues

**Import errors:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync --force
```

**API Key errors:**
```bash
# Verify .env file is in project root
# Check environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

## Usage

1. **Start SearxNG**: `docker-compose up -d` (in searxng directory)
2. **Activate environment**: `source .venv/bin/activate`
3. **Launch Jupyter**: `uv run jupyter lab`
4. **Open notebook**: `notebooks/deep_search_testing.ipynb`
5. **Run cells**: Execute the configuration and testing cells

## Sharing the Environment

To share this setup with others:

1. **Share the repository**: Include the `.env.example` file
2. **Include documentation**: This setup guide and the notebook
3. **Docker configuration**: The SearxNG docker-compose.yml
4. **Installation script**: `scripts/setup_standalone.sh` for automation

## Performance Notes

- **Local SearxNG**: Reduces external API dependencies
- **Caching**: SearxNG caches results for faster repeated searches
- **Rate Limiting**: Configured to respect search engine rate limits
- **Resource Usage**: Monitor Docker container resource usage

## Security Considerations

- **API Keys**: Keep them in `.env` file, never commit to git
- **SearxNG**: Runs locally, no external data sharing
- **Network**: SearxNG only accessible from localhost by default

## Advanced Configuration

### Custom Search Engines

Edit `searxng/settings.yml` to add domain-specific search engines:

```yaml
engines:
  - name: custom_academic
    engine: xpath
    search_url: https://your-academic-site.com/search?q={query}
    # ... xpath selectors
```

### Performance Tuning

Adjust Docker resources in `docker-compose.yml`:

```yaml
services:
  searxng:
    # ...
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
```

## Support

For issues with:
- **AKD Framework**: Open issue on GitHub repository
- **SearxNG**: Check [SearxNG documentation](https://docs.searxng.org/)
- **Setup**: Review troubleshooting section above
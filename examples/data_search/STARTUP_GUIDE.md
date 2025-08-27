# Data Discovery Frontend - Startup Guide

Quick reference for starting the Earth Science Data Discovery interface for any developer on the project.

## Prerequisites

- **Python 3.12+** with `uv` package manager installed
- **Node.js 16+** with `npm`
- **OpenAI API Key** (set in `.env` file)
- **Git** (for cloning the repository)

## Environment Setup (First Time Only)

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd accelerated-discovery
```

### 2. Choose Installation Option

**Option A: Minimal Dependencies (Recommended for Demo)**
```bash
# Navigate to data search demo directory
cd examples/data_search

# Create virtual environment and install minimal dependencies
uv sync
```

**Option B: Full AKD Framework**
```bash
# Install complete framework (from repository root)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys and configurations
# Required: OPENAI_API_KEY=your-api-key-here
```

### 4. Install Frontend Dependencies

```bash
# Navigate to frontend directory and install Node dependencies
cd examples/data_search/frontend
npm install
```

## Quick Start

### 1. Backend (FastAPI Server)

```bash
# From the backend directory (as you prefer!)
cd examples/data_search/backend

# Set PYTHONPATH so Python can find akd modules (3 levels up)
export PYTHONPATH="$(pwd)/../../..:$PYTHONPATH"
uv run python main.py
```

**Backend runs on:** http://localhost:8003

### 2. Frontend (React App)

```bash
# In a new terminal, from project root
cd examples/data_search/frontend

# Start React development server
npm start
```

**Frontend runs on:** http://localhost:3000

## CMR MCP Server Setup

The data search functionality relies on NASA's Common Metadata Repository (CMR) through a Model Context Protocol (MCP) server.

### Prerequisites for CMR MCP

- **NASA Earthdata Account** (register at https://urs.earthdata.nasa.gov)
- **CMR MCP Server** running on port 8080

### Setting up CMR MCP Server

The CMR MCP server is provided by NASA-IMPACT. Follow these steps:

```bash
# 1. Clone the NASA-IMPACT SDE Data Agents repository
git clone https://github.com/NASA-IMPACT/sde-data-agents
cd sde-data-agents

# 2. Install dependencies
uv sync

# 3. Start the FastAPI server with all MCP servers
uv run uvicorn app:app --host 0.0.0.0 --port 8080
```

**CMR MCP Server will run on:** http://localhost:8080

#### Available MCP Endpoints:
- **CMR Tools:** http://localhost:8080/mcp/cmr/mcp
- **SDE Tools:** http://localhost:8080/mcp/sde/mcp
- **Science Expansion:** http://localhost:8080/mcp/science-expansion/mcp
- **Health Check:** http://localhost:8080/health
- **API Docs:** http://localhost:8080/docs

### Verifying CMR MCP Connection

1. **Check MCP server health:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **View API documentation:**
   Open http://localhost:8080/docs in your browser

3. **Check backend logs:**
   The backend will attempt to connect to the CMR MCP server automatically. Monitor backend logs for connection status and any MCP-related messages.

## Using the Interface

1. **Open browser:** Go to http://localhost:3000
2. **Enter query:** Try queries like:
   - "Find MODIS vegetation data over Amazon rainforest"
   - "Sea surface temperature measurements from VIIRS"
   - "Rainfall data in Amazon since 1980"
3. **Watch progress:** Real-time updates through 7 workflow steps
4. **Explore results:** Expandable sections with download links

## Troubleshooting

### Port Conflicts
If ports 8003 or 3000 are in use:
```bash
# Check what's using ports
netstat -an | grep LISTEN | grep -E "(8003|3000)"

# Kill processes if needed
lsof -ti:8003 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Backend Issues
- **Import errors:** Run `uv sync` to install dependencies
- **API key errors:** Check `.env` file exists and has valid OpenAI key
- **MCP connection errors:** Backend tries to connect to MCP server on port 8080

### Frontend Issues
- **Blank page:** Check browser console (F12) for JavaScript errors
- **Search not working:** Verify backend is running on port 8003
- **WebSocket errors:** Check browser network tab for WebSocket connections

## Development Commands

```bash
# Backend development (auto-reload) - from project root
cd examples/data_search/backend && uv run python main.py

# Frontend development (auto-reload) - from project root
cd examples/data_search/frontend && npm start

# Check backend health
curl http://localhost:8003/

# View logs
# Backend logs show in terminal
# Frontend logs in browser console (F12)

# Run tests (if available)
# From project root: pytest tests/
```

## File Structure

```
examples/data_search/                    # Data search frontend example
├── backend/                             # FastAPI server
│   ├── main.py                         # Server entry point
│   ├── websocket_handler.py            # WebSocket connections
│   ├── search_progress.py              # Progress tracking
│   ├── requirements.txt                # Original backend dependencies
│   └── logs/                           # Application logs
├── frontend/                           # React app
│   ├── src/                            # React components
│   │   ├── App.js                      # Main application
│   │   └── config.js                   # Configuration
│   ├── public/                         # Static files
│   └── package.json                    # Node dependencies
├── pyproject.toml                      # Minimal dependencies configuration
├── DEPENDENCY_STRATEGY.md              # Dependency reduction strategy
├── STARTUP_GUIDE.md                    # This file
└── README.md                           # Detailed documentation
```

## Quick URLs

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8003
- **API Docs:** http://localhost:8003/docs (FastAPI auto-generated)
- **Health Check:** http://localhost:8003/

---

**Need help?** Check the detailed `README.md` or browser console for specific error messages.

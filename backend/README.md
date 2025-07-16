# AKD Backend API

A FastAPI-based backend for the Accelerated Knowledge Discovery (AKD) multi-agent system. This backend provides RESTful APIs for creating, executing, and resuming scientific research workflows using LangGraph.

## Features

- **Workflow Planning**: Create simple or full research workflows
- **Workflow Execution**: Execute workflows with state management
- **Resume Capability**: Resume workflows from any node without re-executing previous steps
- **Multi-Agent Architecture**: Literature search, code search, and report generation nodes
- **State Persistence**: Checkpoint-based state management using LangGraph

## Prerequisites

- Python 3.12 or higher
- SearxNG instance (optional - will use mock data if not available)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery

# Checkout the backend experiment branch
git checkout ajinkya/backend_experiment_v2

cd backend
```

### 2. Set Up Environment

Create and activate a virtual environment:

```bash
# Using uv (recommended)
uv venv --python 3.12
source .venv/bin/activate

# Or using standard venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using uv
uv sync

# Or using pip
pip install -r ../requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the parent directory:

```bash
cp ../.env.example ../.env
```

Edit `../.env` and add your configuration:

```env
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional (for SearxNG)
SEARXNG_BASE_URL=http://localhost:8080
SEARXNG_MAX_RESULTS=10
```

### 5. Run the API Server

```bash
python api.py
```

The server will start at `http://localhost:8000`

### 6. Run Tests

In a new terminal (with the API server running):

```bash
python test_apis.py
```

## API Endpoints

### Workflow Management

- **POST** `/api/workflow/plan` - Create a workflow plan
  ```json
  {
    "message": "Your research query",
    "workflow_type": "simple"
  }
  ```
  Note: `workflow_type` can be "simple" or "full"

- **POST** `/api/workflow/execute` - Execute a workflow
  ```json
  {
    "state": {},
    "workflow_id": "uuid"
  }
  ```
  Note: `state` contains the complete workflow state object

- **POST** `/api/workflow/resume` - Resume workflow from a specific node
  ```json
  {
    "state": {},
    "workflow_id": "uuid",
    "start_node": "code_search"
  }
  ```
  Note: `state` contains current state, `start_node` is where to resume from

- **GET** `/api/workflow/list` - List all workflows
- **GET** `/api/workflow/{workflow_id}` - Get workflow status

## Workflow Types

### Simple Workflow
- Single node: Literature Search
- Quick search for research papers and articles

### Full Workflow
- Three nodes: Literature Search → Code Search → Report Generation
- Comprehensive research with code examples and final report

## Node Types

1. **LiteratureSearchNode**: Searches for academic papers and articles using SearxNG
2. **CodeSearchNode**: Searches for relevant code implementations
3. **ReportGenerationNode**: Combines results into a structured report

## Running with SearxNG (Optional)

For real search results, run a local SearxNG instance:

```bash
docker run -d -p 8080:8080 searxng/searxng
```

Without SearxNG, the system will use mock data for testing.

## Development

### Project Structure

```
backend/
├── api.py                    # FastAPI application and endpoints
├── planner.py               # Workflow graph definitions and nodes
├── test_apis.py             # Comprehensive API tests
├── e2e_agents_node_template.py  # Example node implementations
└── README.md                # This file
```

### Adding New Nodes

1. Create a new class inheriting from `AbstractNodeTemplate`
2. Implement the `_execute` method
3. Add the node to the workflow graph in `planner.py`

### Testing Resume Functionality

The resume feature allows you to:
- Skip already executed nodes
- Start execution from any point in the workflow
- Preserve previous node outputs

Example:
```python
# Execute lit_search first
# Then resume from code_search without re-running lit_search
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the backend directory and the parent path is in Python path
2. **SearxNG connection errors**: The system will fall back to mock data
3. **Serialization errors**: Make sure you're using the AKDSerializer (already configured)

### Debug Mode

Enable debug logging in nodes by setting `debug=True` when creating nodes in `planner.py`.

## Contributing

1. Follow the existing node template patterns
2. Add tests for new functionality
3. Update this README for new features

## License

See the main repository for license information.
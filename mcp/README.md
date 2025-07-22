# AKD MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for the Accelerated Knowledge Discovery (AKD) framework. The MCP server allows MCP-compatible clients (such as Claude Desktop) to interact with the AKD research tools and workflows.

## Overview

The AKD MCP server provides MCP clients with access to:
- Literature search capabilities (Semantic Scholar, arXiv, etc.)
- Research workflow execution
- Scientific document analysis
- Query generation and refinement tools

### MCP Compatibility

This server implements the Model Context Protocol (MCP) standard and should work with any MCP-compatible client, including:
- Claude Desktop (Anthropic's official client)
- Custom MCP clients implementing the protocol
- Development tools that support MCP

The examples in this guide primarily reference Claude Desktop for configuration, but the server itself is client-agnostic.

## Prerequisites

1. **Python Environment**: Python 3.12+ with `uv` package manager
2. **API Keys**: Required API keys should be configured in your `.env` file:
   - `OPENAI_API_KEY` - For LLM operations
   - `SEARXNG_BASE_URL` - For web search capabilities (optional)

## Installation

From the project root directory:

```bash
# Create virtual environment
uv venv --python 3.12

# Activate it
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install all dependencies (including MCP)
uv sync
```

That's it! The MCP server and all dependencies are now installed.

## Quick Start

1. **Test the MCP server**:
   ```bash
   cd mcp
   python demo_client.py
   ```

2. **Configure Claude Desktop** (or other MCP client) with the configuration shown below.

3. **Start using the tools** in your MCP client!

## Configuration

If you prefer to configure manually, add the following to your MCP client's configuration file.

### For Claude Desktop

#### macOS
Location: `~/Library/Application Support/Claude/claude_desktop_config.json`

#### Windows
Location: `%APPDATA%\Claude\claude_desktop_config.json`

#### Linux
Location: `~/.config/Claude/claude_desktop_config.json`

### Configuration Content

```json
{
  "mcpServers": {
    "akd-research": {
      "command": "/path/to/your/project/.venv/bin/python",
      "args": [
        "/path/to/your/project/mcp/server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/your/project"
      }
    }
  }
}
```

**Important Notes**: 
- Replace `/path/to/your/project` with the absolute path to your `accelerated-discovery` directory
- The server will automatically load environment variables from your `.env` file, including `OPENAI_API_KEY`
- If you prefer, you can also specify API keys directly in the config (not recommended for security):
  ```json
  {
    "env": {
      "PYTHONPATH": "/path/to/your/project",
      "OPENAI_API_KEY": "your-api-key",
      "SEARXNG_BASE_URL": "http://localhost:8888"
    }
  }
  ```

### For Other MCP Clients

Other MCP-compatible clients may have different configuration formats. Please refer to your client's documentation for specific configuration instructions. The key information you'll need:

- **Server command**: `/path/to/your/project/.venv/bin/python`
- **Server arguments**: `["/path/to/your/project/mcp/server.py"]`
- **Environment variables**:
  - `PYTHONPATH`: Path to the project root
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `SEARXNG_BASE_URL`: (Optional) URL for SearxNG instance

## Available Tools

Once configured, your MCP client will have access to the following tools:

### 1. `run_research_workflow`
Execute a complete research workflow including literature search and optional code search.

**Parameters:**
- `query` (required): Research question to investigate
- `max_sources` (optional): Maximum number of sources to analyze (default: 10)

**Returns:**
- Generated search queries
- Literature findings with sources
- Code findings (if applicable)
- Summary of research
- Whether code search was performed

### 2. `generate_search_queries`
Generate multiple search queries from a research question.

**Parameters:**
- `initial_query` (required): Starting research question
- `num_queries` (optional): Number of queries to generate (default: 3)

**Returns:**
- List of generated queries
- Query category (science/technology/general)

### 3. `search_literature`
Search academic literature using SearxNG.

**Parameters:**
- `queries` (required): List of search queries
- `max_results` (optional): Maximum results to return (default: 10)
- `category` (optional): Search category (default: "science")

**Returns:**
- Search results with titles, URLs, and snippets
- Total number of results found

### 4. `analyze_content_for_code_search`
Determine if code repository search would be beneficial.

**Parameters:**
- `content` (required): Literature search results to analyze
- `original_query` (optional): Original research question

**Returns:**
- Boolean indicating if code search is needed
- Confidence score
- Reasoning for the decision

### 5. `search_code_repositories`
Search local code repositories for relevant implementations.

**Parameters:**
- `query` (required): Search query for code
- `max_results` (optional): Maximum results (default: 5)

**Returns:**
- Code snippets with file paths
- Repository information

### 6. `get_workflow_status`
Check the status of workflow components.

**Returns:**
- Status of each component (agents, tools)
- System health information

### 7. `stream_research_workflow`
Run the research workflow with step-by-step progress updates.

**Parameters:**
- `query` (required): Research question
- `num_queries` (optional): Number of queries to generate (default: 5)

**Returns:**
- Step-by-step progress information
- Final summary and results

## Troubleshooting

### Server Not Starting
1. Ensure the virtual environment is activated
2. Check that all dependencies are installed: `uv sync`
3. Verify the Python path in your configuration

### Connection Issues
1. Check the MCP server logs in your MCP client
2. Ensure the file paths in the configuration are absolute paths
3. Verify API keys are correctly set
4. For Claude Desktop specifically, check the logs at:
   - macOS: `~/Library/Logs/Claude/mcp-*.log`
   - Windows: `%APPDATA%\Claude\logs\mcp-*.log`
   - Linux: `~/.config/Claude/logs/mcp-*.log`

### Testing the Server

#### Option 1: Using the MCP Test Client (Recommended)
The test client provides an interactive interface to test all MCP tools:

```bash
cd mcp
python demo_client.py
```

This will launch an interactive session where you can:
- List all available tools
- Test individual tools with example data
- Run a complete demo workflow
- Check server status

Example commands:
```
mcp> list                           # List all tools
mcp> test generate_search_queries   # Test query generation
mcp> demo                          # Run full demo workflow
mcp> status                        # Check component status
mcp> help                          # Show all commands
mcp> quit                          # Exit
```

#### Option 2: Direct Server Test
You can also test the server standalone:
```bash
cd mcp
python server.py
```

This should output:
```
AKD Research MCP Server running on stdio...
```

## Development

To modify or extend the MCP server:

1. **Server Implementation**: `server.py`
   - Add new tools by defining async functions decorated with `@mcp.tool()`
   - Each tool should have clear parameter descriptions
   - The server imports directly from AKD framework sources
   - Custom classes (NeedsCodeSearchAgent, AKDWorkflow) are implemented in the server file

2. **Testing**: 
   - Run `python test_mcp_client.py` for interactive testing
   - Use the example scripts in `examples/`:
     - `akd_research_workflow_demo_v2.py` - Full workflow demonstration
     - `e2e_agents_node_template_v1.py` - Component testing

3. **Debugging**: 
   - Enable debug logging by setting `LOG_LEVEL=DEBUG` in your environment
   - Check your MCP client's logs for server output
   - The server uses loguru for logging
   - For debugging MCP protocol issues, set `MCP_DEBUG=1`

4. **Architecture**:
   - The server uses a simplified `AKDWorkflow` class that orchestrates:
     - Query generation using `QueryAgent`
     - Literature search using `SearxNGSearchTool`
     - Code search decision using `NeedsCodeSearchAgent`
     - Optional code search using `LocalRepoCodeSearchTool`
   - All components are initialized when the server starts

## Environment Variables

The MCP server respects the following environment variables:

- `OPENAI_API_KEY`: Required for LLM operations
- `SEARXNG_BASE_URL`: URL for SearxNG instance (optional)
- `PYTHONPATH`: Should include the project root directory
- `LOG_LEVEL`: Set to `DEBUG` for verbose logging

## Security Notes

- API keys are passed through environment variables, not hardcoded
- The server runs with the same permissions as the MCP client
- File system access is limited to the project directory
- The MCP protocol uses stdio for communication (no network exposure)
- Consider using a dedicated API key with limited permissions for the MCP server

## Support

For issues or questions:
1. Check the main project documentation in `docs/`
2. Review the example notebooks in `notebooks/`
3. Examine the test scripts in `scripts/`
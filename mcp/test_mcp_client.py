#!/usr/bin/env python3
"""
MCP Client for testing the AKD MCP Server using FastMCP.

This client connects to the AKD MCP server and provides an interactive
interface to test all available tools.
"""

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class AKDMCPClient:
    """MCP Client for testing AKD Research Workflow Server."""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_params: Optional[StdioServerParameters] = None
        self.available_tools: Dict[str, Any] = {}
        
    async def connect(self, server_script_path: str):
        """Connect to the MCP server."""
        project_root = Path(__file__).parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"
        
        # Check if virtual environment exists
        if not venv_python.exists():
            print(f"❌ Virtual environment not found at: {venv_python}")
            print("   Please run ./setup_mcp.sh first")
            return False
            
        # Set up server parameters
        self.server_params = StdioServerParameters(
            command=str(venv_python),
            args=[server_script_path],
            env={
                "PYTHONPATH": str(project_root),
            }
        )
        
        # Connect to server
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await self._initialize_session()
                    await self._interactive_loop()
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False
            
        return True
    
    async def _initialize_session(self):
        """Initialize the session and get available tools."""
        print("🚀 Connected to AKD MCP Server!")
        
        # Initialize the session
        await self.session.initialize()
        
        # Get available tools
        tools_response = await self.session.list_tools()
        self.available_tools = {tool.name: tool for tool in tools_response.tools}
        
        print(f"\n📦 Found {len(self.available_tools)} available tools:")
        for i, (name, tool) in enumerate(self.available_tools.items(), 1):
            print(f"  {i}. {name}: {tool.description}")
    
    async def _interactive_loop(self):
        """Run an interactive loop for testing tools."""
        print("\n" + "="*60)
        print("AKD MCP Client - Interactive Testing")
        print("="*60)
        print("\nCommands:")
        print("  list - List all available tools")
        print("  test <tool_name> - Test a specific tool")
        print("  demo - Run a demo workflow")
        print("  status - Check workflow status")
        print("  quit - Exit the client")
        print("\n")
        
        while True:
            try:
                command = input("mcp> ").strip().lower()
                
                if command == "quit":
                    print("👋 Goodbye!")
                    break
                elif command == "list":
                    await self._list_tools()
                elif command.startswith("test "):
                    tool_name = command[5:].strip()
                    await self._test_tool(tool_name)
                elif command == "demo":
                    await self._run_demo()
                elif command == "status":
                    await self._check_status()
                elif command == "help":
                    await self._show_help()
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _list_tools(self):
        """List all available tools."""
        print("\n📦 Available Tools:")
        for i, (name, tool) in enumerate(self.available_tools.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   Description: {tool.description}")
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                print(f"   Parameters: {json.dumps(tool.inputSchema, indent=6)}")
    
    async def _test_tool(self, tool_name: str):
        """Test a specific tool with example inputs."""
        if tool_name not in self.available_tools:
            print(f"❌ Tool '{tool_name}' not found")
            print(f"   Available tools: {', '.join(self.available_tools.keys())}")
            return
        
        # Tool-specific test cases
        test_cases = {
            "generate_search_queries": {
                "query": "climate change impact on coral reefs",
                "num_queries": 3
            },
            "search_literature": {
                "queries": ["coral reef bleaching climate change", "ocean acidification coral"],
                "category": "science",
                "max_results": 5
            },
            "analyze_content_for_code_search": {
                "content": "This study implements a machine learning model for predicting coral bleaching events using satellite data and temperature anomalies.",
                "original_query": "coral reef prediction models"
            },
            "search_code_repositories": {
                "queries": ["coral reef prediction", "climate model"],
                "max_results": 3
            },
            "run_research_workflow": {
                "query": "quantum computing applications in cryptography",
                "num_queries": 3,
                "max_sources": 5
            },
            "stream_research_workflow": {
                "query": "machine learning for climate prediction",
                "num_queries": 3
            },
            "get_workflow_status": {}
        }
        
        # Get test parameters
        params = test_cases.get(tool_name, {})
        
        print(f"\n🧪 Testing tool: {tool_name}")
        print(f"📝 Parameters: {json.dumps(params, indent=2)}")
        
        try:
            # Call the tool
            result = await self.session.call_tool(tool_name, params)
            
            # Display results
            print(f"\n✅ Success! Results:")
            if hasattr(result, 'content'):
                # Handle different content types
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        # Parse JSON if possible
                        try:
                            data = json.loads(content_item.text)
                            print(json.dumps(data, indent=2))
                        except:
                            print(content_item.text)
                    else:
                        print(content_item)
            else:
                print(json.dumps(result, indent=2, default=str))
                
        except Exception as e:
            print(f"❌ Error calling tool: {e}")
    
    async def _run_demo(self):
        """Run a demonstration workflow."""
        print("\n🎯 Running Demo Workflow...")
        print("This will demonstrate the complete research workflow.\n")
        
        demo_queries = [
            "applications of transformer models in scientific computing",
            "climate change impacts on biodiversity in tropical rainforests",
            "CRISPR gene editing recent advances and ethical considerations"
        ]
        
        print("Select a demo query:")
        for i, query in enumerate(demo_queries, 1):
            print(f"  {i}. {query}")
        print(f"  {len(demo_queries) + 1}. Enter custom query")
        
        try:
            choice = int(input("\nChoice (1-4): "))
            if 1 <= choice <= len(demo_queries):
                query = demo_queries[choice - 1]
            else:
                query = input("Enter your research query: ").strip()
                
            print(f"\n🔍 Research Query: {query}")
            print("="*60)
            
            # Step 1: Generate search queries
            print("\n📝 Step 1: Generating search queries...")
            queries_result = await self.session.call_tool(
                "generate_search_queries",
                {"query": query, "num_queries": 5}
            )
            queries_data = json.loads(queries_result.content[0].text)
            print(f"Generated {len(queries_data.get('generated_queries', []))} queries")
            for q in queries_data.get('generated_queries', []):
                print(f"  - {q}")
            
            # Step 2: Search literature
            print("\n📚 Step 2: Searching literature...")
            lit_result = await self.session.call_tool(
                "search_literature",
                {
                    "queries": queries_data.get('generated_queries', [query]),
                    "max_results": 10
                }
            )
            lit_data = json.loads(lit_result.content[0].text)
            print(f"Found {lit_data.get('total_results', 0)} results")
            
            # Step 3: Analyze for code search
            if lit_data.get('results'):
                content = " ".join([
                    f"{r.get('title', '')}: {r.get('content', '')}"
                    for r in lit_data.get('results', [])[:5]
                ])
                
                print("\n🔍 Step 3: Analyzing content for code search need...")
                analysis_result = await self.session.call_tool(
                    "analyze_content_for_code_search",
                    {"content": content[:1000], "original_query": query}
                )
                analysis_data = json.loads(analysis_result.content[0].text)
                print(f"Code search needed: {analysis_data.get('needs_code_search', False)}")
                print(f"Confidence: {analysis_data.get('confidence', 0):.2f}")
                
                # Step 4: Optional code search
                if analysis_data.get('needs_code_search'):
                    print("\n💻 Step 4: Searching code repositories...")
                    code_result = await self.session.call_tool(
                        "search_code_repositories",
                        {"queries": [query], "max_results": 5}
                    )
                    code_data = json.loads(code_result.content[0].text)
                    print(f"Found {code_data.get('total_results', 0)} code results")
            
            print("\n✅ Demo completed!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    async def _check_status(self):
        """Check the workflow status."""
        print("\n🔍 Checking workflow status...")
        
        try:
            result = await self.session.call_tool("get_workflow_status", {})
            status_data = json.loads(result.content[0].text)
            
            print("\n📊 Workflow Status:")
            for key, value in status_data.items():
                print(f"  {key}: {'✅' if value else '❌'} {value}")
                
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
    
    async def _show_help(self):
        """Show help information."""
        print("\n📖 Help:")
        print("\nAvailable Commands:")
        print("  list              - List all available tools with descriptions")
        print("  test <tool_name>  - Test a specific tool with example data")
        print("  demo              - Run a complete research workflow demo")
        print("  status            - Check the status of workflow components")
        print("  help              - Show this help message")
        print("  quit              - Exit the client")
        print("\nExample:")
        print("  mcp> test generate_search_queries")
        print("  mcp> demo")


async def main():
    """Main entry point."""
    print("🚀 AKD MCP Test Client")
    print("="*60)
    
    # Get server path
    server_path = Path(__file__).parent / "akd_mcp_server.py"
    
    if not server_path.exists():
        print(f"❌ Server script not found: {server_path}")
        return
    
    # Create and run client
    client = AKDMCPClient()
    await client.connect(str(server_path))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
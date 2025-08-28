"""
FastAPI backend for Data Discovery Interface.

Provides REST API and WebSocket endpoints for the CMR data search frontend.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from search_progress import SearchProgressHandler
from websocket_handler import ConnectionManager

from akd.agents.data_search import CMRDataSearchAgent, CMRDataSearchAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema
from akd.configs.data_search_config import get_config
from akd.utils.serialization import safe_model_dump

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Global state management
config = get_config()
connection_manager = ConnectionManager()
active_searches: Dict[str, Any] = {}

# Configure CORS based on environment
allowed_origins = ["http://localhost:3000"]  # React dev server
if config.environment.value != "development":
    # Add production/staging origins
    allowed_origins.extend(
        [
            f"https://{config.websocket.host}",
            f"http://{config.websocket.host}",
        ],
    )

app = FastAPI(
    title="Data Discovery API",
    description="Backend API for Earth Science Data Discovery Interface",
    version="1.0.0",
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """Request model for initiating a search."""

    query: str = Field(..., description="Scientific research query")
    temporal_range: Optional[str] = Field(
        None,
        description="Optional temporal constraint",
    )
    spatial_bounds: Optional[str] = Field(
        None,
        description="Optional spatial constraint",
    )
    max_results: int = Field(50, description="Maximum number of results")


class SearchResponse(BaseModel):
    """Response model for search initiation."""

    search_id: str = Field(..., description="Unique search identifier")
    status: str = Field(..., description="Search status")
    message: str = Field(..., description="Status message")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Data Discovery API is running",
        "timestamp": datetime.now().isoformat(),
        "environment": config.environment.value,
        "version": "1.0.0",
    }


@app.get("/config")
async def get_frontend_config():
    """Get frontend-safe configuration."""
    return config.get_frontend_config()


@app.post("/search", response_model=SearchResponse)
async def start_search(request: SearchRequest):
    """
    Initiate a new data search.

    Creates a unique search ID and starts the search process in the background.
    Real-time updates are sent via WebSocket connection.
    """
    search_id = str(uuid.uuid4())

    # Store search metadata
    active_searches[search_id] = {
        "id": search_id,
        "query": request.query,
        "status": "initiated",
        "created_at": datetime.now().isoformat(),
        "progress": {
            "current_step": "starting",
            "steps_completed": 0,
            "total_steps": 7,
        },
    }

    # Start search in background
    logger.info(f"🚀 About to create asyncio task for search {search_id}")
    task = asyncio.create_task(execute_search(search_id, request))
    logger.info(f"🚀 Asyncio task created successfully: {task}")

    # Add task completion callback for debugging
    def task_done_callback(task):
        if task.exception():
            logger.error(f"🚨 Background task failed: {task.exception()}")
        else:
            logger.info("✅ Background task completed successfully")

    task.add_done_callback(task_done_callback)

    return SearchResponse(
        search_id=search_id,
        status="initiated",
        message="Search started successfully",
    )


@app.get("/search/{search_id}")
async def get_search_status(search_id: str):
    """Get the current status of a search."""
    if search_id not in active_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    return active_searches[search_id]


@app.websocket("/ws/{search_id}")
async def websocket_endpoint(websocket: WebSocket, search_id: str):
    """
    WebSocket endpoint for real-time search progress updates.

    Clients should connect to this endpoint after initiating a search
    to receive live progress updates.
    """
    await connection_manager.connect(websocket, search_id)

    try:
        # Send initial connection confirmation
        await connection_manager.send_personal_message(
            {
                "type": "connection_established",
                "search_id": search_id,
                "message": "Connected to search progress stream",
            },
            search_id,
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (e.g., ping/pong)
                await websocket.receive_text()
                # Handle client messages if needed

            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(search_id)


async def execute_search(search_id: str, request: SearchRequest):
    """
    Execute the data search workflow with progress updates.

    This function runs the CMRDataSearchAgent workflow and emits
    progress updates via WebSocket for real-time UI updates.
    """
    logger.info(f"🔍 STEP 1: execute_search started for {search_id}")
    try:
        logger.info("🔍 STEP 2: About to update search status")
        # Update search status
        active_searches[search_id]["status"] = "running"

        logger.info("🔍 STEP 3: About to create progress handler")
        # Create progress handler for this search
        progress_handler = SearchProgressHandler(
            search_id=search_id,
            connection_manager=connection_manager,
        )
        logger.info("🔍 STEP 4: Progress handler created successfully")

        logger.info("🔍 STEP 5: About to access config object")
        logger.info(f"🔍 STEP 5.1: config.debug = {config.debug}")
        logger.info(f"🔍 STEP 5.2: config.mcp.endpoint = {config.mcp.endpoint}")

        logger.info("🔍 STEP 6: About to create CMRDataSearchAgentConfig")
        # Initialize the CMR Data Search Agent with centralized config
        agent_config = CMRDataSearchAgentConfig(
            debug=config.debug,
            mcp_endpoint=config.mcp.endpoint,
            max_collections_to_search=config.search.max_collections_to_search,
            collection_search_page_size=config.search.collection_search_page_size,
            granule_search_page_size=config.search.granule_search_page_size,
            enable_parallel_search=config.search.enable_parallel_search,
            collection_search_timeout=config.mcp.timeout_seconds,
            granule_search_timeout=config.mcp.timeout_seconds
            + 15.0,  # Slightly longer for granules
            min_collection_relevance_score=config.search.min_collection_relevance_score,
        )
        logger.info("🔍 STEP 7: About to create CMRDataSearchAgent")
        agent = CMRDataSearchAgent(config=agent_config, debug=config.debug)
        logger.info("🔍 STEP 8: CMRDataSearchAgent created successfully")

        logger.info("🔍 STEP 9: About to wait for WebSocket connection")
        # Wait for WebSocket connection to be ready
        connection_ready = await progress_handler.wait_for_connection_ready(timeout=5.0)
        if not connection_ready:
            logger.warning(
                f"WebSocket not ready for search {search_id}, proceeding anyway",
            )
        logger.info(f"🔍 STEP 10: WebSocket connection ready: {connection_ready}")

        logger.info("🔍 STEP 11: About to set progress handler on agent")
        # Set progress handler using the new safe method
        agent.set_progress_handler(progress_handler)
        logger.info("🔍 STEP 12: Progress handler set successfully")

        logger.info("🔍 STEP 13: About to create search parameters")
        # Prepare search parameters
        search_params = DataSearchAgentInputSchema(
            query=request.query,
            temporal_range=request.temporal_range,
            spatial_bounds=request.spatial_bounds,
            max_results=request.max_results,
        )
        logger.info(f"🔍 STEP 14: Search parameters created: {request.query}")

        logger.info("🔍 STEP 15: About to send initial progress update")
        # Send initial progress update
        await progress_handler.emit_progress_update(
            "search_started",
            {
                "query": request.query,
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info("🔍 STEP 16: Initial progress update sent")

        logger.info("🔍 STEP 17: About to execute agent._arun() - THE BIG ONE!")
        # Execute the search workflow
        result = await agent._arun(search_params)
        logger.info("🔍 STEP 18: agent._arun() completed successfully!")

        # Send final results
        await progress_handler.emit_progress_update(
            "search_completed",
            {
                "result": {
                    "granules": result.granules,
                    "search_metadata": result.search_metadata,
                    "total_results": result.total_results,
                    "collections_searched": result.collections_searched,
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Update final status
        active_searches[search_id]["status"] = "completed"
        active_searches[search_id]["result"] = safe_model_dump(result)

    except Exception as e:
        # Handle search errors
        error_msg = f"Search failed: {str(e)}"
        logger.error(f"🚨 SEARCH ERROR at some step: {error_msg}")
        logger.error(f"🚨 Exception type: {type(e)}")
        logger.error(f"🚨 Exception details: {e}")
        import traceback

        logger.error(f"🚨 Full traceback: {traceback.format_exc()}")

        active_searches[search_id]["status"] = "error"
        active_searches[search_id]["error"] = error_msg

        # Send error update via WebSocket
        try:
            if search_id in active_searches:
                # Create a temporary progress handler for error reporting
                temp_progress = SearchProgressHandler(search_id, connection_manager)
                await temp_progress.on_search_error(error_msg)
        except Exception as notify_error:
            logger.error(f"Failed to notify WebSocket of search error: {notify_error}")


if __name__ == "__main__":
    # Use configuration-based port and settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.websocket.port,
        reload=config.environment.value == "development",
        log_level="debug" if config.debug else "info",
    )

"""
Search progress handling for real-time updates during CMR data search workflow.
"""

import asyncio
import os

# Import AKD utilities
import sys
from typing import Any, Dict, List, Optional

from websocket_handler import ConnectionManager

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from akd.utils.logging import ContextualLogger, log_websocket_event
from akd.utils.serialization import create_error_message, create_progress_message


def calculate_chunk_parameters(
    total_items: int,
    chunk_size: int = 2,
) -> tuple[int, int]:
    """Calculate chunking parameters for reliable message delivery."""
    if total_items <= 0:
        return 0, 0
    total_chunks = (total_items + chunk_size - 1) // chunk_size
    return total_chunks, chunk_size


class SearchProgressHandler:
    """
    Handles progress updates during the CMR data search workflow.

    Emits structured progress events via WebSocket for real-time UI updates
    as the search progresses through its 7-step workflow. Uses message queuing
    with rate-limited delivery to prevent WebSocket message drop issues.
    """

    def __init__(self, search_id: str, connection_manager: ConnectionManager):
        self.search_id = search_id
        self.connection_manager = connection_manager
        self.current_step = 0
        self.total_steps = 7
        self.is_connection_ready = False

        # Set up logging
        self.logger = ContextualLogger("SearchProgressHandler", search_id)

        # Set up error callback for connection failures
        connection_manager.set_error_callback(search_id, self._on_connection_error)

        # Define the workflow steps
        self.steps = [
            "search_started",
            "scientific_expansion",
            "scientific_angles_generated",
            "cmr_queries_generated",
            "collections_searched",
            "collections_synthesized",
            "granules_searched",
            "search_completed",
        ]

    async def wait_for_connection_ready(self, timeout: float = 10.0) -> bool:
        """Wait for WebSocket connection to be ready."""
        if await self.connection_manager.wait_for_connection(self.search_id, timeout):
            self.is_connection_ready = True
            return True
        return False

    async def _on_connection_error(self, error_message: str):
        """Handle connection errors."""
        log_websocket_event(self.search_id, "CONNECTION_ERROR", error_message)
        self.logger.error(f"Connection error: {error_message}")
        self.is_connection_ready = False
        # Could implement additional error recovery logic here

    async def emit_progress_update(
        self,
        event_type: str,
        data: Dict[str, Any],
        retry_on_failure: bool = True,
    ):
        """
        Emit a progress update via WebSocket with error handling.

        Args:
            event_type: Type of progress event
            data: Event-specific data payload
            retry_on_failure: Whether to retry if connection fails
        """
        try:
            # Use standardized message creation with safe serialization
            message = create_progress_message(
                event_type=event_type,
                search_id=self.search_id,
                data=data,
                current_step=self.current_step,
                total_steps=self.total_steps,
            )

            # Queue the message for rate-limited delivery
            success = await self.connection_manager.enqueue_message(
                message,
                self.search_id,
            )

            if not success and retry_on_failure:
                log_websocket_event(
                    self.search_id,
                    "PROGRESS_UPDATE_FAILED",
                    event_type,
                )
                self.logger.warning(f"Failed to enqueue progress update {event_type}")
            else:
                log_websocket_event(
                    self.search_id,
                    "PROGRESS_UPDATE_QUEUED",
                    event_type,
                )

            return success

        except Exception as e:
            self.logger.error(f"Error creating progress message for {event_type}: {e}")
            return False

    async def on_search_started(self, query: str):
        """Called when search workflow begins."""
        self.current_step = 1
        await self.emit_progress_update(
            "search_started",
            {
                "message": "Starting data discovery workflow...",
                "query": query,
            },
        )

    async def on_scientific_expansion_started(self):
        """Called when document retrieval begins."""
        self.current_step = 2
        await self.emit_progress_update(
            "scientific_expansion_started",
            {
                "message": "Retrieving relevant scientific documents...",
            },
        )

    async def on_scientific_expansion_completed(self, documents: List[Dict[str, Any]]):
        """Called when document retrieval completes."""
        await self.emit_progress_update(
            "scientific_expansion_completed",
            {
                "message": f"Retrieved {len(documents)} reference documents",
                "documents": documents,
            },
        )

    async def on_scientific_angles_started(self):
        """Called when scientific angle generation begins."""
        self.current_step = 3
        await self.emit_progress_update(
            "scientific_angles_started",
            {
                "message": "Generating scientific research angles...",
            },
        )

    async def on_scientific_angles_generated(self, angles: List[Dict[str, Any]]):
        """Called when scientific angles are generated."""
        await self.emit_progress_update(
            "scientific_angles_generated",
            {
                "message": f"Generated {len(angles)} scientific angles",
                "angles": angles,
            },
        )

    async def on_cmr_queries_started(self):
        """Called when CMR query generation begins."""
        self.current_step = 4
        await self.emit_progress_update(
            "cmr_queries_started",
            {
                "message": "Converting angles to CMR search parameters...",
            },
        )

    async def on_cmr_queries_generated(self, queries: List[Dict[str, Any]]):
        """Called when CMR queries are generated."""
        await self.emit_progress_update(
            "cmr_queries_generated",
            {
                "message": f"Generated {len(queries)} CMR search queries",
                "queries": queries,
            },
        )

    async def on_collections_search_started(self, num_queries: int):
        """Called when collection searches begin."""
        self.current_step = 5
        await self.emit_progress_update(
            "collections_search_started",
            {
                "message": f"Executing {num_queries} collection searches...",
            },
        )

    async def on_collections_search_completed(self, search_summary: Dict[str, Any]):
        """Called when collection search completes with summary info."""
        await self.emit_progress_update(
            "collections_search_completed",
            {
                "message": search_summary.get("message", "Collection search completed"),
                "total_collections_found": search_summary.get(
                    "total_collections_found",
                    0,
                ),
            },
        )

    async def on_collections_synthesis_started(self):
        """Called when collection synthesis begins."""
        self.current_step = 6
        await self.emit_progress_update(
            "collections_synthesis_started",
            {
                "message": "Ranking and filtering collections...",
            },
        )

    async def on_collections_synthesized(
        self,
        selected_collections: List[Dict[str, Any]],
    ):
        """
        Called when collection synthesis completes.

        NOTE: Collections are sent in small chunks to prevent WebSocket message delivery issues.
        Large payloads were causing messages to be dropped or not delivered to the frontend.
        This chunking approach ensures reliable delivery of all collection data.
        """
        collections_count = len(selected_collections)

        # Send collections in reliable chunks to avoid large message delivery issues
        total_chunks, chunk_size = calculate_chunk_parameters(collections_count)

        self.logger.info(
            f"Sending {collections_count} collections in {total_chunks} chunks",
        )

        # Send initial notification
        await self.emit_progress_update(
            "collections_chunking_started",
            {
                "message": f"Selected {collections_count} best collections for data search",
                "total_collections": collections_count,
                "total_chunks": total_chunks,
            },
        )

        # Send collections in chunks
        for chunk_index in range(total_chunks):
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, collections_count)
            chunk_collections = selected_collections[start_idx:end_idx]

            await self.emit_progress_update(
                "collections_chunk",
                {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "collections": chunk_collections,
                    "chunk_size": len(chunk_collections),
                },
            )

            # Small delay between chunks to ensure reliable delivery
            if chunk_index < total_chunks - 1:  # Don't delay after last chunk
                await asyncio.sleep(0.1)

        # Send completion signal
        await self.emit_progress_update(
            "collections_chunking_completed",
            {
                "message": f"All {collections_count} collections transmitted",
                "total_collections": collections_count,
                "total_chunks": total_chunks,
            },
        )

    async def on_granules_search_started(self, num_collections: int):
        """Called when granule searches begin."""
        self.current_step = 7
        await self.emit_progress_update(
            "granules_search_started",
            {
                "message": f"Searching for data files in {num_collections} collections...",
            },
        )

    async def on_granules_found(
        self,
        granules: List[Dict[str, Any]],
        total_size_mb: Optional[float] = None,
    ):
        """Called when granules are found."""
        message = f"Found {len(granules)} data files"
        if total_size_mb:
            message += f" ({total_size_mb:.1f} MB total)"

        await self.emit_progress_update(
            "granules_found",
            {
                "message": message,
                "granules": granules,
                "total_size_mb": total_size_mb,
            },
        )

    async def on_search_completed(self, final_result: Dict[str, Any]):
        """Called when the entire search workflow completes."""
        self.current_step = self.total_steps
        await self.emit_progress_update(
            "search_completed",
            {
                "message": "Data discovery completed successfully!",
                "result": final_result,
            },
        )

    async def on_search_error(self, error_message: str, step: Optional[str] = None):
        """Called when an error occurs during search."""
        try:
            # Use standardized error message creation
            message = create_error_message(
                search_id=self.search_id,
                error=error_message,
                failed_step=step or f"step_{self.current_step}",
            )

            # Queue error message for delivery
            await self.connection_manager.enqueue_message(message, self.search_id)

        except Exception as e:
            self.logger.error(f"Error creating error message: {e}")

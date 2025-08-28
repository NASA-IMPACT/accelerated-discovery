"""
WebSocket connection management for real-time progress updates.
"""

import asyncio
import os

# Import logging utilities
import sys
from typing import Callable, Dict

from fastapi import WebSocket

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from akd.utils.logging import ContextualLogger, log_websocket_event
from akd.utils.serialization import safe_serialize


class ConnectionManager:
    """
    Manages WebSocket connections for search progress updates.

    Handles multiple concurrent searches with individual WebSocket connections
    for real-time progress streaming with error handling and recovery.
    Uses in-memory message queues with rate-limited delivery to prevent
    WebSocket message drop issues.
    """

    def __init__(self):
        # Map search_id to WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Track connection states
        self.connection_states: Dict[
            str,
            str,
        ] = {}  # 'connecting', 'connected', 'error', 'disconnected'
        # Error callbacks for notifying search workflows
        self.error_callbacks: Dict[str, Callable[[str], None]] = {}

        # Message queue system for rate-limited delivery
        # NOTE: This message queuing system is essential for reliable WebSocket delivery.
        # Without it, messages were being dropped or not delivered to the frontend.
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        self.delivery_delay: float = (
            0.1  # 100ms between messages - prevents message drops
        )

        # Logger for connection management
        self.logger = ContextualLogger("ConnectionManager")

    async def connect(self, websocket: WebSocket, search_id: str):
        """Accept a new WebSocket connection for a specific search."""
        try:
            self.connection_states[search_id] = "connecting"
            await websocket.accept()
            self.active_connections[search_id] = websocket
            self.connection_states[search_id] = "connected"

            # Start message queue and consumer for this search
            await self._start_message_queue(search_id)

            log_websocket_event(search_id, "CONNECTED")
            self.logger.info(f"WebSocket connected for search {search_id}")
        except Exception as e:
            self.connection_states[search_id] = "error"
            log_websocket_event(search_id, "CONNECTION_FAILED", str(e))
            self.logger.error(
                f"Failed to establish WebSocket connection for {search_id}: {e}",
            )
            raise

    def disconnect(self, search_id: str):
        """Remove a WebSocket connection and clean up associated resources."""
        if search_id in self.active_connections:
            del self.active_connections[search_id]
        if search_id in self.connection_states:
            self.connection_states[search_id] = "disconnected"
        if search_id in self.error_callbacks:
            del self.error_callbacks[search_id]

        # Clean up message queue and consumer task
        self._cleanup_message_queue(search_id)

        log_websocket_event(search_id, "DISCONNECTED")
        self.logger.info(f"WebSocket disconnected for search {search_id}")

    async def send_personal_message(
        self,
        message: dict,
        search_id: str,
        retry_count: int = 3,
    ):
        """Send a message to a specific search's WebSocket connection with retry logic."""
        if search_id not in self.active_connections:
            log_websocket_event(search_id, "NO_CONNECTION")
            self.logger.warning(f"No active connection for search {search_id}")
            return False

        websocket = self.active_connections[search_id]

        for attempt in range(retry_count):
            try:
                await websocket.send_text(safe_serialize(message))
                return True
            except Exception as e:
                log_websocket_event(search_id, "SEND_FAILED", f"attempt_{attempt + 1}")
                self.logger.warning(
                    f"Failed to send message to {search_id} (attempt {attempt + 1}): {e}",
                )

                if attempt == retry_count - 1:
                    # Final attempt failed - mark connection as error and notify callback
                    self.connection_states[search_id] = "error"
                    self.disconnect(search_id)

                    # Notify search workflow of connection failure
                    if search_id in self.error_callbacks:
                        try:
                            await self.error_callbacks[search_id](
                                f"WebSocket connection failed: {e}",
                            )
                        except Exception as callback_error:
                            self.logger.error(
                                f"Error callback failed for {search_id}: {callback_error}",
                            )

                    return False
                else:
                    # Wait before retry
                    await asyncio.sleep(0.5 * (attempt + 1))

        return False

    async def enqueue_message(self, message: dict, search_id: str) -> bool:
        """Add message to queue for rate-limited delivery instead of sending directly."""
        if search_id not in self.message_queues:
            log_websocket_event(search_id, "NO_QUEUE")
            self.logger.warning(f"No message queue for search {search_id}")
            return False

        try:
            # Non-blocking put - if queue is full, we could add size limits here
            self.message_queues[search_id].put_nowait(message)
            log_websocket_event(
                search_id,
                "MESSAGE_QUEUED",
                message.get("type", "unknown"),
            )
            return True
        except asyncio.QueueFull:
            log_websocket_event(search_id, "QUEUE_FULL", message.get("type", "unknown"))
            self.logger.error(f"Message queue full for search {search_id}")
            return False
        except Exception as e:
            log_websocket_event(search_id, "QUEUE_ERROR", str(e))
            self.logger.error(f"Failed to enqueue message for {search_id}: {e}")
            return False

    async def broadcast_to_search(self, message: dict, search_id: str):
        """Broadcast a message to all connections for a specific search."""
        # For now, same as enqueue_message since we have one connection per search
        return await self.enqueue_message(message, search_id)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def is_connected(self, search_id: str) -> bool:
        """Check if a search has an active WebSocket connection."""
        return (
            search_id in self.active_connections
            and self.connection_states.get(search_id) == "connected"
        )

    def get_connection_state(self, search_id: str) -> str:
        """Get the current connection state for a search."""
        return self.connection_states.get(search_id, "disconnected")

    def set_error_callback(self, search_id: str, callback: Callable[[str], None]):
        """Set an error callback for a specific search."""
        self.error_callbacks[search_id] = callback

    async def wait_for_connection(self, search_id: str, timeout: float = 10.0) -> bool:
        """Wait for a WebSocket connection to be established."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            state = self.connection_states.get(search_id, "disconnected")
            if state == "connected":
                return True
            elif state == "error":
                return False
            await asyncio.sleep(0.1)

        log_websocket_event(search_id, "CONNECTION_TIMEOUT")
        self.logger.warning(
            f"Timeout waiting for WebSocket connection for search {search_id}",
        )
        return False

    async def _start_message_queue(self, search_id: str):
        """Initialize message queue and start consumer for a search."""
        # Create message queue for this search
        self.message_queues[search_id] = asyncio.Queue()

        # Start rate-limited consumer task
        consumer_task = asyncio.create_task(self._message_consumer(search_id))
        self.consumer_tasks[search_id] = consumer_task

        self.logger.debug(f"Started message queue and consumer for search {search_id}")

    def _cleanup_message_queue(self, search_id: str):
        """Clean up message queue and consumer task for a search."""
        # Cancel consumer task
        if search_id in self.consumer_tasks:
            task = self.consumer_tasks[search_id]
            if not task.done():
                task.cancel()
            del self.consumer_tasks[search_id]

        # Drain and clean up message queue
        if search_id in self.message_queues:
            queue = self.message_queues[search_id]
            # Drain remaining messages to prevent memory leaks
            drained_count = 0
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                    drained_count += 1
                except asyncio.QueueEmpty:
                    break

            del self.message_queues[search_id]

            if drained_count > 0:
                self.logger.debug(
                    f"Drained {drained_count} messages from queue for search {search_id}",
                )

        self.logger.debug(f"Cleaned up message queue for search {search_id}")

    async def _message_consumer(self, search_id: str):
        """Rate-limited message consumer that delivers messages from queue to WebSocket."""
        queue = self.message_queues[search_id]

        try:
            while search_id in self.active_connections:
                try:
                    # Wait for message with timeout to periodically check connection state
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)

                    # Deliver message using existing direct send logic
                    success = await self._send_message_direct(message, search_id)
                    queue.task_done()

                    if success:
                        log_websocket_event(
                            search_id,
                            "MESSAGE_DELIVERED",
                            message.get("type", "unknown"),
                        )
                        # Rate limiting - this is the key fix for message drops!
                        await asyncio.sleep(self.delivery_delay)
                    else:
                        log_websocket_event(
                            search_id,
                            "DELIVERY_FAILED",
                            message.get("type", "unknown"),
                        )
                        # Longer delay on failure to avoid rapid retry
                        await asyncio.sleep(self.delivery_delay * 5)

                except asyncio.TimeoutError:
                    # Timeout is normal - just check if connection still active
                    continue
                except asyncio.CancelledError:
                    # Consumer task was cancelled - clean shutdown
                    break
                except Exception as e:
                    self.logger.error(f"Consumer error for {search_id}: {e}")
                    # Brief backoff on unexpected errors
                    await asyncio.sleep(1.0)

        except Exception as e:
            self.logger.error(f"Fatal consumer error for {search_id}: {e}")
        finally:
            self.logger.debug(f"Message consumer stopped for search {search_id}")

    async def _send_message_direct(self, message: dict, search_id: str) -> bool:
        """Direct WebSocket send without retry (retry is handled by consumer)."""
        if search_id not in self.active_connections:
            return False

        websocket = self.active_connections[search_id]

        try:
            await websocket.send_text(safe_serialize(message))
            return True
        except Exception as e:
            self.logger.warning(f"Direct message send failed for {search_id}: {e}")
            # Don't retry here - let consumer handle it
            return False

    def get_queue_stats(self, search_id: str) -> dict:
        """Get debugging information about message queue state."""
        if search_id not in self.message_queues:
            return {"error": "No queue found"}

        queue = self.message_queues[search_id]
        consumer_task = self.consumer_tasks.get(search_id)

        return {
            "queue_size": queue.qsize(),
            "consumer_running": consumer_task is not None and not consumer_task.done(),
            "connection_active": search_id in self.active_connections,
            "delivery_delay_ms": int(self.delivery_delay * 1000),
        }

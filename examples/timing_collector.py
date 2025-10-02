"""
TimingCollector - Hierarchical performance tracking for data search workflows.

Provides context managers and utilities to collect detailed timing data
for each component, topic, and decomposition in the search pipeline.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


class TimingCollector:
    """
    Collects hierarchical timing data for data search workflows.

    Supports component-level, path-level (topic/decomposition), and operation-level timing
    with context managers for clean instrumentation.
    """

    def __init__(self, search_id: str = None):
        """Initialize timing collector."""
        self.search_id = search_id or f"search_{int(time.time())}"
        self.start_time = time.time()

        # Hierarchical timing data
        self.timings = {
            "search_id": self.search_id,
            "start_time": datetime.now().isoformat(),
            "total_duration": None,
            "components": {},
        }

        # Current processing context
        self.current_topic_idx: Optional[int] = None
        self.current_decomp_idx: Optional[int] = None

        # Active timers stack for nested operations
        self._active_timers: List[Dict[str, Any]] = []

    def set_context(self, topic_idx: int = None, decomp_idx: int = None):
        """Set current processing context for path-specific timing."""
        self.current_topic_idx = topic_idx
        self.current_decomp_idx = decomp_idx

    @asynccontextmanager
    async def measure(self, component_name: str, **metadata):
        """
        Context manager for timing component execution.

        Args:
            component_name: Name of component being timed
            **metadata: Additional metadata to store with timing

        Usage:
            async with timer.measure("topic_splitting") as timer:
                result = await component.process(query)
                timer.add_metadata(topics_found=len(result.topics))
        """
        timer_data = {
            "component": component_name,
            "start_time": time.time(),
            "metadata": metadata.copy(),
            "topic_idx": self.current_topic_idx,
            "decomp_idx": self.current_decomp_idx,
        }

        self._active_timers.append(timer_data)

        # Create timer context object
        timer_context = TimerContext(timer_data)

        try:
            yield timer_context
        finally:
            # Calculate duration
            end_time = time.time()
            duration = end_time - timer_data["start_time"]
            timer_data["duration"] = duration
            timer_data["end_time"] = end_time

            # Remove from active timers
            self._active_timers.remove(timer_data)

            # Store in hierarchical structure
            self._store_timing(timer_data)

    def _store_timing(self, timer_data: Dict[str, Any]):
        """Store timing data in hierarchical structure."""
        component = timer_data["component"]
        duration = timer_data["duration"]
        topic_idx = timer_data["topic_idx"]
        decomp_idx = timer_data["decomp_idx"]
        metadata = timer_data["metadata"]

        # Initialize component data if needed
        if component not in self.timings["components"]:
            self.timings["components"][component] = {
                "total_duration": 0,
                "call_count": 0,
                "metadata": [],
            }

        comp_data = self.timings["components"][component]
        comp_data["total_duration"] += duration
        comp_data["call_count"] += 1

        # Store metadata with timing info
        timing_entry = {
            "duration": duration,
            "start_time": datetime.fromtimestamp(timer_data["start_time"]).isoformat(),
            "topic_idx": topic_idx,
            "decomp_idx": decomp_idx,
            **metadata,
        }
        comp_data["metadata"].append(timing_entry)

        # For path-specific components, also store in path structure
        if topic_idx is not None:
            self._store_path_timing(
                component,
                duration,
                topic_idx,
                decomp_idx,
                metadata,
            )

    def _store_path_timing(
        self,
        component: str,
        duration: float,
        topic_idx: int,
        decomp_idx: Optional[int],
        metadata: Dict[str, Any],
    ):
        """Store timing data organized by processing path."""
        if "paths" not in self.timings:
            self.timings["paths"] = {}

        # Topic-level path
        topic_key = f"topic_{topic_idx}"
        if topic_key not in self.timings["paths"]:
            self.timings["paths"][topic_key] = {
                "total_duration": 0,
                "components": {},
                "decompositions": {},
            }

        topic_data = self.timings["paths"][topic_key]
        topic_data["total_duration"] += duration

        if component not in topic_data["components"]:
            topic_data["components"][component] = {
                "total_duration": 0,
                "call_count": 0,
            }

        topic_data["components"][component]["total_duration"] += duration
        topic_data["components"][component]["call_count"] += 1

        # Decomposition-level path (if applicable)
        if decomp_idx is not None:
            decomp_key = f"decomp_{decomp_idx}"
            if decomp_key not in topic_data["decompositions"]:
                topic_data["decompositions"][decomp_key] = {
                    "total_duration": 0,
                    "components": {},
                }

            decomp_data = topic_data["decompositions"][decomp_key]
            decomp_data["total_duration"] += duration

            if component not in decomp_data["components"]:
                decomp_data["components"][component] = {
                    "total_duration": 0,
                    "call_count": 0,
                }

            decomp_data["components"][component]["total_duration"] += duration
            decomp_data["components"][component]["call_count"] += 1

    def add_metadata(self, component: str, **metadata):
        """Add metadata to the most recent timing for a component."""
        if component in self.timings["components"]:
            comp_data = self.timings["components"][component]
            if comp_data["metadata"]:
                # Add to most recent entry
                comp_data["metadata"][-1].update(metadata)

    def finalize(self) -> Dict[str, Any]:
        """Finalize timing collection and return complete timing data."""
        end_time = time.time()
        total_duration = end_time - self.start_time

        self.timings["total_duration"] = total_duration
        self.timings["end_time"] = datetime.now().isoformat()

        # Add summary statistics
        self.timings["summary"] = self._generate_summary()

        return self.timings

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from timing data."""
        summary = {
            "total_components": len(self.timings["components"]),
            "component_breakdown": {},
            "slowest_components": [],
            "path_breakdown": {},
        }

        # Component breakdown
        for comp_name, comp_data in self.timings["components"].items():
            total_time = comp_data["total_duration"]
            call_count = comp_data["call_count"]
            avg_time = total_time / call_count if call_count > 0 else 0

            summary["component_breakdown"][comp_name] = {
                "total_time": total_time,
                "average_time": avg_time,
                "call_count": call_count,
                "percentage": (total_time / self.timings["total_duration"]) * 100
                if self.timings["total_duration"]
                else 0,
            }

        # Slowest components
        sorted_components = sorted(
            summary["component_breakdown"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True,
        )
        summary["slowest_components"] = [
            {"component": name, **data} for name, data in sorted_components[:5]
        ]

        # Path breakdown (if available)
        if "paths" in self.timings:
            for path_name, path_data in self.timings["paths"].items():
                summary["path_breakdown"][path_name] = {
                    "total_time": path_data["total_duration"],
                    "percentage": (
                        path_data["total_duration"] / self.timings["total_duration"]
                    )
                    * 100
                    if self.timings["total_duration"]
                    else 0,
                    "decomposition_count": len(path_data.get("decompositions", {})),
                }

        return summary

    def get_component_stats(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific component."""
        if component_name not in self.timings["components"]:
            return None

        comp_data = self.timings["components"][component_name]
        durations = [entry["duration"] for entry in comp_data["metadata"]]

        if not durations:
            return None

        return {
            "total_time": comp_data["total_duration"],
            "call_count": comp_data["call_count"],
            "average_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "all_calls": comp_data["metadata"],
        }

    def print_summary(self):
        """Print a human-readable summary of timing data."""
        if not self.timings["total_duration"]:
            print("❌ Timing data not finalized yet")
            return

        summary = self.timings["summary"]

        print(f"\n⏱️  TIMING SUMMARY - Search ID: {self.search_id}")
        print(f"🕐 Total Duration: {self.timings['total_duration']:.2f}s")
        print(f"🧩 Components: {summary['total_components']}")

        print("\n📊 COMPONENT BREAKDOWN:")
        for comp_name, stats in summary["component_breakdown"].items():
            print(
                f"   {comp_name:25} {stats['total_time']:8.2f}s ({stats['percentage']:5.1f}%) [{stats['call_count']} calls]",
            )

        if summary["path_breakdown"]:
            print("\n🛤️  PATH BREAKDOWN:")
            for path_name, stats in summary["path_breakdown"].items():
                print(
                    f"   {path_name:15} {stats['total_time']:8.2f}s ({stats['percentage']:5.1f}%) [{stats['decomposition_count']} decomps]",
                )

        print("\n🐌 SLOWEST COMPONENTS:")
        for i, comp in enumerate(summary["slowest_components"][:3], 1):
            print(
                f"   {i}. {comp['component']:20} {comp['total_time']:8.2f}s ({comp['percentage']:5.1f}%)",
            )


class TimerContext:
    """Context object returned by timing context manager."""

    def __init__(self, timer_data: Dict[str, Any]):
        self.timer_data = timer_data

    def add_metadata(self, **metadata):
        """Add metadata to this timing entry."""
        self.timer_data["metadata"].update(metadata)

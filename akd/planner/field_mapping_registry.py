"""
Field mapping registry for managing explicit and LLM-generated field mappings.

Provides three-tier mapping strategy:
1. Explicit mappings (human-defined)
2. Exact name matching (automatic)
3. LLM-generated mappings (intelligent fallback)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from akd.utils import get_akd_root


class LLMMappingEntry(BaseModel):
    """Single LLM-generated mapping entry with metadata."""

    mapping: dict[str, str] = Field(..., description="Field mappings: target -> source")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="ISO timestamp of generation",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    user_approved: bool = Field(..., description="Whether user approved this mapping")
    reasoning: dict[str, str] = Field(
        default_factory=dict,
        description="Per-field reasoning for mappings",
    )


class FieldMappingRegistry:
    """
    Manages field mappings between agents.

    Supports:
    - Explicit mappings from field_mappings.json
    - LLM-generated mappings with approval tracking
    - Persistent caching of approved mappings
    """

    def __init__(
        self,
        explicit_path: str | None = None,
        llm_path: str | None = None,
    ):
        """
        Initialize field mapping registry.

        Args:
            explicit_path: Path to explicit mappings JSON file
            llm_path: Path to LLM-generated mappings JSON file
        """
        self.explicit_path = explicit_path or str(
            get_akd_root() / "akd" / "mapping" / "field_mappings.json",
        )
        self.llm_path = llm_path or str(
            get_akd_root() / "akd" / "mapping" / "llm_generated_mappings.json",
        )

        self.explicit_mappings: dict[str, dict[str, str]] = {}
        self.llm_mappings: dict[str, LLMMappingEntry] = {}

        self._load_mappings()

    def _load_mappings(self):
        """Load both explicit and LLM-generated mappings."""
        self._load_explicit()
        self._load_llm()

    def _load_explicit(self):
        """Load explicit mappings from JSON file."""
        try:
            path = Path(self.explicit_path)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    self.explicit_mappings = data.get("mappings", {})
                    logger.info(
                        f"Loaded {len(self.explicit_mappings)} explicit mappings from {self.explicit_path}",
                    )
            else:
                logger.info(f"No explicit mappings file found at {self.explicit_path}")
                self.explicit_mappings = {}
        except Exception as e:
            logger.error(f"Error loading explicit mappings: {e}")
            self.explicit_mappings = {}

    def _load_llm(self):
        """Load LLM-generated mappings from JSON file."""
        try:
            path = Path(self.llm_path)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    raw_mappings = data.get("mappings", {})

                    # Convert to LLMMappingEntry objects
                    self.llm_mappings = {key: LLMMappingEntry(**value) for key, value in raw_mappings.items()}

                    approved_count = sum(1 for entry in self.llm_mappings.values() if entry.user_approved)

                    logger.info(
                        f"Loaded {len(self.llm_mappings)} LLM mappings "
                        f"({approved_count} approved) from {self.llm_path}",
                    )
            else:
                logger.info(f"No LLM mappings file found at {self.llm_path}")
                self.llm_mappings = {}
        except Exception as e:
            logger.error(f"Error loading LLM mappings: {e}")
            self.llm_mappings = {}

    def get_mapping(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> dict[str, str] | None:
        """
        Get field mapping for source->target agent pair.

        Priority:
        1. Explicit mappings (highest trust)
        2. LLM-generated mappings (if user approved)

        Args:
            source_agent_id: Source agent identifier
            target_agent_id: Target agent identifier

        Returns:
            Dict mapping target_field_name -> source_field_name, or None
        """
        key = f"{source_agent_id}->{target_agent_id}"

        # Priority 1: Explicit mappings
        if key in self.explicit_mappings:
            logger.debug(f"Using explicit mapping for {key}")
            return self.explicit_mappings[key]

        # Priority 2: LLM-generated (if approved)
        if key in self.llm_mappings:
            entry = self.llm_mappings[key]
            if entry.user_approved:
                logger.debug(
                    f"Using LLM-generated mapping for {key} (confidence: {entry.confidence:.2f})",
                )
                return entry.mapping
            else:
                logger.debug(
                    f"LLM mapping exists for {key} but not user-approved, skipping",
                )

        return None

    def save_llm_mapping(
        self,
        source_agent_id: str,
        target_agent_id: str,
        mapping: dict[str, str],
        confidence: float,
        user_approved: bool,
        reasoning: dict[str, str],
    ):
        """
        Save LLM-generated mapping to persistent storage.

        Args:
            source_agent_id: Source agent identifier
            target_agent_id: Target agent identifier
            mapping: Field mappings (target -> source)
            confidence: Overall confidence score
            user_approved: Whether user approved the mapping
            reasoning: Per-field reasoning
        """
        key = f"{source_agent_id}->{target_agent_id}"

        entry = LLMMappingEntry(
            mapping=mapping,
            confidence=confidence,
            user_approved=user_approved,
            reasoning=reasoning,
        )

        self.llm_mappings[key] = entry
        self._save_llm_file()

        approval_status = "approved" if user_approved else "pending approval"
        logger.info(
            f"Saved LLM mapping for {key} (confidence: {confidence:.2f}, {approval_status})",
        )

    def _save_llm_file(self):
        """Persist LLM mappings to JSON file."""
        try:
            # Ensure directory exists
            path = Path(self.llm_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict for JSON serialization
            data = {
                "version": "1.0.0",
                "description": "LLM-generated field mappings with approval history",
                "mappings": {key: entry.model_dump() for key, entry in self.llm_mappings.items()},
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved LLM mappings to {self.llm_path}")
        except Exception as e:
            logger.error(f"Error saving LLM mappings: {e}")

    def update_approval_status(
        self,
        source_agent_id: str,
        target_agent_id: str,
        approved: bool,
    ) -> bool:
        """
        Update approval status of an existing LLM mapping.

        Args:
            source_agent_id: Source agent identifier
            target_agent_id: Target agent identifier
            approved: New approval status

        Returns:
            True if updated, False if mapping not found
        """
        key = f"{source_agent_id}->{target_agent_id}"

        if key not in self.llm_mappings:
            logger.warning(f"No LLM mapping found for {key}")
            return False

        entry = self.llm_mappings[key]
        old_status = entry.user_approved
        entry.user_approved = approved

        self._save_llm_file()

        logger.info(
            f"Updated approval status for {key}: {old_status} -> {approved}",
        )
        return True

    def has_mapping(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> bool:
        """
        Check if any mapping exists for source->target pair.

        Args:
            source_agent_id: Source agent identifier
            target_agent_id: Target agent identifier

        Returns:
            True if explicit or approved LLM mapping exists
        """
        mapping = self.get_mapping(source_agent_id, target_agent_id)
        return mapping is not None

    def get_all_mappings_for_target(
        self,
        target_agent_id: str,
    ) -> dict[str, dict[str, str]]:
        """
        Get all mappings that target a specific agent.

        Args:
            target_agent_id: Target agent identifier

        Returns:
            Dict of source_agent_id -> field_mapping
        """
        result = {}

        # Check explicit mappings
        for key, mapping in self.explicit_mappings.items():
            if key.endswith(f"->{target_agent_id}"):
                source_id = key.split("->")[0]
                result[source_id] = mapping

        # Check LLM mappings
        for key, entry in self.llm_mappings.items():
            if entry.user_approved and key.endswith(f"->{target_agent_id}"):
                source_id = key.split("->")[0]
                result[source_id] = entry.mapping

        return result

    def get_mapping_info(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> dict[str, Any] | None:
        """
        Get detailed information about a mapping including metadata.

        Args:
            source_agent_id: Source agent identifier
            target_agent_id: Target agent identifier

        Returns:
            Dict with mapping info or None
        """
        key = f"{source_agent_id}->{target_agent_id}"

        # Check explicit
        if key in self.explicit_mappings:
            return {
                "type": "explicit",
                "mapping": self.explicit_mappings[key],
                "source": "human-defined",
            }

        # Check LLM
        if key in self.llm_mappings:
            entry = self.llm_mappings[key]
            return {
                "type": "llm-generated",
                "mapping": entry.mapping,
                "confidence": entry.confidence,
                "user_approved": entry.user_approved,
                "generated_at": entry.generated_at,
                "reasoning": entry.reasoning,
            }

        return None

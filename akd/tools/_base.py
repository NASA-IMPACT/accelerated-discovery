from __future__ import annotations

from typing import Optional

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema


class BaseToolConfig(BaseConfig):
    """
    Configuration class for BaseTool.
    This class can be extended to add tool-specific configurations.
    """

    title: Optional[str] = None
    description: Optional[str] = None


class BaseTool[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](AbstractBase):
    config_schema = BaseToolConfig

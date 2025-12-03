from pydantic import Field
from data_types import SearchedSTACData
from typing import List
from copy import deepcopy

from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd._base import InputSchema, OutputSchema

from data_types import STACCollection

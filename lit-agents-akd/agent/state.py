from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

from agent.utils.outline_utils import Editor, Outline
from agent.utils.article_utils import WikiSection
from agent.utils.interview_utils import add_messages, update_editor, update_references


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    sections: List[WikiSection]
    article: str
    
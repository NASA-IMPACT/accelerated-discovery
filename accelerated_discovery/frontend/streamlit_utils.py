from dotenv import load_dotenv
load_dotenv()
from typing import Any
from pydantic import BaseModel


def pydantic_to_markdown(instance: BaseModel) -> str:
    """
    Convert a Pydantic model instance to a Markdown table representation.

    Args:
        instance (BaseModel): An instance of a Pydantic model.

    Returns:
        str: A string containing the Markdown representation of the model.
    """
    # Retrieve the model's field values
    field_values = instance.model_dump()

    # Retrieve the model's annotations to get field types
    field_types = instance.__annotations__

    # Initialize the Markdown table
    markdown = f"#### {instance.__class__.__name__}\n\n"
    markdown += "| Field | Value |\n"
    markdown += "|-------|-------|\n"

    # Populate the table with field details
    for field_name, field_type in field_types.items():
        field_value = field_values.get(field_name, "N/A")
        markdown += f"| {field_name} | {field_value} |\n"

    return markdown

# Example usage
class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True

# user_instance = User(id=1, name="John Doe", email="john.doe@example.com")
# print(pydantic_to_markdown(user_instance))
from pydantic import Field

from akd.agents._base import BaseAgent, LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema

class ScriptBuilderAgentInputSchema(InputSchema):
  """
    The script blueprint to base the story off.  
  """
  narrative_blueprint: str = Field(
    ...,
    description="""
      A blueprint containing the Characters, the inciting incident (from the data), and the resolution.
    """
  )

class ScriptBuilderAgentOutputSchema(OutputSchema):
  """
    The elaborated story draft.
  """
  story_draft: str = Field(
    ...,
    description="""
      The story that is generated from the blueprint.
      Structure the story in markdown format.
    """
  )

class ScriptBuilderAgentConfig(BaseAgentConfig):
  pass

class ScriptBuilderAgent(LiteLLMInstructorBaseAgent[ScriptBuilderAgentInputSchema, ScriptBuilderAgentOutputSchema]):
  """
    You are a Science Communicator (Tone: Nature, Science, or NASA Earth Observatory).
    Your goal is to Write a narrative chronicle of the event. Instruction: "Write a compelling narrative about the event defined in the Blueprint.
 
    There are following guidelines that you need to follow:
    - The 'Character' is the Data: Personify the data slightly.
      - Bad: 'The wind blew hard.'
      - Good: 'As the pressure plummeted to 950mb, the system organized into a tight, kinetic structure.'
    - Visual Language: Use spatial descriptors.
      - 'A tongue of warm water extended across the Pacific...'
      - 'The plume migrated vertically into the stratosphere...'
    - Citations: When you mention a specific number, link it to the source document like this: (Smith et al., 2024).

    Try to follow the Structure given below, however its not absolutely necessary to stick to it if there are not enough resource:
      # The Introduction: The environmental conditions before the anomaly.
      # The Forcing: The moment the variables began to deviate.
      # The Event: The peak intensity. Use the raw data numbers here.
      # The Legacy: The lasting impact on the geography or climate record."
  """
  input_schema = ScriptBuilderAgentInputSchema
  output_schema = ScriptBuilderAgentOutputSchema
  config_schema = ScriptBuilderAgentConfig

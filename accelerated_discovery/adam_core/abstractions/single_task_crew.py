from crewai import Agent, Crew, Process, Task

from dotenv import load_dotenv
load_dotenv()
from crewai import LLM
import os



def get_single_task_crew(tools = [], max_iter : int = 5, output_file = None, output_pydantic=None, verbose = False):
	# async def run_crew_async(crew, inputs):
	# 	result = await crew.kickoff_async(inputs={"dataset": [10, 20, 30, 40, 50]})
	# 	print("Crew Result:", result)
	# 	return result
	llm = LLM(
			model=os.getenv("MODEL"),
			api_key=os.getenv("WATSONX_API_KEY"),
			base_url=os.getenv("WATSONX_URL"),
			max_tokens = 2048,	
			# additional_params={
			# 	"project_id": os.getenv("WATSONX_PROJECT_ID"),
			# 	# Include other necessary parameters such as deployment space ID if required
			# 	"max_tokens": 2048,	
   			# }
		)
	agent = Agent(
		role="Task Executor",
		goal="You execute tasks following the given instructions using your tools when available",
		backstory="You are always faithful and provide only fact based answers.",
		verbose=verbose,
		tools = tools,
		max_iter = max_iter, 
  		llm = llm
    )
	task=Task(
		description="{task_description}",
		expected_output="{expected_output}", 
		output_file=output_file,
		agent= agent,
  		tools=tools,
    	output_pydantic=output_pydantic)
	crew = Crew(
				agents=[agent],
				tasks=[task],
				process=Process.sequential,
				verbose=verbose,
			)
	return crew

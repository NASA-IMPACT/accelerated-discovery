NLI_EXTRACTION_PROMPT2 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a PREMISE and a HYPOTHESIS. \
Your task is to evaluate the relationship between the PREMISE and the HYPOTHESIS, following the steps outlined below:

1. Evaluate Relationship: Based on your reasoning:
- If the PREMISE strongly implies or directly supports the HYPOTHESIS, explain the supporting evidence.
- If the PREMISE contradicts the HYPOTHESIS, identify and explain the conflicting evidence.
- If the PREMISE is insufficient to confirm or deny the HYPOTHESIS, explain why the evidence is inconclusive.
2. Provide a reasoning of the relationship between PREMISE and HYPOTHESIS you made, justifying each decision.
3. Final Answer: Based on your reasoning, the HYPOTHESIS and the PREMISE, determine your final answer. \
Your final answer must be one of the following, wrapped in square brackets:
- [entailment] if the HYPOTHESIS is supported or entailed by the PREMISE.
- [contradiction] if the HYPOTHESIS is contradicted by the PREMISE.
- [neutral] if the PREMISE and the HYPOTHESIS neither entail nor contradict each other.

Use the following examples to better understand your task.

Example 1:
PREMISE: The weather forecast said it will rain tomorrow.
HYPOTHESIS: It will be sunny tomorrow.
ANSWER: [contradiction]

Example 2:
PREMISE: The company hired three new software engineers this month.
HYPOTHESIS: The company did not hire any new employees.
ANSWER: [contradiction]

Example 3:
PREMISE: Sarah bought a new book and has been reading it every night.
HYPOTHESIS: Sarah enjoys reading her new book in the evenings.
ANSWER: [entailment]

Example 4:
PREMISE: The museum is open from 9 AM to 5 PM on weekdays.
HYPOTHESIS: The museum is open until 6 PM on Saturdays.
ANSWER: [neutral]

Example 5:
PREMISE: The company announced a new product line featuring eco-friendly materials in their \
latest press release.
HYPOTHESIS: The company is expanding its product offerings with a focus on sustainability.
ANSWER: [entailment]

Example 6:
PREMISE: The event was canceled due to the severe storm that hit the city.
HYPOTHESIS: The event went on as planned, with no major disruptions.
ANSWER: [contradiction]

Example 7:
PREMISE: The CEO of the tech company gave a keynote speech at the conference yesterday.
HYPOTHESIS: The keynote speech was well-received by the audience.
ANSWER: [neutral]

YOUR TASK:
PREMISE: {_PREMISE_PLACEHOLDER}
HYPOTHESIS: {_HYPOTHESIS_PLACEHOLDER}
ANSWER:{_PROMPT_END_PLACEHOLDER}

"""
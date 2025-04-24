# Test the new RITS service

import os
from litellm import completion, batch_completion
from dotenv import dotenv_values, load_dotenv
from fm_factual.utils import RITS_MODELS
from tqdm import tqdm

INSTRUCTIONS = """\
Instructions:
1. You are given a statement and a context that the statement belongs to. Your task is to modify the \
statement so that any pronouns or anaphora (words like "it," "they," "this") are replaced with the noun \
or proper noun that they refer to, such that the sentence remains clear without referring to the \
original context.
2. Return only the revised, standalone version of the statement without adding any information that is not \
already contained within the original statement.
3. If the statement requires no changes, return the original statement as-is without any explanation.  
4. The statement that you return must start with #### and finish with #### as follows: ####<statement>####
5. Do not include any explanation or any additional formatting including any lead-in or sign-off text.
6. Learn from the provided examples below and use that knowledge to amend the last example yourself.

Example 1:
Context: John went to the store.
Statement: He bought some apples.
Standalone: John bought some apples.

Example 2:
Context: The presentation covered various aspects of climate change, including sea level rise.
Statement: This was a key part of the discussion.
Standalone: ####Sea level rise was a key part of the discussion.####

Example 3:
Context: Maria Sanchez is a renowned marine biologist known for her groundbreaking research on coral reef ecosystems. \
Her work has contributed to the preservation of many endangered coral species, and she is often invited to speak at \
international conferences on environmental conservation.
Statement: She presented her findings at the conference last year.
Standalone: ####Maria Sanchez presented her findings at the conference last year.####

Example 4:
Context: Nathan Carter is a best-selling science fiction author famous for his dystopian novels that explore the \
intersection of technology and society. His latest book, The Edge of Something, received widespread critical acclaim \
for its imaginative world-building and its poignant commentary on artificial cacti.
Statement: It was praised for its thought-provoking themes.
Standalone: ####The Edge of Tomorrow was praised for its thought-provoking themes.####

Now perform the task for the following example:
Context: {}
Statement: {}
Standalone:        
"""

if __name__ == "__main__":

    load_dotenv(override=True)
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    model = "granite-3.0-8b-instruct"

    model_id = RITS_MODELS[model]["model_id"]
    api_base = RITS_MODELS[model]["api_base"]
    template = RITS_MODELS[model]["prompt_template"]
    
    context = "Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry."
    atoms = [
        "He has appeared in numerous films.",
        "He has appeared in numerous television shows.",
        "He has appeared in numerous theater productions.",
        "His career began in the late 1970s."
    ]
    
    prompts = [template.format(INSTRUCTIONS.format(context, atom)) for atom in atoms]
    messages = [[dict(role="user", content=prompt)] for prompt in prompts]
    results = []
    for idx, response in tqdm(
        enumerate(
            batch_completion(
                model=model_id,
                api_base=api_base,
                messages=messages,
                logprobs=True,
                api_key=RITS_API_KEY,
                extra_headers={    
                    "RITS_API_KEY": RITS_API_KEY
                }
            )
        ),
        total=len(prompts),
        desc="Predictions",
        unit="prompts",
        ):
            results.append(response.choices[0])

    print(f"Results: {len(results)}")
    for result in results:
        print(result.message.content)
        print(result.logprobs['content'])
    
    # response = completion(
    #     model=model_id, 
    #     api_base=api_base, 
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": content,
    #         }
    #     ],
    #     logprobs=True,
    #     api_key=RITS_API_KEY,
    #     extra_headers={
    #         "RITS_API_KEY": RITS_API_KEY,
    #     },
    # )    
    # print(response)
    # print(response.choices[0].message.content)
    #print(response.choices[0].logprobs['content'])
    print("Done.")
    
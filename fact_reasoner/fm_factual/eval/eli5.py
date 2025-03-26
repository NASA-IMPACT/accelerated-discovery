# Read the Human annotations from the VeriScore paper.

import os
import json
import litellm

from typing import List
from tqdm import tqdm
from dotenv import load_dotenv

from fm_factual.fact_utils import Atom, Context, build_atoms
from fm_factual.atom_extractor import AtomExtractor
from fm_factual.atom_reviser import AtomReviser
from fm_factual.utils import RITS_MODELS, DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END
from fm_factual.context_retriever import fetch_text_from_link, make_uniform

GEN_PROMPT_TEMPLATE = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a QUESTION. \
Your task is to generate a RESPONSE spanning at most two paragraphs. The continuation should be objective and factual.

QUESTION: {_QUESTION_PLACEHOLDER}
RESPONSE:{_PROMPT_END_PLACEHOLDER}
"""

def generate_model_responses(model: str, questions: List[str]) -> List[str]:
    """
    Generate model responses for the input questions.
    """

    rits_model_info = RITS_MODELS[model]
    prompt_template = rits_model_info.get("prompt_template", None)
    max_new_tokens = rits_model_info.get("max_new_tokens", None)
    api_base = rits_model_info.get("api_base", None)
    model_id = rits_model_info.get("model_id", None)
    prompt_begin = rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
    prompt_end = rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
    use_short_prompt = True if max_new_tokens <= 4096 else False

    assert prompt_template is not None \
        and max_new_tokens is not None \
        and api_base is not None \
        and model_id is not None
    
    load_dotenv(override=True)
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    print(f"Using LLM on RITS: {model_id}")
    print(f"Using short prompt: {use_short_prompt}")

    # Format the prompts
    print(f"Formatting the prompts ...")
    prompts = []
    for q in questions:
        prompt = GEN_PROMPT_TEMPLATE.format(
            _QUESTION_PLACEHOLDER=q,
            _PROMPT_BEGIN_PLACEHOLDER=prompt_begin,
            _PROMPT_END_PLACEHOLDER=prompt_end
        )
        prompts.append(prompt)

    # Prepare the LLM call
    results = []
    messages = [[dict(role="user", content=prompt)] for prompt in prompts]
    for _, response in tqdm(
        enumerate(
            litellm.batch_completion(
                model=model_id,
                api_base=api_base,
                messages=messages,
                api_key=RITS_API_KEY,
                extra_headers={
                    "RITS_API_KEY": RITS_API_KEY
                }
            )
        ),
        total=len(messages),
        desc="Generations",
        unit="prompts",
        ):
            results.append(response.choices[0].message.content)

    return results

def read_data(filename: str):
    questions = []
    print(f"Processing dataset: {filename}")

    with open(filename) as f:
        lines = f.read().splitlines()
        f.close()

    print(f"Found {len(lines)} data elements")
    print(f"Collecting the questions ...")
    questions = []
    human_ans = []
    URLs = []
    for line in lines:
        elem_dict = json.loads(line)
        questions.append(elem_dict["question"])
        human_ans.append(elem_dict["details"]["human_ans"])
        URLs.append(elem_dict["details"]["URL"])

    return questions, human_ans, URLs

def generate_data(model: str, questions: List[str]):
    """
    Generate responses.
    """
    responses = generate_model_responses(model, questions)
    return responses

if __name__ == "__main__":

    model = "llama-3.3-70b-instruct"
    filename = "/home/radu/git/fm-factual/data/data_eli5.jsonl"
    questions, human_ans, URLs = read_data(filename)
    responses = generate_data(model=model, questions=questions)

    output_filename = "/home/radu/git/fm-factual/data/eli5-unlabeled.jsonl"

    # Create the atom extractor, atom reviser and context retriever
    atom_extractor = AtomExtractor(model=model, prompt_version="v2")
    atom_reviser = AtomReviser(model=model, prompt_version="v1")

    output_data = []
    for i, response in enumerate(responses):
        input = questions[i]
        output = response
        ans = human_ans[i]
        url = URLs[i]

        print(f"Processing response: {i+1}/{len(responses)}")

        # Extract atoms
        print(f"Extracting the atoms ...")
        atoms = build_atoms(response, atom_extractor)

        # Revise atoms
        print(f"Decontextualize the atoms ...")
        atom_ids = [aid for aid in sorted(atoms.keys())]
        old_atoms = [atoms[aid].get_text() for aid in atom_ids]
        result = atom_reviser.run(old_atoms, response)
        for i, aid in enumerate(atom_ids):
            res_dict = result[i]
            atoms[aid].set_text(res_dict["revised_atom"])
            print(atoms[aid])

        # Save the new data point into a common format
        data = {}
        data["input"] = input
        data["output"] = output
        data["human_ans"] = ans
        data["url"] = url
        data["atoms"] = []
        for aid, atom in atoms.items():
            data_elem = {}
            data_elem["id"] = aid
            data_elem["text"] = atom.get_text()
            data_elem["original"] = atom.get_original()
            data_elem["label"] = atom.get_label()
            data["atoms"].append(data_elem)
        
        output_data.append(data)

    print(f"Writing {len(output_data)} output data elements")
    with open(output_filename, "w") as f:
        for data_dict in output_data:
            f.write(f"{json.dumps(data_dict)}\n")

    print("Done.")

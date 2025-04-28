# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import torch
import pandas as pd
import litellm

from typing import List
from tqdm import tqdm
from dotenv import load_dotenv

from fm_factual.fact_utils import Atom, Context, build_atoms, build_contexts
from fm_factual.atom_extractor import AtomExtractor
from fm_factual.atom_reviser import AtomReviser
from fm_factual.context_retriever import ContextRetriever
from fm_factual.utils import RITS_MODELS, DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_PROMPT_TEMPLATE = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a QUESTION. \
Your task is to generate a RESPONSE spanning at most two paragraphs.

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

class DatasetProcessor:
    """
    Preprocessor for a given dataset. It handles atom decontextualization and
    context retrieval from wikipedia, google, chromadb.
    """

    def __init__(
            self, 
            input_file: str,
            output_file: str,
            dataset_name: str,
            model: str = "llama-3.1-70b-instruct", 
            service_type: str = "google",
            top_k: int = 3,
            cache_dir: str = None,
            is_annotated: bool = False,
            fetch_text: bool = False
    ):
        """
        Initialize the dataset processor.

        Args:
            input_file: str
                The input file (a .jsonl file).
            output_file: str
                The output file (a .jsonl file).
            dataset_name: str
                Name of the dataset.
            model: str
                Model name or path (e.g., meta-llama/llama-3-70b-instruct).
            service_type: str
                Type of the retrieval service (chromadb or langchain).
            top_k: int
                Top k retrieved passages / contexts
            cache_dir: str
                Path to the cache directory (used for google search results)
            is_annotated: bool
                Flag indicating an annotated dataset (ground truth)
            fetch_text: bool
                Flag indicating if text is fetched from a link (google link)
        """

        self.input_file = input_file
        self.output_file = output_file
        self.dataset_name = dataset_name
        self.model = model
        self.service_type = service_type
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.is_annotated = is_annotated
        self.fetch_text = fetch_text

        # Create the atom extractor
        self.atom_extractor = AtomExtractor(model=self.model, prompt_version="v2")
        self.atom_reviser = AtomReviser(model=self.model, prompt_version="v1")

        # Create the context retriever
        self.context_retriever = ContextRetriever(
            db_remote=False,
            service_type=self.service_type,
            top_k=self.top_k,
            cache_dir=self.cache_dir,
            fetch_text=self.fetch_text
        )

    def _process_bio_labeled(self):
        """
        This is the FactScore Biographies dataset with human labeled atomic facts.
        """
   
        filename = self.input_file # The input file
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            lines = f.read().splitlines()
            df_inter = pd.DataFrame(lines)
            df_inter.columns = ['json_element']
            df_inter['json_element'].apply(json.loads)
            df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
            dataset = df.to_dict('records')
        f.close()

        print(f"[Preprocessor] Loading data from: {filename}")
        print(f"[Preprocessor] Found {len(dataset)} elements")
        print(f"Service type: {self.service_type}")
        print(f"top k: {self.top_k}")

        print(f"[FactVerifier] Reading previous results from: {self.output_file}")
        processed_data = []
        if os.path.isfile(self.output_file):
            with open(self.output_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    processed_data.append(json.loads(line))

        print(f"Found {len(processed_data)} existing processed data points.")
        output_data = []
        num_elem = 0
        for elem_dict in dataset: # for each bio
            num_elem += 1

            # Check if current data has been processed already
            processed = False
            for proc_data in processed_data:
                if proc_data["input"] == elem_dict["input"]:
                    processed = True
                    break
            if processed:
                topic = elem_dict["topic"]
                print(f"Found already processed element: {topic}")
                continue

            response = elem_dict["output"]
            topic = elem_dict["topic"]
            annotations = elem_dict["annotations"]
            if isinstance(annotations, list) == False:
                continue # invalid annotation (skip it)
            
            print(f"Processing elem {num_elem}/{len(dataset)}: {topic}")
            
            # Retrieve the atomic facts
            atoms = {}
            i = 0
            for ann in elem_dict["annotations"]: # get the annotated atomic facts
                if ann["is-relevant"] == True:
                    for fact in ann["human-atomic-facts"]:
                        if fact["label"] in ["S", "NS"]:
                            aid = f"a{i}"
                            atoms[aid] = Atom(
                                id=aid, 
                                text=fact["text"], 
                                label=fact["label"]
                            )
                            i += 1

            print(f"Found {len(atoms)} labeled atomic facts")

            # Decontextualize the atoms
            print(f"Decontextualize the atoms ...")
            atom_ids = [aid for aid in sorted(atoms.keys())]
            old_atoms = [atoms[aid].get_text() for aid in atom_ids]
            result = self.atom_reviser.run(old_atoms, response)
            for i, aid in enumerate(atom_ids):
                res_dict = result[i]
                atoms[aid].set_text(res_dict["revised_atom"])
                print(atoms[aid])

            # Retrieve contexts for each atom
            print(f"Building the contexts ...")
            contexts = build_contexts(
                atoms=atoms,
                retriever=self.context_retriever,
            )
            print(f"Retrieved {len(contexts)} contexts with the {self.service_type} context retriever")

            # Save the new data point into a common format
            data = {}
            data["input"] = elem_dict["input"]
            data["output"] = elem_dict["output"]
            data["topic"] = elem_dict["topic"]
            data["cat"] = elem_dict["cat"]
            data["atoms"] = []
            for aid, atom in atoms.items():
                data_elem = {}
                data_elem["id"] = aid
                data_elem["text"] = atom.get_text()
                data_elem["original"] = atom.get_original()
                data_elem["label"] = atom.get_label()
                data_elem["contexts"] = [c.get_id() for c in atom.get_contexts()]
                data["atoms"].append(data_elem)
            data["contexts"] = []
            for cid, context in contexts.items():
                data_elem = {}
                data_elem["id"] = cid
                data_elem["title"] = context.get_title()
                data_elem["text"] = context.get_text()
                data_elem["link"] = context.get_link()
                data_elem["snippet"] = context.get_snippet()
                data["contexts"].append(data_elem)
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

        print("Done.")

    def _process_bio_unlabeled(self):
        """
        This is the FactScore Biographies with unlabeled atomic facts.
        """

    def _process_human_labeled(self):
        """
        This is the VeriScore human annotated claims with search results. There
        are 330 randomly sampled claims (decontextualized) with human labels.
        Each claim has up to 10 google search results.
        """

        filename = self.input_file # The input file
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            dataset = json.load(f)
        f.close()

        print(f"[Preprocessor] Loading data from: {filename}")
        print(f"[Preprocessor] Found {len(dataset)} elements")
        print(f"Service type: {self.service_type}")
        print(f"Ignoring top k: {self.top_k}")

        output_data = []
        for elem_dict in dataset:
            claim = elem_dict["claim"]
            search_results = elem_dict["search_results"]
            human_label = elem_dict["human_label"]

            label = "S" if "supported" in human_label.lower() else "NS"

            # Save the new data point into a common format
            data = {}
            data["input"] = "Unknown"
            data["output"] = "Unknown"
            data["topic"] = "Unknown"
            data["cat"] = "Unknown"
            data["atoms"] = []

            data_elem = {}
            data_elem["id"] = "a0"
            data_elem["text"] = claim
            data_elem["original"] = claim
            data_elem["label"] = label
            data_elem["contexts"] = [f"c_a0_{j}" for j in range(len(search_results))]
            data["atoms"].append(data_elem)

            data["contexts"] = []
            for j, result_dict in enumerate(search_results):
                data_elem = {}
                data_elem["id"] = f"c_a0_{j}"
                data_elem["title"] = result_dict["title"]
                data_elem["text"] = result_dict["text"]
                data_elem["link"] = result_dict["link"]
                data_elem["snippet"] = result_dict["snippet"]
                data["contexts"].append(data_elem)
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

        print("Done.")

    def _process_eli5_unlabeled(self):
        """
        Preprocessing the ELI5 unlabeled dataset. Requires model response 
        generation, atom extractor, atom reviser, and context retriever.

        The input file contains the generated response as well as the atoms.
        """
        
        filename = self.input_file # The ELI5 jsonl file containing the dataset
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            lines = f.read().splitlines()
            f.close()

        print(f"Found {len(lines)} data elements")

        output_data = []
        for i, line in enumerate(lines):
            
            # Get the input data element (a dict)
            input_elem = json.loads(line)
            lst_atoms = input_elem["atoms"]
            print(f"Processing input {i+1}/{len(lines)} with {len(lst_atoms)} atoms")
            
            atoms = {}
            print(f"Building the atoms ...")
            for a_dict in lst_atoms:
                aid = a_dict["id"]
                orig = a_dict["original"]
                text = a_dict["text"]
                a = Atom(id=aid, text=text)
                a.set_original(orig)
                atoms[aid] = a

            # Retrieve contexts for each atom
            print(f"Building the contexts ...")
            contexts = build_contexts(
                atoms=atoms,
                retriever=self.context_retriever,
            )
            print(f"Retrieved {len(contexts)} contexts with the {self.service_type} context retriever")

            # Save the new data point into a common format
            data = {}
            data["input"] = input_elem["input"]
            data["output"] = input_elem["output"]
            data["topic"] = input_elem["human_ans"]
            data["cat"] = [input_elem["url"]]
            data["atoms"] = []
            for aid, atom in atoms.items():
                data_elem = {}
                data_elem["id"] = aid
                data_elem["text"] = atom.get_text()
                data_elem["original"] = atom.get_original()
                data_elem["label"] = atom.get_label()
                data_elem["contexts"] = [c.get_id() for c in atom.get_contexts()]
                data["atoms"].append(data_elem)
            data["contexts"] = []
            for cid, context in contexts.items():
                data_elem = {}
                data_elem["id"] = cid
                data_elem["title"] = context.get_title()
                data_elem["text"] = context.get_text()
                data_elem["link"] = context.get_link()
                data_elem["snippet"] = context.get_snippet()
                data["contexts"].append(data_elem)
            
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

    def _process_askhist_unlabeled(self):
        """
        Preprocessing the AskHistory unlabeled dataset. Requires model response
        generation, atom extractor, atom reviser, and context retriever.
        """
        filename = self.input_file # The AskHist jsonl file containing the dataset
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            lines = f.read().splitlines()
            f.close()

        print(f"Found {len(lines)} data elements")

        output_data = []
        for i, line in enumerate(lines):
            
            # Get the input data element (a dict)
            input_elem = json.loads(line)
            lst_atoms = input_elem["atoms"]
            print(f"Processing input {i+1}/{len(lines)} with {len(lst_atoms)} atoms")
            
            atoms = {}
            print(f"Building the atoms ...")
            for a_dict in lst_atoms:
                aid = a_dict["id"]
                orig = a_dict["original"]
                text = a_dict["text"]
                a = Atom(id=aid, text=text)
                a.set_original(orig)
                atoms[aid] = a

            # Retrieve contexts for each atom
            print(f"Building the contexts ...")
            contexts = build_contexts(
                atoms=atoms,
                retriever=self.context_retriever,
            )
            print(f"Retrieved {len(contexts)} contexts with the {self.service_type} context retriever")

            # Save the new data point into a common format
            data = {}
            data["input"] = input_elem["input"]
            data["output"] = input_elem["output"]
            data["topic"] = input_elem["human_ans"]
            data["cat"] = [input_elem["url"]]
            data["atoms"] = []
            for aid, atom in atoms.items():
                data_elem = {}
                data_elem["id"] = aid
                data_elem["text"] = atom.get_text()
                data_elem["original"] = atom.get_original()
                data_elem["label"] = atom.get_label()
                data_elem["contexts"] = [c.get_id() for c in atom.get_contexts()]
                data["atoms"].append(data_elem)
            data["contexts"] = []
            for cid, context in contexts.items():
                data_elem = {}
                data_elem["id"] = cid
                data_elem["title"] = context.get_title()
                data_elem["text"] = context.get_text()
                data_elem["link"] = context.get_link()
                data_elem["snippet"] = context.get_snippet()
                data["contexts"].append(data_elem)
            
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

    def _process_books_unlabeled(self):
        """
        Preprocessing the Books unlabeled dataset. Requires model response
        generation, atom extractor, atom reviser and context retriever.
        """
        filename = self.input_file # The ELI5 jsonl file containing the dataset
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            lines = f.read().splitlines()
            f.close()

        print(f"Found {len(lines)} data elements")

        output_data = []
        for i, line in enumerate(lines):
            
            # Get the input data element (a dict)
            input_elem = json.loads(line)
            lst_atoms = input_elem["atoms"]
            print(f"Processing input {i+1}/{len(lines)} with {len(lst_atoms)} atoms")
            
            atoms = {}
            print(f"Building the atoms ...")
            for a_dict in lst_atoms:
                aid = a_dict["id"]
                orig = a_dict["original"]
                text = a_dict["text"]
                a = Atom(id=aid, text=text)
                a.set_original(orig)
                atoms[aid] = a

            # Retrieve contexts for each atom
            print(f"Building the contexts ...")
            contexts = build_contexts(
                atoms=atoms,
                retriever=self.context_retriever,
            )
            print(f"Retrieved {len(contexts)} contexts with the {self.service_type} context retriever")

            # Save the new data point into a common format
            data = {}
            data["input"] = input_elem["input"]
            data["output"] = input_elem["output"]
            data["topic"] = input_elem.get("topic", "Unknown")
            data["cat"] = input_elem.get("cat", "Uknown")
            data["atoms"] = []
            for aid, atom in atoms.items():
                data_elem = {}
                data_elem["id"] = aid
                data_elem["text"] = atom.get_text()
                data_elem["original"] = atom.get_original()
                data_elem["label"] = atom.get_label()
                data_elem["contexts"] = [c.get_id() for c in atom.get_contexts()]
                data["atoms"].append(data_elem)
            data["contexts"] = []
            for cid, context in contexts.items():
                data_elem = {}
                data_elem["id"] = cid
                data_elem["title"] = context.get_title()
                data_elem["text"] = context.get_text()
                data_elem["link"] = context.get_link()
                data_elem["snippet"] = context.get_snippet()
                data["contexts"].append(data_elem)
            
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

    def _process_lfobj_unlabeled(self):
        """
        Preprocessing the Long Form objects unlabeled dataset. Requires model response
        generation, atom extractor, atom reviser and context retriever.
        """
        filename = self.input_file # The lfobj jsonl file containing the dataset
        print(f"Processing dataset: {filename}")

        with open(filename) as f:
            lines = f.read().splitlines()
            f.close()

        print(f"Found {len(lines)} data elements")

        output_data = []
        for i, line in enumerate(lines):
            
            # Get the input data element (a dict)
            input_elem = json.loads(line)
            lst_atoms = input_elem["atoms"]
            print(f"Processing input {i+1}/{len(lines)} with {len(lst_atoms)} atoms")
            
            atoms = {}
            print(f"Building the atoms ...")
            for a_dict in lst_atoms:
                aid = a_dict["id"]
                orig = a_dict["original"]
                text = a_dict["text"]
                a = Atom(id=aid, text=text)
                a.set_original(orig)
                atoms[aid] = a

            # Retrieve contexts for each atom
            print(f"Building the contexts ...")
            contexts = build_contexts(
                atoms=atoms,
                retriever=self.context_retriever,
            )
            print(f"Retrieved {len(contexts)} contexts with the {self.service_type} context retriever")

            # Save the new data point into a common format
            data = {}
            data["input"] = input_elem["input"]
            data["output"] = input_elem["output"]
            data["topic"] = input_elem.get("topic", None)
            data["cat"] = input_elem.get("cat", None)
            data["atoms"] = []
            for aid, atom in atoms.items():
                data_elem = {}
                data_elem["id"] = aid
                data_elem["text"] = atom.get_text()
                data_elem["original"] = atom.get_original()
                data_elem["label"] = atom.get_label()
                data_elem["contexts"] = [c.get_id() for c in atom.get_contexts()]
                data["atoms"].append(data_elem)
            data["contexts"] = []
            for cid, context in contexts.items():
                data_elem = {}
                data_elem["id"] = cid
                data_elem["title"] = context.get_title()
                data_elem["text"] = context.get_text()
                data_elem["link"] = context.get_link()
                data_elem["snippet"] = context.get_snippet()
                data["contexts"].append(data_elem)
            
            output_data.append(data)

        print(f"Writing {len(output_data)} output data elements")
        with open(self.output_file, "w") as f:
            for data_dict in output_data:
                f.write(f"{json.dumps(data_dict)}\n")

    def run(self):
        """
        Run the pre-processor to decontextualize the atoms and retrieve the
        corresponding contexts.
        """
        if self.dataset_name == "bio_labeled":
            self._process_bio_labeled()
        elif self.dataset_name == "bio_unlabeled":
            self._process_bio_unlabeled()
        elif self.dataset_name == "human_labeled":
            self._process_human_labeled()
        elif self.dataset_name == "eli5_unlabeled":
            self._process_eli5_unlabeled()
        elif self.dataset_name == "askhist_unlabeled":
            self._process_askhist_unlabeled()
        elif self.dataset_name == "books_unlabeled":
            self._process_books_unlabeled()
        elif self.dataset_name == "lfobj_unlabeled":
            self._process_lfobj_unlabeled()
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported yet.")
        


if __name__ == "__main__":

    # Bio labeled (gold)
    input_file = "/home/radu/git/fm-factual/data/bio-labeled-ChatGPT.jsonl"
    output_file = "/home/radu/git/fm-factual/data/bio-labeled-ChatGPT-langchain-doc2.jsonl"
    dataset_name = "bio_labeled"

    # Human labeled (gold)
    # input_file = "/home/radu/git/fm-factual/data/human-labeled.json"
    # output_file = "/home/radu/git/fm-factual/data/claims-labeled-google-doc.jsonl"
    # dataset_name = "human_labeled"

    # ELI5 unlabeled
    # input_file = "/home/radu/git/fm-factual/data/eli5-unlabeled.jsonl"
    # output_file = "/home/radu/git/fm-factual/data/eli5-unlabeled-langchain-doc.jsonl"
    # dataset_name = "eli5_unlabeled"

    # AskHist unlabeled
    # input_file = "/home/radu/git/fm-factual/data/askhist-unlabeled.jsonl"
    # output_file = "/home/radu/git/fm-factual/data/askhist-unlabeled-google-doc.jsonl"
    # dataset_name = "askhist_unlabeled"

    # Books unlabeled
    # input_file = "/home/radu/git/fm-factual/data/books-unlabeled.jsonl"
    # output_file = "/home/radu/git/fm-factual/data/books-unlabeled-langchain-doc.jsonl"
    # dataset_name = "books_unlabeled"

    # LFobjects unlabeled
    # input_file = "/home/radu/git/fm-factual/data/lfobj-unlabeled.jsonl"
    # output_file = "/home/radu/git/fm-factual/data/lfobj-unlabeled-google-doc.jsonl"
    # dataset_name = "lfobj_unlabeled"

    # Create a dataset processor
    proc = DatasetProcessor(
        input_file=input_file,
        output_file=output_file,
        dataset_name=dataset_name,
        service_type="langchain",
        model="llama-3.1-70b-instruct",
        top_k=3,
        cache_dir="/home/radu/data/cache",
        fetch_text=True
    )

    proc.run()
    print("Done.")


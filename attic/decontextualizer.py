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

# Atomic fact decontextualization using LLMs

import os

from dotenv import dotenv_values, load_dotenv
from tqdm import tqdm
from copy import deepcopy
from litellm import completion, batch_completion

from fm_factual.utils import RITS_MODELS

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

class Decontextualizer:
    """
    Class for  decontextualizing atoms or context.

    Parameters:
    model: LLM used for decontextualization
    data : various types (e.g., dict, list, etc.)
        The input data to be processed or stored. If datatype not set, must be list
        of (atom, context) tuples where 'atom' is to be decontextualized using 'context'
    datatype : str, optional
        Specifies the type of data for processing. If None, data is stored as-is.
    """
    def __init__(
            self,
            model: str = "llama-3.1-70b-instruct",
            data = None,
            datatype: str = None            
    ):
        self.input_data = deepcopy(data)
        self.datatype = datatype
        self.model = model
        self.rits_model_info = RITS_MODELS[model]

        self.prompt_template = self.rits_model_info.get("prompt_template", None)
        self.max_new_tokens = self.rits_model_info.get("max_new_tokens", None)
        self.api_base = self.rits_model_info.get("api_base", None)
        self.model_id = self.rits_model_info.get("model_id", None)

        assert self.prompt_template is not None \
            and self.max_new_tokens is not None \
            and self.api_base is not None \
            and self.model_id is not None
        
        self.parameters = dict(
            max_new_tokens=1000,
            min_new_tokens=1,
            decoding_method="greedy",
        )

        load_dotenv(override=True)
        self.RITS_API_KEY = os.getenv("RITS_API_KEY")

        if datatype:
            self._process_data(data, datatype)
        else:
            # If no processing is required, store the data directly
            self.data = data
        
        self._create_prompts()
    
    def _process_data(self, data, datatype):
        """
        Processes the input data based on the specified datatype (bios, contexts, etc).

        Parameters:
        data : dict
            The input data to be processed.
        datatype : str
            The type of data to be processed ('labeled_bios', 'unlabeled_bios', 'contexts').

        Returns:
        Processed data based on the datatype.
        """
        if datatype == 'labeled_bios':
            return self._process_labeled_bios(data)
        elif datatype == 'unlabeled_bios':
            return self._process_unlabeled_bios(data)
        elif datatype == 'contexts':
            return self._process_contexts(data)
        else:
            raise ValueError(f"Unknown datatype: {datatype}")

    def _process_labeled_bios(self, data):
        self.data = []
        for instance in data:
            if instance['annotations'] is None: continue
            for annotation in instance['annotations']:
                if annotation["human-atomic-facts"] is None: continue
                for fact in annotation["human-atomic-facts"]:
                    if fact['label'] not in ["S","NS"]: continue
                    self.data.append((fact['text'], instance['output']))

    def _process_unlabeled_bios(self, data):
        self.data = []
        for instance in data:
            if not isinstance(instance['facts'],list):continue
            for fact in instance['facts']:
                self.data.append((fact, instance['output']))

    def _process_contexts(self, data):
        self.data = deepcopy(data)

    def _create_prompts(self):
        self.prompts = []
        for atom, context in self.data:
            prompt = INSTRUCTIONS.format(context, atom)

            try:
                prompt = self.prompt_template.format(prompt)
            except KeyError:
                None

            # Truncate the prompt if too long
            if len(prompt) > self.max_new_tokens:
                prompt = prompt[:self.max_new_tokens]
            self.prompts.append(prompt)
        
    def __call__(self):

        results = []
        messages = [[dict(role="user", content=prompt)] for prompt in self.prompts]

        for idx, response in tqdm(
            enumerate(
                batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=messages,
                    api_key=self.RITS_API_KEY,
                    extra_headers={
                        "RITS_API_KEY": self.RITS_API_KEY,
                    }
                )
            ),
            total=len(messages),
            desc="Decontextualization",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        if self.datatype in ['labeled_bios','unlabeled_bios', 'contexts']:
            self._process_results(results)

        return self.input_data
    
    def _process_results(self, results):
        if self.datatype == 'labeled_bios':
            for instance in self.input_data:
                if instance['annotations'] is None: continue
                for annotation in instance['annotations']:
                    if annotation["human-atomic-facts"] is None: continue
                    for fact in annotation["human-atomic-facts"]:
                        if fact['label'] not in ["S","NS"]: continue
                        result = results.pop(0).strip()
                        if result.count('####') != 2:
                            continue 
                        else:
                            fact['text'] = result.split('####')[1]
        elif self.datatype == 'unlabeled_bios':
            for instance in self.input_data:
                if not isinstance(instance['facts'],list):continue
                for ii in range(len(instance['facts'])):
                    result = results.pop(0).strip()
                    if result.count('####') != 2:
                        continue 
                    else:
                        instance['facts'][ii] = result.split('####')[1]
        elif self.datatype == "contexts":
            for ii in range(len(self.input_data)):
                result = results.pop(0).strip()
                if result.count('####') != 2:
                    continue
                else:
                    atom = result.split('####')[1]
                    context = self.input_data[ii][1]
                    self.input_data[ii] = (atom, context)

if __name__ == "__main__":
    
    context = "Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry."
    atoms = [
        "He has appeared in numerous films.",
        "He has appeared in numerous television shows.",
        "He has appeared in numerous theater productions.",
        "His career began in the late 1970s."
    ]

    old_atoms = [(atom, context) for atom in atoms]
    model_id = "llama-3.1-70b-instruct"
    print(f"Decontextualizing the atoms using {model_id}")
    decontextualizer = Decontextualizer(
        model=model_id,
        data=old_atoms, 
        datatype="contexts"
    )

    new_atoms = decontextualizer()
    for atom, _ in new_atoms:
        print(atom)
    print("Done.")
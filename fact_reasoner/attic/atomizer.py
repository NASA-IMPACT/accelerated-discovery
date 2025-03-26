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

# Split the input text into smaller atoms i.e., atomic facts, sentences, etc.

import os
import re
import nltk
import numpy as np
import itertools

from tqdm import tqdm
from typing import List
from nltk import tokenize
from dotenv import dotenv_values, load_dotenv
from litellm import completion, batch_completion

from fm_factual.utils import RITS_MODELS

nltk.download('punkt', quiet=True)

MONTHS = [
    m.lower()
    for m in [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
]

#SPACY_MODEL = spacy.load('en_core_web_sm')
NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

ATOMIC_FACT_INSTRUCTION = """\
Instructions:
1. You are given a paragraph. Your task is to break the sentence down into \
a list of atomic statements without adding any new information.
2. An atomic statement is a sentence containing a singular piece of information \
directly extracted from the provided paragraph.
3. Atomic statements may contradict one another.
4. The paragraph may contain information that is factually incorrect. Even in such \
cases, you are not to alter any information contained in the paragraph and must \
produce atomic statements that are completely faithful to the information in the paragraph.
5. Each atomic statement in the outputted list should check a different piece of \
information found explicitly in the paragraph.
6. Each atomic statement is standalone in that any actual nouns or proper nouns \
should be used in place of pronouns or anaphors.
7. Each atomic statement must not include any information beyond what is explicitly \
stated in the provided paragraph.
8. Where possible, avoid paraphrasing and instead try to only use language used in the \
paragraph without introducing new words. 
9. Use the previous examples to learn how to do this.
10. You should only output the atomic statement as a list, with each item starting \
with "- ". Do not include other formatting.
11. Your task is to do this for the last paragraph that is given. 
"""

FEW_SHOTS = """\
Please breakdown the following paragraph into independent statements: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
- Glenn Allen Anzalone was born on June 23, 1955.
- Glenn Allen Anzalone is better known by his stage name Glenn Danzig.
- Glenn Danzig is an American singer, songwriter, musician, and record producer.
- Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig.
- Glenn Danzig owns the Evilive record label.
- Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company.

Please breakdown the following paragraph into independent statements: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
- Luiz Inácio Lula da Silva was born on October 27, 1945.
- Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula.
- Lula is a Brazilian politician.
- Lula is the 39th and current president of Brazil since 2023.
- Lula is a member of the Workers' Party.
- Lula served as the 35th president of Brazil from 2003 to 2010.
- Lula holds the presidency of the G20 since 2023.
- Lula quit school after the second grade to work.
- Lula did not learn to read until he was ten years old.
- As a teenager, Lula worked as a metalworker.
- Lula became a trade unionist.
"""

def detect_initials(text):
    pattern = r'[A-Z]\. ?[A-Z]\.'
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    """Fix sentence splitter issues."""
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split('.') if t.strip()]

        for i, (sent1, sent2) in enumerate(
            zip(curr_sentences, curr_sentences[1:])
        ):
            if sent1.endswith(alpha1 + '.') and sent2.startswith(alpha2 + '.'):
                # merge sentence i and i+1
                curr_sentences = (
                    curr_sentences[:i]
                    + [curr_sentences[i] + ' ' + curr_sentences[i + 1]]
                    + curr_sentences[i + 2 :]
                )
            break

    sentences, combine_with_previous = [], None

    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)

    return sentences


class Atomizer:
    def __init__(
            self,
            granularity: str = "none",
            model: str = "llama-3.1-70b-instruct"
    ):
        """
        Atomizer constructor.

        Args:
            granularity: str
                The granularity level used for decomposing a sequence into atomic
                facts. Allowed values are: [none, fact, sentence, paragraph]
            model_id: str
                The model id (e.g., meta-llama/llama-3-70b-instruct) used for
                extracting atomic facts from the input sequence.    
        """

        self.granularity = granularity
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

    def _split_sentences(
            self, 
            text: str
    ) -> List[str]:
        """
        Decompose the input text into sentences.

        Args:
            text: str
                The input text.

        Returns:
            List[str]
                The list of sentences.
        """

        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences

    def _split_paragraphs(
            self, 
            text:str
    ) -> List[str]:
        """ 
        Decompose the input text into paragraphs. We assume that paragraphs
        are delimited by two new-lines ('\n\n').

        Args:
            text: str
                The input text.

        Returns:
            List[str]
                The list of paragraphs.
        """
        
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def _split_facts(
            self, 
            text: str,
            facts_per_sentence: bool = False
    ) -> List[str]:
        """
        Decompose the input text into atomic facts.

        Args:
            text: str
                The input text.
            facts_per_sentence: bool
                Boolean flag indicating that atomic fact decomposition is done
                per sentence. If false, then the decomposition is per paragraph.

        Returns:
            List[str]
                The list of atomic facts extracted from the input text.                
        """

        if facts_per_sentence is False:
            paragraphs = self._split_paragraphs(text)
            print(f"[Getting atomic facts from {len(paragraphs)} paragraphs]")
            atomic_facts = self._get_atomic_facts_from_paragraphs(paragraphs)
            return list(itertools.chain.from_iterable(atomic_facts))
        else:
            sentences = self._split_sentences(text)
            print(f"[Generating atomic facts from {len(sentences)} sentences]")
            atomic_facts = self._get_atomic_facts_from_sentences(sentences)
            return list(itertools.chain.from_iterable(atomic_facts))
        
    def _get_atomic_facts_from_paragraphs(
            self, 
            paragraphs: List[str]
    ) -> List[str]:
        """
        Get the atomic facts from a list of paragraphs.
        
        Args:
            paragraphs: List[str]
                A list of paragraphs (strings).

        Returns:
            List[str]
                The list of atomic facts
        """

        paragraphs4prompts = []

        for paragraph in paragraphs:

            paragraphs4prompts.append([])
            initials = detect_initials(paragraph)
            curr_sentences = tokenize.sent_tokenize(paragraph)
            curr_sentences_2 = tokenize.sent_tokenize(paragraph)
            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)
        
            # ensure the credability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (
                paragraph,
                curr_sentences,
                curr_sentences_2,
            )
            paragraphs4prompts[-1] += curr_sentences

        prompts = self._create_prompts(paragraphs4prompts)
        atoms = self._get_atomic_facts(prompts) # list of lists of strings

        return atoms

    def _get_atomic_facts_from_sentences(
            self, 
            sentences: List[str]
    ) -> List[str]:
        """
        Get the atomic facts from a list of sentences.
        
        Args:
            sentences: List[str]
                A list of sentences (strings).

        Returns:
            List[str]
                The list of atomic facts corresponding to the input sentences.
        """

        sentences4prompts = []

        for sentence in sentences:
            sentences4prompts.append(sentence)

        prompts = self._create_prompts(sentences4prompts)
        atoms = self._get_atomic_facts(prompts) # list of strings

        return atoms

    def _get_atomic_facts(
            self, 
            prompts: List[str]
    ) -> List[List[str]]:
        """
        Get the atomic facts by quering the language model with the 
        correponding prompts.

        Args:
            prompts: List[str]
                A list of prompts (string) for the language model.
        
        Returns:
            List[List[str]]
                The list of repsonses (atomic facts) returned by the LLM. It is
                a list of lists i.e., List[List[str]]
        """
        
        results = []
        messages = [[dict(role="user", content=prompt)] for prompt in prompts]
        for idx, response in tqdm(
            enumerate(
                batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=messages,
                    api_key=self.RITS_API_KEY,
                    extra_headers={
                        "RITS_API_KEY": self.RITS_API_KEY
                    }
                )
            ),
            total=len(messages),
            desc="Atomizer",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        atoms = self._postprocess_results(results)  

        return atoms

    def _postprocess_results(
            self, 
            results: List[str]
    ) -> List[str]:
        """
        Postprocess the responses returned by the language model.

        Args:
            results: List[str]
                The list of responses returned by the language model.

        Returns:
            List[str]
                A list of postprocessed responses.
        """

        
        atoms = []
        for result in results:
            atoms.append([])
            sentences = result.split('\n')
            for sentence in sentences:
                if sentence[:2] != '- ':continue
                if len(sentence[2:])<5:continue #TODO: include better processing of atoms (e.g., check for verbs etc)
                atoms[-1].append(sentence[2:])

        return atoms

    def _create_prompts(
            self, 
            paragraphs: List[str]
    ) -> List[str]:
        """
        Create the prompts.

        Args:
            paragraphs: List[str]
                The list of paragraphs that need to be processed.

        Returns:
            List[str]
                A list of prompts corresponding to the paragraphs.
        """

        prompts = []
        for paragraph in paragraphs:
            #prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n'+ATOMIC_FACT_INSTRUCTION + '\n\n'
            prompt = ATOMIC_FACT_INSTRUCTION + '\n\n'
            prompt += FEW_SHOTS
            prompt+='\n\n'+'Please breakdown the following paragraph into independent facts:'+'\n\n'
            
            for sentence in paragraph:
                prompt+=sentence+' '
            #prompt+='\n<|start_header_id|>assistant<|end_header_id|>'

            try:
                prompt = self.prompt_template.format(prompt)
            except KeyError:
                None

            prompts.append(prompt)

        return prompts

    def __call__(
            self,
            text: str,
            facts_per_sentence: bool = False
    ) -> List[str]:
        
        """
        Splits the input sequence into a list of atoms. The atoms can be:
        atomic facts, sentences or small paragraphs.

        Returns:
            List of atoms (strings)
        """

        # print(f"[Calling atomizer with granularity: {self.granularity}]")

        if self.granularity == "none":
            return [text]
        elif self.granularity == "sentence":
            return self._split_sentences(text)
        elif self.granularity == "paragraph":
            return self._split_paragraphs(text)
        elif self.granularity == "fact":
            return self._split_facts(text, facts_per_sentence)
        else:
            raise ValueError(
                f"Granularity {self.granularity} is not supported."
            )

def scores(scorer, references: List[str], candidates: List[str]):
    # BERTScore calculation
    P, R, F1 = scorer.score(candidates, references)
    return F1.numpy()


if __name__ == "__main__":
    
    atomizer = Atomizer(
        granularity="fact",
        model="granite-3.0-8b-instruct"
    )

    text = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
        This mission was significant as it marked the third time humans set \
        foot on the lunar surface, with astronauts Alan Shepard and Edgar \
        Mitchell joining Captain Stuart Roosa, who had previously flown on \
        Apollo 13. The mission lasted for approximately 8 days, during which \
        the crew conducted various experiments and collected samples from the \
        lunar surface. Apollo 14 brought back approximately 70 kilograms of \
        lunar material, including rocks, soil, and core samples, which have \
        been invaluable for scientific research ever since."
    atoms = atomizer(text, True)

    print(f"Found {len(atoms)} atomic facts")
    for i, atom in enumerate(atoms):
        print(f"{i}: {atom}")

    print("Done.")

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

# NLI scorer with Prompting (llama-3-70b-instruct)

import os
import numpy as np
import logging
import nltk

from difflib import SequenceMatcher
from operator import itemgetter
from dotenv import dotenv_values, load_dotenv
from litellm import batch_completion
from tqdm import tqdm

from fm_factual.utils import RITS_MODELS, dotdict

NLI_LABELS = ['entailment', 'contradiction', 'neutral']

os.environ["LITELLM_LOG"] = 'ERROR'
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_label_probability(samples: list, labels: list):
    candidates = []
    for label in labels:
        distances = [similarity(label, sample) for sample in samples]
        candidates.append((label, np.average(distances)))
    candidates = sorted(candidates, key=itemgetter(1), reverse=True)
    return candidates[0]

class NLIScorerPrompting:
    def __init__(
            self,
            model: str = "llama-3.1-70b-instruct",
            granularity: str = "sentence",
            scoring_method: str = "logprobs"
    ):
        self.model = model
        self.granularity = granularity
        self.scoring_method = scoring_method
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

    def score(
            self, 
            premises: list,
            hypthotheses: list, 
            temperature: float = 1.0,
            n_trials: int = 10,
            contexts = None
    ):
        assert self.scoring_method in ['temperature', 'logprobs', 'consistency'], f"invlalid scoring method"
        
        self.create_prompts(premises, hypthotheses, contexts)

        if self.scoring_method == 'temperature':
            self.n_trials = n_trials
            self.prompts = [item for sublist in [[prompt]*self.n_trials for prompt in self.prompts] for item in sublist]
            results = []
            messages = [[dict(role="user", content=prompt)] for prompt in self.prompts]
            for idx, response in tqdm(
                enumerate(
                    batch_completion(
                        model=self.model_id,
                        api_base=self.api_base,
                        messages=messages,
                        temperature=temperature,
                        api_key=self.RITS_API_KEY,
                        extra_headers={
                            "RITS_API_KEY": self.RITS_API_KEY
                        }
                    )
                ),
                total=len(messages),
                desc="NLI",
                unit="prompts",
                ):
                    results.append(response.choices[0].message.content.strip().lower())
                
            results_labels = []
            start_ind = 0  
            for ii in range(len(premises)):
                result = results[start_ind:start_ind+self.n_trials]
                start_ind+=self.n_trials

                res = {}
                for label in ['entailment', 'contradiction', 'neutral']:
                    res[label] = result.count(label)
                    
                results_labels.append(res)

            results_labels = [(max(x, key=x.get),x[max(x, key=x.get)]/self.n_trials) for x in results_labels]
        elif self.scoring_method == 'logprobs': # white-box UQ
            results = []
            messages = [[dict(role="user", content=prompt)] for prompt in self.prompts]
            for idx, response in tqdm(
                enumerate(
                    batch_completion(
                        model=self.model_id,
                        api_base=self.api_base,
                        messages=messages,
                        logprobs=True,
                        api_key=self.RITS_API_KEY,
                        extra_headers={
                            "RITS_API_KEY": self.RITS_API_KEY
                        }
                    )
                ),
                total=len(messages),
                desc="NLI",
                unit="prompts",
                ):
                    results.append(response.choices[0])

            results_labels = []
            for result in results:
                label = result.message.content.strip().lower()
                if label not in ['entailment', 'contradiction', 'neutral']:
                    label = 'neutral' #'invalid_label'
                    prob = 1.0
                else:
                    logprob_sum = 0.0
                    generated_tokens = result.logprobs['content'][:-1]
                    for token in generated_tokens: #last token is just <|eot_id|>
                        token = dotdict(token)
                        logprob_sum +=token.logprob

                    prob = np.exp(logprob_sum/len(generated_tokens))

                results_labels.append((label, prob))
        elif self.scoring_method == "consistency": # black-box UQ
            # Consistency-based UQ based on similarity aggregation.
            samples = {}
            temperatures = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
            for t in temperatures:
                self.n_trials = n_trials
                prompts = [item for sublist in [[prompt]*self.n_trials for prompt in self.prompts] for item in sublist]
                results = []

                print(f"Generating {n_trials} samples for temperature {t}")
                messages = [[dict(role="user", content=prompt)] for prompt in prompts]
                for idx, response in tqdm(
                    enumerate(
                        batch_completion(
                            model=self.model_id,
                            api_base=self.api_base,
                            messages=messages,
                            temperature=t,
                            api_key=self.RITS_API_KEY,
                            extra_headers={
                                "RITS_API_KEY": self.RITS_API_KEY
                            }
                        )
                    ),
                    total=len(messages),
                    desc="NLI",
                    unit="prompts",
                    ):
                        results.append(response.choices[0].message.content.strip().lower())
                
                samples[t] = results

            # Compute pairwise similarities between samples and aggregate them
            results_labels = []
            start_ind = 0  
            for ii in range(len(premises)): # for each (premise, hypothesis)
                sampled_labels = []
                for t in temperatures: # collect the samples at different temperatures
                    result = samples[t][start_ind:start_ind+self.n_trials]
                    sampled_labels.extend(result)
                results_labels.append(get_label_probability(sampled_labels, NLI_LABELS))                    
                start_ind += self.n_trials

        return results_labels

    def create_prompts(self, premises, hypthotheses, contexts=None):
        
        if contexts is not None:
            INSTRUCTIONS = """\
Instructions:
1. You are given a premise and a hypothesis and a context. \
Your task is to identify the relationship between them: does the premise entail, contradict, \
or remain neutral toward the hypothesis?
2. Your only output must be one of: (entailment | contradiction | neutral) without any \
lead-in, sign-off, new lines or any other formatting.
3. Do not provide any explanation or rationale to your output.
4. Use the following examples to learn how to do this, and provide your output for the last \
example given.

Premise: Contrary to popular belief, the Great Wall is not visible from space without aid.
Hypothesis: Astronauts have managed to see the wall from Space unaided. 
Context: The Great Wall of China is one of the most famous landmarks in the world. \
It stretches over 13,000 miles and was primarily built during the Ming Dynasty. \
Contrary to popular belief, the Great Wall is not visible from space without aid. \
The primary purpose of the Great Wall was to protect against invasions from nomadic tribes. \
The wall is a UNESCO World Heritage site and attracts millions of tourists each year. \
Astronauts have managed to see the wall from Space unaided. 
Output: Contradiction

Premise: It is estimated that around 20 percent of the world's oxygen is produced by the Amazon.
Hypothesis: However, the Amazon Rainforest produces no significant amount of oxygen as the plants \
consume almost all of it through respiration.
Context: The Amazon Rainforest is often referred to as the lungs of the Earth due to its \
vast capacity to produce oxygen. This immense rainforest spans nine countries in South America. \
It is estimated that around 20 percent of the world's oxygen is produced by the Amazon. However, the \
Amazon Rainforest produces no significant amount of oxygen as the plants consume almost all of it \
through respiration. The biodiversity of the Amazon is unparalleled, hosting millions of species of \
plants and animals.
Output: Contradiction

Premise: It is estimated that around 20 percent of the world's oxygen is produced by the Amazon.
Hypothesis: This immense rainforest spans nine countries in South America.
Context: The Amazon Rainforest is often referred to as the lungs of the Earth due to its \
vast capacity to produce oxygen. This immense rainforest spans nine countries in South America. \
It is estimated that around 20 percent of the world's oxygen is produced by the Amazon. However, the \
Amazon Rainforest produces no significant amount of oxygen as the plants consume almost all of it \
through respiration. The biodiversity of the Amazon is unparalleled, hosting millions of species of \
plants and animals.
Output: Neutral

Premise: It is estimated that around 20 percent of the world's oxygen is produced by the Amazon.
Hypothesis: The Amazon Rainforest is often referred to as the lungs of the Earth due to its \
vast capacity to produce oxygen.
Context: The Amazon Rainforest is often referred to as the lungs of the Earth due to its \
vast capacity to produce oxygen. This immense rainforest spans nine countries in South America. \
It is estimated that around 20 percent of the world's oxygen is produced by the Amazon. However, the \
Amazon Rainforest produces no significant amount of oxygen as the plants consume almost all of it \
through respiration. The biodiversity of the Amazon is unparalleled, hosting millions of species of \
plants and animals.
Output: Entailment

Premise: {}
Hypothesis: {}
Context: {}
Output:
"""
        else:

            INSTRUCTIONS = """\
Instructions:
1. You are given a premise and a hypothesis. Your task is to identify the relationship \
between them: does the premise entail, contradict, or remain neutral toward the hypothesis?
2. Your only output must be one of: (entailment | contradiction | neutral) without any \
lead-in, sign-off, new lines or any other formatting.
3. Do not provide any explanation or rationale to your output.
4. Use the following examples to learn how to do this, and provide your output for the last \
example given.

Premise: The weather forecast said it will rain tomorrow.
Hypothesis: It will be sunny tomorrow.
Output: contradiction

Premise: The company hired three new software engineers this month.
Hypothesis: The company did not hire any new employees.
Output: contradiction

Premise: Sarah bought a new book and has been reading it every night.
Hypothesis: Sarah enjoys reading her new book in the evenings.
Output: entailment

Premise: The museum is open from 9 AM to 5 PM on weekdays.
Hypothesis: The museum is open until 6 PM on Saturdays.
Output: neutral

Premise: The company announced a new product line featuring eco-friendly materials in their \
latest press release.
Hypothesis: The company is expanding its product offerings with a focus on sustainability.
Output: Entailment

Premise: The event was canceled due to the severe storm that hit the city.
Hypothesis: The event went on as planned, with no major disruptions.
Output: Contradiction

Premise: The CEO of the tech company gave a keynote speech at the conference yesterday.
Hypothesis: The keynote speech was well-received by the audience.
Output: Neutral

Premise: {}
Hypothesis: {}
Output:
"""

        self.prompts = []
        for premise, hypothesis in zip(premises,hypthotheses):
            if "granite" in self.model:
                premise = premise[:2000]
                hypothesis = hypothesis[:2000]
            if contexts is not None:
                prompt = INSTRUCTIONS.format(premise, hypothesis, '\n'.join(contexts))
            else:
                prompt = INSTRUCTIONS.format(premise, hypothesis)

            try:
                prompt = self.prompt_template.format(prompt)
            except KeyError:
                None

            # print(f"prompt length: {len(prompt)} chars, {len(nltk.word_tokenize(prompt))} words")

            self.prompts.append(prompt)

if __name__ == '__main__':
    
    import pickle
    with open('examples/premises.pkl','rb') as f:premises = pickle.load(f)
    with open('examples/hypothesis.pkl','rb') as f:hypotheses = pickle.load(f)
    
    scorer = NLIScorerPrompting(model="granite-3.0-8b-instruct", scoring_method="logprobs")
    results_labels = scorer.score(premises[:10],hypotheses[:10], n_trials=5)
    print(results_labels)
    
    # for temperature in [1.2,1.4,1.6,1.8]:
    #     results_labels = scorer.score(premises[:10],hypotheses[:10],temperature=temperature)

    #     print("temperature= ",temperature)
    #     print(results_labels)
    
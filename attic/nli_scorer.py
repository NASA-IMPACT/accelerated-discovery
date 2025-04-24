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


import nltk
import numpy as np 
import torch
import os
import json

from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local
from fm_factual.utils import batcher
from attic.atomizer import Atomizer

# Do it once
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

model_map = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0, "contradiction_idx": 2},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
    # "decomp": 0,
}


def card_to_name(card):
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    if card in card2name:
        return card2name[card]
    return card


def name_to_card(name):
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]


class NLIImager:
    def __init__(
            self, 
            model_name="mnli", 
            granularity="sentence",
            max_doc_sents=100, 
            **kwargs
    ):
        """
        NLIImager constructor

        Args:
            model_name: str
                The NLI model name.
            granularity: str
                The premise/hypothesis granularity level used to detemine entailment, 
                neutrality and contradiction relationships (sentence, paragraph).
            max_doc_sents: int
                The maximum number of sentences allowed in a document.

        """

        assert granularity in ["paragraph", "sentence"], \
            "Unrecognized `granularity` %s" % (granularity)
        assert model_name in model_map.keys(), \
            "Unrecognized model name: `%s`" % (model_name)

        self.model_name = model_name
        self.model_card = name_to_card(model_name)
        self.entailment_idx = model_map[model_name]["entailment_idx"]
        self.contradiction_idx = model_map[model_name]["contradiction_idx"]
        self.neutral_idx = get_neutral_idx(
            ent_idx=self.entailment_idx, 
            con_idx=self.contradiction_idx
        )

        # Determine available device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.granularity = granularity

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500
        self.model = None # Lazy loader
        self.atomizer = Atomizer(granularity=self.granularity)

    def load_nli(self):
        """
        Load the NLI model into memory (on GPU if available).
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval()
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()

    def build_chunk_dataset(
            self, 
            premise: str, 
            hypothesis:str, 
            pair_idx=None
    ):
        """
        Build the dataset used for infering the relation between the premise (left)
        and the hypothesis (right), namely (P relation H).

        Args:
            premise: str
                The premise text (left).
            hypothesis: str
                The hypothesis text (right).

        Returns:
            tuple (dict, int, int)
                A dictionary containing the dataset as well as the number of
                chunks in the premise and the number of chunks in the hypothesis.
        """
        
        premise_chunks = self.atomizer(text=premise)
        hypothesis_chunks = self.atomizer(text=hypothesis)

        num_premise, num_hypothesis = len(premise_chunks), len(hypothesis_chunks)
        dataset = [
            {
                "premise": premise_chunks[i], 
                "hypothesis": hypothesis_chunks[j], 
                "doc_i": i, 
                "gen_i": j, 
                "pair_idx": pair_idx
            } for i in range(num_premise) for j in range(num_hypothesis)
        ]
        
        return dataset, num_premise, num_hypothesis

    def build_image(
            self, 
            premise: str, 
            hypothesis: str
    ):
        """
        Build the NLI pair matrix with the NLI scores.

        Args:
            premise: str
                The premise text.
            hypothesis: str
                The hypothesis text.
        
        Returns:
            array
                The NLI score matrix (entailment, contradiction, neutral)
        """
        
        dataset, num_premise, num_hypothesis = self.build_chunk_dataset(premise, hypothesis)

        if len(dataset) == 0:
            return np.zeros((3, 1, 1))

        image = np.zeros((3, num_premise, num_hypothesis))

        if self.model is None:
            self.load_nli()

        for batch in batcher(dataset, batch_size=20):
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]
            batch_tokens = self.tokenizer.batch_encode_plus(
                list(zip(batch_prems, batch_hypos)), 
                padding=True, 
                truncation=True, 
                max_length=self.max_input_length, 
                return_tensors="pt", 
                truncation_strategy="only_first"
            )
            
            with torch.no_grad():
                model_outputs = self.model(**{k: v.to(self.device) for k, v in batch_tokens.items()})

            batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
            batch_evids = batch_probs[:, self.entailment_idx].tolist()
            batch_conts = batch_probs[:, self.contradiction_idx].tolist()
            batch_neuts = batch_probs[:, self.neutral_idx].tolist()

            for b, evid, cont, neut in zip(batch, batch_evids, batch_conts, batch_neuts):
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut

        return image


class NLIScorer:
    def __init__(
            self, 
            model_name="mnli", 
            granularity = "sentence",
            **kwargs
    ):
        """
        NLIScorer constructor.

        Args:
            model_name: str
                Name of the pretrained NLI model (e.g., mnli).
            granularity: str
                Level of granularity used to split a passage (sentence, paragraph).
        """
        
        self.imager = NLIImager(
            model_name=model_name, 
            granularity=granularity,
            **kwargs
        )

        self.granularity = granularity

    def score(
            self, 
            premise: str, 
            hypothesis: str,
            op1: str = "max",
            op2: str = "max"
    ) -> Dict:
        """
        Compute the NLI scores between premise and hypothesis. For example,
        the premise textually entails the hypothesis, or the premise is neutral
        to the hypothesis, or the premise contradicts the hypothesis.

        Args:
            premise: str
                The premise (left).
            hypothesis: str
                The hypothesis (right).
            op1: str
                Operator used when processing the NLI matrices (min, max, mean).
            op2: str
                Operator used whrn processing the NLI matrices (min, max, mean).

        Returns:
            A dict containing the score and the NLI matrix. The score is a
            dict with scores for entailment, contradiction and neutrality.
        """

        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"

        self.op2 = op2
        self.op1 = op1

        image = self.imager.build_image(
            premise=premise, 
            hypothesis=hypothesis
        )
        score = self.image2score(image)
        # return {"image": image, "score": score}

        return score

    def image2score(self, image) -> Dict:
        """
        Computes the final scores.

        Args:
            image: array
                The NLI matrix containing the scores between each of the
                components of the premise and hypothesis texts (i.e., both
                premise and hypothesis are processed at the granularity level
                specified in the NLI scorer constructor).

        Returns:
            A dict containing the aggregated scores for entailment, contradiction 
            and neutrality relations between premise and hypothesis.
        """
        if self.op1 == "max":
            ent_scores = np.max(image[0], axis=1) # entailment
            con_scores = np.max(image[1], axis=1) # contradiction
            neu_scores = np.max(image[2], axis=1) # neutrality
        elif self.op1 == "min":
            ent_scores = np.min(image[0], axis=1) # entailment
            con_scores = np.min(image[1], axis=1) # contradiction
            neu_scores = np.min(image[2], axis=1) # neutrality
        elif self.op1 == "mean":
            ent_scores = np.mean(image[0], axis=1) # entailment
            con_scores = np.mean(image[1], axis=1) # contradiction
            neu_scores = np.mean(image[2], axis=1) # neutrality

        if self.op2 == "mean":
            ent_score = np.mean(ent_scores)
            con_score = np.mean(con_scores)
            neu_score = np.mean(neu_scores)
        elif self.op2 == "min":
            ent_score = np.min(ent_scores)
            con_score = np.min(con_scores)
            neu_score = np.min(neu_scores)
        elif self.op2 == "max":
            ent_score = np.max(ent_scores)
            con_score = np.max(con_scores)
            neu_score = np.mean(neu_scores) #np.max(neu_scores)


        final_score = {
            "entailment": ent_score, 
            "contradiction": con_score, 
            "neutrality": neu_score
        }
        return final_score


if __name__ == "__main__":

    # Initialize the NLI Scorer
    model = NLIScorer(
        granularity="sentence",
        model_name="vitc", 
    ) 

    # atom = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    # context = "Apollo 14 (January 31 – February 9, 1971) was the eighth crewed mission in the United States Apollo program, the third to land on the Moon, and the first to land in the lunar highlands. It was the last of the H missions, landings at specific sites of scientific interest on the Moon for two-day stays with two lunar extravehicular activities (EVAs or moonwalks)."
    # context1 = "Apollo 14 (January 31 - February 9, 1971) was the eighth crewed mission in the United States Apollo program, the third to land on the Moon, and the first to land in the lunar highlands."
    # context2 = "It was the last of the \"H missions\", landings at specific sites of scientific interest on the Moon for two-day stays with two lunar extravehicular activities (EVAs or moonwalks)."
    # res = model.score(premise=context, hypothesis=atom)
    # res1 = model.score(premise=context1, hypothesis=atom)
    # res2 = model.score(premise=context2, hypothesis=atom)

    # print(f"c->a  : {res}")
    # print(f"c1->a  : {res1}")
    # print(f"c2->a  : {res2}")

    # atom = "Apollo 14 brought back approximately 70 kilograms of lunar material, including rocks, soil, and core samples, which have been invaluable for scientific research ever since."
    # # context = "A total of 94 pounds (43 kilograms) of Moon rocks, or lunar samples, were brought back from Apollo 14."
    # context = "A total of 94 pounds (43 kg) of Moon rocks, or lunar samples, were brought back from Apollo 14. Most are breccias, which are rocks composed of fragments of other, older rocks. Breccias form when the heat and pressure of meteorite impacts fuse small rock fragments together. There were a few basalts that were collected in this mission in the form of clasts (fragments) in breccia. The Apollo 14 basalts are generally richer in aluminum and sometimes richer in potassium than other lunar basalts. Most lunar mare basalts collected during the Apollo program were formed from 3.0 to 3.8\u00a0billion years ago. The Apollo 14 basalts were formed 4.0 to 4.3\u00a0billion years ago, older than the volcanism known to have occurred at any of the mare locations reached during the Apollo program."
    # atom = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    # context = "The Saturn V used for Apollo 14 was designated SA-509, and was similar to those used on Apollo 8 through 13. At , it was the heaviest vehicle yet flown by NASA,  heavier than the launch vehicle for Apollo 13."

    # atom = "The treasure hunters were looking for buried artifacts."
    # atom = "The treasure hunters were looking for coins."
    # context = "Treasure hunting is the physical search for treasure. For example, treasure hunters try to find sunken shipwrecks and retrieve artifacts with market value. This industry is generally fueled by the market for antiquities. The practice of treasure-hunting can be controversial, as locations such as sunken wrecks or cultural sites may be protected by national or international law concerned with property ownership, marine salvage, sovereign or state vessels, commercial diving regulations, protection of cultural heritage and trade controls. Treasure hunting can also refer to geocaching – a sport in which participants use GPS units to find hidden caches of toys or trinkets, or various other treasure-hunting games."
    # context = "Large portable metal detectors are used by archaeologists and treasure hunters to locate metallic items, such as jewelry, coins, clothes buttons and other accessories, bullets, and other various artifacts buried beneath the surface."

    # atom = "The treasure hunters were looking for buried artifacts."
    # context = "Treasure hunting is the physical search for treasure. For example, treasure hunters try to find sunken shipwrecks and retrieve artifacts with market value. This industry is generally fueled by the market for antiquities. The practice of treasure-hunting can be controversial, as locations such as sunken wrecks or cultural sites may be protected by national or international law concerned with property ownership, marine salvage, sovereign or state vessels, commercial diving regulations, protection of cultural heritage and trade controls. Treasure hunting can also refer to geocaching – a sport in which participants use GPS units to find hidden caches of toys or trinkets, or various other treasure-hunting games."

    # atom = "Apollo 14 brought back approximately 70 kilograms of lunar material."
    # context = "A total of 94 pounds (43 kg) of Moon rocks, or lunar samples, were brought back from Apollo 14. Most are breccias, which are rocks composed of fragments of other, older rocks. Breccias form when the heat and pressure of meteorite impacts fuse small rock fragments together. There were a few basalts that were collected in this mission in the form of clasts (fragments) in breccia. The Apollo 14 basalts are generally richer in aluminum and sometimes richer in potassium than other lunar basalts. Most lunar mare basalts collected during the Apollo program were formed from 3.0 to 3.8 billion years ago. The Apollo 14 basalts were formed 4.0 to 4.3 billion years ago, older than the volcanism known to have occurred at any of the mare locations reached during the Apollo program."
    # context = "A total of 94 pounds (43 kg) of Moon rocks, or lunar samples, were brought back from Apollo 14."

    # context1 = "Apollo 14's backup crew was Eugene A. Cernan as commander, Ronald E. Evans Jr. as CMP and Joe H. Engle as LMP. The backup crew, with Harrison Schmitt replacing Engle, would become the prime crew of Apollo 17. Schmitt flew instead of Engle because there was intense pressure on NASA to fly a scientist to the Moon (Schmitt was a geologist) and Apollo 17 was the last lunar flight. Engle, who had flown the X-15 to the edge of outer space, flew into space for NASA in 1981 on STS-2, the second Space Shuttle flight."
    # context2 = "Shepard and his crew had originally been designated by Deke Slayton, Director of Flight Crew Operations and one of the Mercury Seven, as the crew for Apollo 13. NASA's management felt that Shepard needed more time for training given he had not flown in space since 1961, and chose him and his crew for Apollo 14 instead. The crew originally designated for Apollo 14, Jim Lovell as the commander, Ken Mattingly as CMP and Fred Haise as LMP, all of whom had backed up Apollo 11, was made the prime crew for Apollo 13 instead."

    atom = "Lanny Flaherty is an American."
    # atom = "Lanny Flaherty is an actor."
    # atom = "Lanny Flaherty was born on December 18, 1949."
    # atom = "Lanny Flaherty notable film credits include Natural Born Killers."
    # context = "lanny flaherty ( july 27, 1942 – february 18, 2024 ) was an american actor. lanny flaherty ( july 27, 1942 – february 18, 2024 ) was an american actor. = = life and career = = flaherty had roles in films and miniseries such as lonesome dove, natural born killers, book of shadows : blair witch 2 and signs. he also had a brief role in men in black 3, and appeared as jack crow in jim mickles 2014 adaptation of cold in july. other film appearances include winter people, millers crossing, blood in blood out, tom and huck and home fries while television roles include guest appearances on the equalizer, new york news and white collar as well as a two - episode stint on the education of max bickford as whammo. flaherty was a graduate of pontotoc high school, and attended university of southern mississippi after high school. he resided in new york city. flaherty died following surgery in february 2024, at the age of 81. = = filmography = = = = = film = = = = = = television = = = = = references = ="
    # context = "= = external links = = lanny flaherty at imdb"
    # context = "= = cast = = drew barrymore as sally jackson catherine o'hara as beatrice lever luke wilson as dorian montier jake busey as angus montier shelley duvall as mrs. jackson kim robillard as billy daryl mitchell as roy lanny flaherty as red jackson chris ellis as henry lever blue deckert as sheriff mark walters as deputy shane steiner as soldier in jeep theresa merritt as mrs. vaughan ( final film role ) jill parker - jones as lamaze instructor morgana shaw as lucy garland"
    context = "lloyd earned a third emmy for his 1992 guest appearance as alistair dimple in road to avonlea ( 1992 ), and won an independent spirit award for his performance in twenty bucks ( 1993 ). he has done extensive voice work, including merlock in ducktales the movie : treasure of the lost lamp ( 1990 ), grigori rasputin in anastasia ( 1997 ), the hacker in the pbs kids series cyberchase ( 2002 – present ), which earned him daytime emmy nominations, and the woodsman in the cartoon network miniseries over the garden wall ( 2014 )."
    
    res = model.score(premise=context, hypothesis=atom, op1="max", op2="max")

    print(f"c->a  : {res}")

    # atom = "The Twelve Days of Christmas ... in most of the Western Church are \
    #     the twelve days from Christmas until the beginning of Epiphany (January \
    #         6th ; the 12 days count from December 25th until January 5th )."

    # context1 = "Some households exchange gifts on the first ( 25 December ) and \
    #     last ( 5 January ) days of the Twelve Days."

    # context2 = "In these traditions , the twelve days begin December 26 ( th ) \
    #     and include Epiphany on January."

    # # Get the scores of the relations between Premise -> Hypothesis
    # res1 = model.score(premise=context1, hypothesis=atom)
    # res2 = model.score(premise=context2, hypothesis=atom)
    # res3 = model.score(premise=context1, hypothesis=context2)
    # res4 = model.score(premise=context2, hypothesis=context1)

    # print(f"c1->a  : {res1['score']}")
    # print(f"c2->a  : {res2['score']}")
    # print(f"c1->c2 : {res3['score']}")
    # print(f"c2->c1 : {res4['score']}")


    # context1 = """Fischer served as chairman of Tourism Australia from 2004 to 
    # 2007, and was later Ambassador to the Holy See from 2009 to 2012."""
    
    # context2 = """Tim Fischer started serving as the Ambassador to the Holy See 
    # on January 30, 2009. He was the first permanent resident ambassador from 
    # Australia to the Holy See. His term ended on January 20, 2012."""

    # context2 = """Tim Fischer started serving as the Ambassador to the Holy See 
    # on January 30, 2009. His term ended on January 20, 2012."""

    # context3 = """On 21 July 2008, Fischer was nominated by Prime Minister Kevin 
    # Rudd as the first resident Australian Ambassador to the Holy See. Fischer 
    # worked closely with the Vatican on all aspects of the canonisation of 
    # Australia's first Roman Catholic saint, Mary MacKillop. He retired from the 
    # post on 20 January 2012."""

    # # Get the scores of the relations between Premise -> Hypothesis
    # scores = model.score(premise=context2, hypothesis=context3)
    # print(f"Score: {scores['score']}")
    # print(f"Image: {scores['image']}")

    # document = """Scientists are studying Mars to learn about the Red Planet 
    # and find landing sites for future missions. One possible site, known as 
    # Arcadia Planitia, is covered instrange sinuous features. The shapes could be 
    # signs that the area is actually made of glaciers, which are large masses of 
    # slow-moving ice. Arcadia Planitia is in Mars' northern lowlands."""

    # summary1 = """There are strange shape patterns on Arcadia Planitia. The 
    # shapes could indicate the area might be made of glaciers. This makes Arcadia 
    # Planitia ideal for future missions."""

    # summary2 = """There are strange shape patterns on Arcadia Planitia. The shapes 
    # could indicate the area might be made of glaciers."""
    
    # document = """Jeff joined Microsoft in 1992 to lead corporate developer 
    # evangelism for Windows NT. He then served as a Group Program manager in 
    # Microsoft's Internet Business Unit. In 1998, he led the creation of 
    # SharePoint Portal Server, which became one of Microsoft’s fastest-growing 
    # businesses, exceeding $2 billion in revenues. Jeff next served as Corporate
    # Vice President for Program Management across Office 365 Services and Servers, 
    # which is the foundation of Microsoft's enterprise cloud leadership. He then 
    # led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft's 
    # mobile-first/cloud-first transformation and acquisitions. Prior to joining 
    # Microsoft, Jeff was vice president for software development for an investment 
    # firm in New York. He leads Office shared experiences and core applications, 
    # as well as OneDrive and SharePoint consumer and business services in Office 
    # 365. Jeff holds a Master of Business Administration degree from Harvard 
    # Business School and a Bachelor of Science degree in information systems and 
    # finance from New York University."""
    
    # summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

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


from typing import List

import torch

from akd.tools.fact_reasoner.fm_factual.nli.alignscore import AlignScore
from akd.tools.fact_reasoner.fm_factual.utils import get_freer_gpu


class AlignScorer:
    """
    A textual consistency scorer based on the recent AlignScore.
    """
    def __init__(
            self,
            model: str,
            ckpt_path: str,
            evaluation_mode: str = "nli_sp",
            granularity: str = "sentence"
    ):
        """
        Initialize the AlignScore scorer.

        Args:
            model: str
                The backbone language model (roberta-base or roberta-large).
            ckpt_path: str
                The path to the model checkpoint.
            evaluation_mode: str
                The evaluation mode (nli_sp - default, nli, bin_sp, bin).
            granularity: str
                Level of granularity for aggregating scores (sentence, paragraph)
        """
        
        self.model = model
        self.ckpt_path = ckpt_path
        self.evaluation_mode = evaluation_mode
        self.granularity = granularity
        assert granularity in ["sentence", "paragraph"], f"Uknown granularity level: {granularity}"
        self.device = "cpu"
        if torch.cuda.is_available():
            # self.device = "cuda"
            self.device = f"cuda:{get_freer_gpu()}"
        
        self.scorer = AlignScore(
            model=self.model,
            batch_size=32,
            device=self.device,
            ckpt_path=self.ckpt_path,
            evaluation_mode=self.evaluation_mode,
            granularity=self.granularity,
            verbose=False
        )

    def score(
            self, 
            premise: str, 
            hypothesis: str,
            op1: str = "max",
            op2: str = "max"
    ):
        """
        Compute the AlignScore between context and claim.

        Args:
            premise: str
                The string representing the context (left-hand-side text).
            hypothesis: str
                The string representing the claim (right-hand-side text).
            op1: str
                Operator used when processing the NLI matrices (min, max, mean).
            op2: str
                Operator used whrn processing the NLI matrices (min, max, mean).

        Returns: dict
            A dictionary containing the alignment, contradiction and neutral scores.
        """

        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"

        self.op2 = op2
        self.op1 = op1
     
        assert self.scorer is not None, f"The AlignScore based scorer must be initialized."

        score = self.scorer.score(contexts=[premise], claims=[hypothesis], op1=op1, op2=op2)
        if "_sp" in self.evaluation_mode:
            score = score[0]
        return dict(entailment=score[0], contradiction=score[1], neutral=score[2])

    def score_all(
            self, 
            premises: List[str], 
            hypotheses: List[str],
            op1: str = "max",
            op2: str = "max"
    ):
        """
        Compute the AlignScore between context and claim.

        Args:
            premise: str
                The string representing the context (left-hand-side text).
            hypothesis: str
                The string representing the claim (right-hand-side text).
            op1: str
                Operator used when processing the NLI matrices (min, max, mean).
            op2: str
                Operator used whrn processing the NLI matrices (min, max, mean).

        Returns: dict
            A dictionary containing the alignment, contradiction and neutral scores.
        """

        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"

        self.op2 = op2
        self.op1 = op1
     
        assert self.scorer is not None, f"The AlignScore based scorer must be initialized."

        scores = self.scorer.score(contexts=premises, claims=hypotheses, op1=op1, op2=op2)
        # if "_sp" in self.evaluation_mode:
        #     score = score[0]
        return [
            dict(entailment=score[0], contradiction=score[1], neutral=score[2]) for score in scores
        ]

if __name__ == "__main__":

    # Initialize the AlignScorer scorer
    model = AlignScorer(
        model="roberta-large", 
        ckpt_path="/home/radu/ckpts/AlignScore-large.ckpt",
        evaluation_mode="nli_sp",
        granularity="paragraph"
    ) 

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

    # summary = """Jeff joined Microsoft in 1992 to lead the company's corporate 
    # evangelism. He then served as a Group Manager in Microsoft's Internet 
    # Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the 
    # company's fastest-growing business, surpassing $3 million in revenue. Jeff 
    # next leads corporate strategy for SharePoint and Servers which is the basis 
    # of Microsoft's cloud-first strategy. He leads corporate strategy for Satya 
    # Nadella and Amy Hood on Microsoft's mobile-first."""

    # res = model.score(context=document, claim=summary)
    # print(f"c->a  : {res}")
    # atom = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    # context = "The Saturn V used for Apollo 14 was designated SA-509, and was similar to those used on Apollo 8 through 13. At , it was the heaviest vehicle yet flown by NASA,  heavier than the launch vehicle for Apollo 13."

    # atom = "The treasure hunters were looking for buried artifacts."
    # context = "Treasure hunting is the physical search for treasure. For example, treasure hunters try to find sunken shipwrecks and retrieve artifacts with market value. This industry is generally fueled by the market for antiquities. The practice of treasure-hunting can be controversial, as locations such as sunken wrecks or cultural sites may be protected by national or international law concerned with property ownership, marine salvage, sovereign or state vessels, commercial diving regulations, protection of cultural heritage and trade controls. Treasure hunting can also refer to geocaching – a sport in which participants use GPS units to find hidden caches of toys or trinkets, or various other treasure-hunting games."

    atom = "Lanny Flaherty is an American."
    # atom = "Lanny Flaherty is an actor."
    # atom = "Lanny Flaherty was born on December 18, 1949."
    # atom = "Lanny Flaherty notable film credits include Natural Born Killers."
    context = "lanny flaherty ( july 27, 1942 – february 18, 2024 ) was an american actor. lanny flaherty ( july 27, 1942 – february 18, 2024 ) was an american actor. = = life and career = = flaherty had roles in films and miniseries such as lonesome dove, natural born killers, book of shadows : blair witch 2 and signs. he also had a brief role in men in black 3, and appeared as jack crow in jim mickles 2014 adaptation of cold in july. other film appearances include winter people, millers crossing, blood in blood out, tom and huck and home fries while television roles include guest appearances on the equalizer, new york news and white collar as well as a two - episode stint on the education of max bickford as whammo. flaherty was a graduate of pontotoc high school, and attended university of southern mississippi after high school. he resided in new york city. flaherty died following surgery in february 2024, at the age of 81. = = filmography = = = = = film = = = = = = television = = = = = references = ="
    # context = "= = external links = = lanny flaherty at imdb"
    # context = "= = cast = = drew barrymore as sally jackson catherine o'hara as beatrice lever luke wilson as dorian montier jake busey as angus montier shelley duvall as mrs. jackson kim robillard as billy daryl mitchell as roy lanny flaherty as red jackson chris ellis as henry lever blue deckert as sheriff mark walters as deputy shane steiner as soldier in jeep theresa merritt as mrs. vaughan ( final film role ) jill parker - jones as lamaze instructor morgana shaw as lucy garland"
    # context = "lloyd earned a third emmy for his 1992 guest appearance as alistair dimple in road to avonlea ( 1992 ), and won an independent spirit award for his performance in twenty bucks ( 1993 ). he has done extensive voice work, including merlock in ducktales the movie : treasure of the lost lamp ( 1990 ), grigori rasputin in anastasia ( 1997 ), the hacker in the pbs kids series cyberchase ( 2002 – present ), which earned him daytime emmy nominations, and the woodsman in the cartoon network miniseries over the garden wall ( 2014 )."

    # context1 = "Apollo 14's backup crew was Eugene A. Cernan as commander, Ronald E. Evans Jr. as CMP and Joe H. Engle as LMP. The backup crew, with Harrison Schmitt replacing Engle, would become the prime crew of Apollo 17. Schmitt flew instead of Engle because there was intense pressure on NASA to fly a scientist to the Moon (Schmitt was a geologist) and Apollo 17 was the last lunar flight. Engle, who had flown the X-15 to the edge of outer space, flew into space for NASA in 1981 on STS-2, the second Space Shuttle flight."
    # context2 = "Shepard and his crew had originally been designated by Deke Slayton, Director of Flight Crew Operations and one of the Mercury Seven, as the crew for Apollo 13. NASA's management felt that Shepard needed more time for training given he had not flown in space since 1961, and chose him and his crew for Apollo 14 instead. The crew originally designated for Apollo 14, Jim Lovell as the commander, Ken Mattingly as CMP and Fred Haise as LMP, all of whom had backed up Apollo 11, was made the prime crew for Apollo 13 instead."
    res = model.score(premise=context, hypothesis=atom, op1="mean", op2="mean")

    print(f"c->a  : {res}")

    # atom = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    # context = "Apollo 14 (January 31 – February 9, 1971) was the eighth crewed mission in the United States Apollo program, the third to land on the Moon, and the first to land in the lunar highlands. It was the last of the H missions, landings at specific sites of scientific interest on the Moon for two-day stays with two lunar extravehicular activities (EVAs or moonwalks)."
    # context1 = "Apollo 14 (January 31 - February 9, 1971) was the eighth crewed mission in the United States Apollo program, the third to land on the Moon, and the first to land in the lunar highlands."
    # context2 = "It was the last of the \"H missions\", landings at specific sites of scientific interest on the Moon for two-day stays with two lunar extravehicular activities (EVAs or moonwalks)."

    # atom = "Apollo 14 brought back approximately 70 kilograms of lunar material."
    # context = "A total of 94 pounds (43 kg) of Moon rocks, or lunar samples, were brought back from Apollo 14. Most are breccias, which are rocks composed of fragments of other, older rocks. Breccias form when the heat and pressure of meteorite impacts fuse small rock fragments together. There were a few basalts that were collected in this mission in the form of clasts (fragments) in breccia. The Apollo 14 basalts are generally richer in aluminum and sometimes richer in potassium than other lunar basalts. Most lunar mare basalts collected during the Apollo program were formed from 3.0 to 3.8 billion years ago. The Apollo 14 basalts were formed 4.0 to 4.3 billion years ago, older than the volcanism known to have occurred at any of the mare locations reached during the Apollo program."
    # context = "A total of 94 pounds (43 kg) of Moon rocks, or lunar samples, were brought back from Apollo 14."

    # context = "List of shipwrecks in August 1828 The list of shipwrecks in August 1828 includes all ships sunk, foundered, grounded, or otherwise lost during August 1828."
    # res = model.score(premise=context1, hypothesis=context2, op1="max", op2="max")
    # res1 = model.score(context=context1, claim=atom)
    # res2 = model.score(context=context2, claim=atom)

    # print(f"c->a  : {res}")
    # print(f"c1->a  : {res1}")
    # print(f"c2->a  : {res2}")

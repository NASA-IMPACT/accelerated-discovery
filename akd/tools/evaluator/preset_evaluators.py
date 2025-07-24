from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    BinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import DAGMetric
from deepeval.test_case import LLMTestCaseParams

# NOTE: in DAG evaluators, each VerdictNode represents a possible endstate of
# the evaluation, the output of the preceeding BinaryJudgementNode if False,
# and otherwise continues to the next BinaryJudgementNode in the chain. In
# the case of reaching an end state, the score assigned to the evaluation is
# the score defined in the VerdictNode. These accept values between 0 and 10,
# for ease of use (easier to score multiple criteria on a 0-10 scale), but this
# score will be normalized back to a 0-1 scale to match the scores from other
# evaluations.

usefulness_5 = BinaryJudgementNode(
    label="Usefullness 5",
    criteria="Highlights Key Substantive Contributions: Does the agent accurately extract/highlight the key findings, arguments, or theoretical contributions from this paper's content?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[VerdictNode(verdict=False, score=9), VerdictNode(verdict=True, score=10)],
)

usefulness_4 = BinaryJudgementNode(
    label="Usefullness 4",
    criteria="Appropriate Content Scope: Is the breadth and depth of the information and arguments within this paper appropriate for its contribution to the user's query?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=6),
        VerdictNode(verdict=True, child=usefulness_5),
    ],
)

usefulness_3 = BinaryJudgementNode(
    label="Usefullness 3",
    criteria="Supports Research Advancement: Does the substance of this paper enable the user to advance their understanding, refine their research, or identify next steps?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=4),
        VerdictNode(verdict=True, child=usefulness_4),
    ],
)

usefulness_2 = BinaryJudgementNode(
    label="Usefullness 2",
    criteria="Provides Novel Insights: Does the content of this paper offer information, arguments, or perspectives that are likely new or non-obvious to the user?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=2),
        VerdictNode(verdict=True, child=usefulness_3),
    ],
)

usefulness_1 = BinaryJudgementNode(
    label="Usefullness 1",
    criteria="Addresses Core Topic: Does the content of this paper directly address the primary research question or topic?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=usefulness_2),
    ],
)

completeness_5 = BinaryJudgementNode(
    label="Completeness 5",
    criteria="Extraction of Key Content Elements: Does the agent effectively identify/extract crucial content elements (e.g., methodology, key findings, main arguments) from this paper?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[VerdictNode(verdict=False, score=9), VerdictNode(verdict=True, score=10)],
)

completeness_4 = BinaryJudgementNode(
    label="Completeness 4",
    criteria="Sufficient Internal Detail: Is the content within this specific paper presented with sufficient depth and detail for the user to understand its primary claims, methodology, and findings clearly?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=6),
        VerdictNode(verdict=True, child=completeness_5),
    ],
)

completeness_3 = BinaryJudgementNode(
    label="Completeness 3",
    criteria="Offers Distinct Perspective/Methodology: Does the content of this paper offer a distinct or alternative perspective/methodology relevant to the query, contributing to a broader understanding if applicable?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=4),
        VerdictNode(verdict=True, child=completeness_4),
    ],
)

completeness_2 = BinaryJudgementNode(
    label="Completeness 2",
    criteria="Presents Significant Content: Does this paper present content that is (or appears to be) a significant, foundational, or key piece of work relevant to the query?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=2),
        VerdictNode(verdict=True, child=completeness_3),
    ],
)

completeness_1 = BinaryJudgementNode(
    label="Completeness 1",
    criteria="Addresses Key Theme/Sub-topic: Does the content of this paper significantly address a relevant theme or sub-topic defined by the user's query?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=completeness_2),
    ],
)

faithfulness_5 = BinaryJudgementNode(
    label="Faithfulness 5",
    criteria="Preserves Content Nuance: Does the agent's representation of this paper's content avoid oversimplification that loses critical nuances, caveats, or limitations mentioned in the source?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[VerdictNode(verdict=False, score=9), VerdictNode(verdict=True, score=10)],
)

faithfulness_4 = BinaryJudgementNode(
    label="Faithfulness 4",
    criteria="No Hallucination/Fabrication of Content: Is all information asserted by the agent about this paper's content actually present in the source document?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=6),
        VerdictNode(verdict=True, child=faithfulness_5),
    ],
)

faithfulness_3 = BinaryJudgementNode(
    label="Faithfulness 3",
    criteria="Correct Attribution of Content: Are specific claims, findings, or data points from this paper's content clearly and correctly attributed?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=4),
        VerdictNode(verdict=True, child=faithfulness_4),
    ],
)

faithfulness_2 = BinaryJudgementNode(
    label="Faithfulness 2",
    criteria="No Misleading Content Snippets: Are extracted textual snippets or paraphrased content from this paper presented in a way that preserves their original meaning and context?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=2),
        VerdictNode(verdict=True, child=faithfulness_3),
    ],
)

faithfulness_1 = BinaryJudgementNode(
    label="Faithfulness 1",
    criteria="Accurate Representation of Main Points: If summaries/analyses of this paper are provided by the agent, do they accurately reflect its main points and conclusions?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=faithfulness_2),
    ],
)

accuracy_5 = BinaryJudgementNode(
    label="Accuracy 5",
    criteria="No Unsubstantiated External Inferences: Does the agent avoid introducing conclusions or interpretations about this paper not directly supported by its explicit content?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[VerdictNode(verdict=False, score=9), VerdictNode(verdict=True, score=10)],
)

accuracy_4 = BinaryJudgementNode(
    label="Accuracy 4",
    criteria="Consistent Interpretation of Content: If content from this paper is used in multiple places by the agent, is its interpretation consistent?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=6),
        VerdictNode(verdict=True, child=accuracy_5),
    ],
)

accuracy_3 = BinaryJudgementNode(
    label="Accuracy 3",
    criteria="Accuracy of Quoted/Paraphrased Content: Are direct quotations or closely paraphrased statements from this paper's content accurate and contextually sound?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=4),
        VerdictNode(verdict=True, child=accuracy_4),
    ],
)

accuracy_2 = BinaryJudgementNode(
    label="Accuracy 2",
    criteria="Justified Content-Based Categorization: If this paper is categorized/tagged by the agent based on its substantive content, are these classifications accurate?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=2),
        VerdictNode(verdict=True, child=accuracy_3),
    ],
)

accuracy_1 = BinaryJudgementNode(
    label="Accuracy 1",
    criteria="Accurate Thesis/Argument Representation: Does the agent accurately represent the main argument, thesis, or research questions of this paper without distortion?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=accuracy_2),
    ],
)

timeliness_4 = BinaryJudgementNode(
    label="Timeliness 4",
    criteria="Prioritization of Currently Relevant Content: Does this paper's content reflect current understanding or key ongoing discussions relevant to the query (unless historical perspective is explicitly sought for this paper)?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[VerdictNode(verdict=False, score=9), VerdictNode(verdict=True, score=10)],
)

timeliness_3 = BinaryJudgementNode(
    label="Timeliness 3",
    criteria="Discernment of Outdated Content: If applicable, does the agent help discern if this paper's content, while perhaps historically relevant, might represent outdated or superseded findings?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=6),
        VerdictNode(verdict=True, child=timeliness_4),
    ],
)

timeliness_2 = BinaryJudgementNode(
    label="Timeliness 2",
    criteria="Enduring Relevance of Foundational Content: If this paper is older, is its core content still considered seminal or essential for understanding the topic's evolution as relevant to the query?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=4),
        VerdictNode(verdict=True, child=timeliness_3),
    ],
)

timeliness_1 = BinaryJudgementNode(
    label="Timeliness 1",
    criteria="Current Applicability of Content: Is the substance/findings of this paper current enough to be relevant, considering the field and query?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=timeliness_2),
    ],
)

usefulness = DAGMetric(
    name="Evaluate usefulness of search result",
    dag=DeepAcyclicGraph(root_nodes=[usefulness_1]),
    verbose_mode=True,
)
completeness = DAGMetric(
    name="Evaluate completeness of search result",
    dag=DeepAcyclicGraph(root_nodes=[completeness_1]),
    verbose_mode=True,
)

faithfulness = DAGMetric(
    name="Evaluate faithfulness of search result",
    dag=DeepAcyclicGraph(root_nodes=[faithfulness_1]),
    verbose_mode=True,
)
accuracy = DAGMetric(
    name="Evaluate accuracy of search result",
    dag=DeepAcyclicGraph(root_nodes=[accuracy_1]),
    verbose_mode=True,
)
timeliness = DAGMetric(
    name="Evaluate timeliness of search result",
    dag=DeepAcyclicGraph(root_nodes=[timeliness_1]),
    verbose_mode=True,
)

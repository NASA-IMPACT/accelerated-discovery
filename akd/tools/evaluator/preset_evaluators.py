from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCaseParams

from akd.tools.evaluator.custom_deepeval_extensions import GraniteGuardianBinaryNode
from akd.tools.granite_guardian_tool import GraniteGuardianTool, RiskDefinition

# NOTE: in DAG evaluators, each VerdictNode represents a possible endstate of
# the evaluation, the output of the preceeding BinaryJudgementNode if False,
# and otherwise continues to the next BinaryJudgementNode in the chain. In
# the case of reaching an end state, the score assigned to the evaluation is
# the score defined in the VerdictNode. These accept values between 0 and 10,
# for ease of use (easier to score multiple criteria on a 0-10 scale), but this
# score will be normalized back to a 0-1 scale to match the scores from other
# evaluations.

# --- Usefulness ---
usefulness_agg_node = NonBinaryJudgementNode(
    criteria="Are values of usefulness_1, usefulness_2, usefulness_3, usefulness_4 and usefulness_5 True?",
    children=[
        VerdictNode(verdict="All five are True", score=10),
        VerdictNode(verdict="Four out of the five are true", score=8),
        VerdictNode(verdict="Three of the five are True", score=6),
        VerdictNode(verdict="Two of the five are True", score=4),
        VerdictNode(verdict="Only one of the five is True", score=2),
        VerdictNode(verdict="None of the five are true", score=0),
    ],
)

usefulness_5 = TaskNode(
    output_label="usefulness_5",
    instructions="Based on the user's query (input) and the retrieved paper content, does the agent's output accurately highlight the key findings, arguments, or theoretical contributions?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[usefulness_agg_node],
)

usefulness_4 = TaskNode(
    output_label="usefulness_4",
    instructions="Considering the user's query (input) and the retrieved paper content, does the agent's output reflect appropriate breadth and depth for addressing the query?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[usefulness_agg_node],
)

usefulness_3 = TaskNode(
    output_label="usefulness_3",
    instructions="Given the user's query and the retrieved content, does the output help the user advance their understanding, refine their research, or identify next steps?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[usefulness_agg_node],
)

usefulness_2 = TaskNode(
    output_label="usefulness_2",
    instructions="Does the agent's output provide novel or non-obvious insights based on the retrieved paper content and the user's query?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[usefulness_agg_node],
)

usefulness_1 = TaskNode(
    output_label="usefulness_1",
    instructions="Does the agent's output directly address the primary research question or topic as posed in the input query, using the retrieved paper content?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[usefulness_agg_node],
)

# --- Completeness ---
completeness_agg_node = NonBinaryJudgementNode(
    criteria="Are values of completeness_1, completeness_2, completeness_3, completeness_4 and completeness_5 True?",
    children=[
        VerdictNode(verdict="All five are True", score=10),
        VerdictNode(verdict="Four out of the five are true", score=8),
        VerdictNode(verdict="Three of the five are True", score=6),
        VerdictNode(verdict="Two of the five are True", score=4),
        VerdictNode(verdict="Only one of the five is True", score=2),
        VerdictNode(verdict="None of the five are true", score=0),
    ],
)

completeness_5 = TaskNode(
    output_label="completeness_5",
    instructions="Based on the retrieved paper content, does the agent's output effectively extract crucial elements such as methodology, key findings, and main arguments?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[completeness_agg_node],
)

completeness_4 = TaskNode(
    output_label="completeness_4",
    instructions="Does the retrieved paper content contain sufficient detail, and does the agent reflect that detail in its output to make the paper's claims, methodology, and findings clear?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[completeness_agg_node],
)

completeness_3 = TaskNode(
    output_label="completeness_3",
    instructions="Does the retrieved paper provide a distinct or alternative perspective or methodology relevant to the query, and is that represented in the output?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[completeness_agg_node],
)

completeness_2 = TaskNode(
    output_label="completeness_2",
    instructions="Does the paper, as represented in the output, present foundational or key content that is significant to the user query?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[completeness_agg_node],
)

completeness_1 = TaskNode(
    output_label="completeness_1",
    instructions="Does the agent's output identify and represent key themes or sub-topics from the retrieved paper that are relevant to the user query?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[completeness_agg_node],
)

# --- Faithfulness ---
faithfulness_agg_node = NonBinaryJudgementNode(
    criteria="Are values of faithfulness_1, faithfulness_2, faithfulness_3, faithfulness_4 and faithfulness_5 True?",
    children=[
        VerdictNode(verdict="All five are True", score=10),
        VerdictNode(verdict="Four out of the five are true", score=8),
        VerdictNode(verdict="Three of the five are True", score=6),
        VerdictNode(verdict="Two of the five are True", score=4),
        VerdictNode(verdict="Only one of the five is True", score=2),
        VerdictNode(verdict="None of the five are true", score=0),
    ],
)

faithfulness_5 = TaskNode(
    output_label="Faithfulness 5",
    instructions="Does the output preserve the nuance of the original content from the retrieved paper, avoiding oversimplification or omission of important caveats?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[faithfulness_agg_node],
)

faithfulness_4 = TaskNode(
    output_label="Faithfulness 4",
    instructions="Is all information asserted in the output about the paper actually present in the retrieved content, with no hallucinated or fabricated details?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[faithfulness_agg_node],
)

faithfulness_3 = TaskNode(
    output_label="Faithfulness 3",
    instructions="Are the claims, findings, and data points from the paper correctly and clearly attributed in the output based on the retrieved content?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[faithfulness_agg_node],
)

faithfulness_2 = TaskNode(
    output_label="Faithfulness 2",
    instructions="Are paraphrased or quoted snippets in the output faithful to the original meaning and context of the retrieved paper content?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[faithfulness_agg_node],
)

faithfulness_1 = TaskNode(
    output_label="Faithfulness 1",
    instructions="Do the summaries or analyses in the output accurately represent the main points and conclusions of the retrieved paper?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[faithfulness_agg_node],
)

# --- Accuracy ---
accuracy_agg_node = NonBinaryJudgementNode(
    criteria="Are values of accuracy_1, accuracy_2, accuracy_3, accuracy_4 and accuracy_5 True?",
    children=[
        VerdictNode(verdict="All five are True", score=10),
        VerdictNode(verdict="Four out of the five are true", score=8),
        VerdictNode(verdict="Three of the five are True", score=6),
        VerdictNode(verdict="Two of the five are True", score=4),
        VerdictNode(verdict="Only one of the five is True", score=2),
        VerdictNode(verdict="None of the five are true", score=0),
    ],
)

accuracy_5 = TaskNode(
    output_label="accuracy_5",
    instructions="Does the output avoid introducing conclusions or interpretations not directly supported by the retrieved paper content?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[accuracy_agg_node],
)

accuracy_4 = TaskNode(
    output_label="accuracy_4",
    instructions="When the agent references the same paper content in multiple places, does it do so with consistent interpretation?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[accuracy_agg_node],
)

accuracy_3 = TaskNode(
    output_label="accuracy_3",
    instructions="Are direct quotes or paraphrased content in the output accurate and faithful to the meaning in the retrieved paper?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[accuracy_agg_node],
)

accuracy_2 = TaskNode(
    output_label="accuracy_2",
    instructions="If the agent categorizes the paper based on its content, are those classifications justified by the retrieved content?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[accuracy_agg_node],
)

accuracy_1 = TaskNode(
    output_label="accuracy_1",
    instructions="Does the output represent the paper's thesis or main argument accurately, based on the retrieved content?",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[accuracy_agg_node],
)

# --- Timeliness ---
timeliness_agg_node = NonBinaryJudgementNode(
    criteria="Are values of accuracy_1, accuracy_2, accuracy_3 and accuracy_4 True?",
    children=[
        VerdictNode(verdict="All four are True", score=10),
        VerdictNode(verdict="Three of the four are True", score=7.5),
        VerdictNode(verdict="Two of the four are True", score=5),
        VerdictNode(verdict="Only one of the four is True", score=2.5),
        VerdictNode(verdict="None of the four are true", score=0),
    ],
)

timeliness_4 = TaskNode(
    output_label="timeliness_4",
    instructions="Does the retrieved paper reflect current knowledge or active discussions relevant to the query, unless a historical perspective is explicitly relevant?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[timeliness_agg_node],
)

timeliness_3 = TaskNode(
    output_label="timeliness_3",
    instructions="Does the output help the user recognize if the paper contains outdated or superseded information, where applicable?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[timeliness_agg_node],
)

timeliness_2 = TaskNode(
    output_label="timeliness_2",
    instructions="If the paper is older, does the agent's output indicate whether it still holds enduring relevance for understanding the topic?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    children=[timeliness_agg_node],
)

timeliness_1 = TaskNode(
    output_label="timeliness_1",
    instructions="Is the retrieved paper current enough to be relevant to the user's query, considering the domain?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[timeliness_agg_node],
)

# --- Guardian ---
groundedness_guardian_node = GraniteGuardianBinaryNode(
    criteria=RiskDefinition.GROUNDEDNESS,
    evaluation_params=[
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    granite_guardian_tool=GraniteGuardianTool(),
    children=[
        VerdictNode(verdict=False, score=0.0),
        VerdictNode(verdict=True, score=0.5),
    ],
)

answer_relevance_guardian_node = GraniteGuardianBinaryNode(
    criteria=RiskDefinition.ANSWER_RELEVANCE,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    granite_guardian_tool=GraniteGuardianTool(),
    children=[
        VerdictNode(verdict=False, score=0.0),
        VerdictNode(verdict=True, score=0.5),
    ],
)

guardian_root = TaskNode(
    output_label="Guardian root node status",
    instructions="Is the retrieved content non-trivial?",
    evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[
        answer_relevance_guardian_node,
        groundedness_guardian_node,
    ],
)

# --- Article structure ---
structure_agg_node = NonBinaryJudgementNode(
    criteria="Are values of is_complete, has_references and is_approprate True?",
    children=[
        VerdictNode(verdict="All three are True", score=10),
        VerdictNode(verdict="Two out of the three are true", score=6.66),
        VerdictNode(verdict="Only one of the three is True", score=3.33),
        VerdictNode(verdict="None of the three are true", score=0),
    ],
)

structure_check_completeness_task = TaskNode(
    output_label="is_complete",
    instructions=(
        "Verify that the article contains all key sections: "
        "Abstract, Introduction, Literature Review, Methodology, Results, Discussion/Conclusion, and References. "
        "Each should have non-trivial content."
    ),
    evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[structure_agg_node],
)

structure_check_content_task = TaskNode(
    output_label="is_appropriate",
    instructions=(
        "Verify that each section contains content relevant to its heading: "
        "For example, Results should contain findings, Methodology should describe methods, "
        "and Introduction should provide context and objectives."
    ),
    evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[structure_agg_node],
)

structure_check_references_task = TaskNode(
    output_label="has_references",
    instructions=(
        "Verify that the References section contains valid scholarly citations (e.g., numbered list, APA, MLA, or similar formats). "
        "Ensure it is not empty and contains at least a few distinct citations."
    ),
    evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
    children=[structure_agg_node],
)


structure = DAGMetric(
    name="Evaluate structure of scientific article",
    dag=DeepAcyclicGraph(
        root_nodes=[
            structure_check_completeness_task,
            structure_check_content_task,
            structure_check_references_task,
        ],
    ),  # Root node
    verbose_mode=True,
)
usefulness = DAGMetric(
    name="Evaluate usefulness of search result",
    dag=DeepAcyclicGraph(
        root_nodes=[
            usefulness_1,
            usefulness_2,
            usefulness_3,
            usefulness_4,
            usefulness_5,
        ],
    ),
    verbose_mode=True,
)
completeness = DAGMetric(
    name="Evaluate completeness of search result",
    dag=DeepAcyclicGraph(
        root_nodes=[
            completeness_1,
            completeness_2,
            completeness_3,
            completeness_4,
            completeness_5,
        ],
    ),
    verbose_mode=True,
)
faithfulness = DAGMetric(
    name="Evaluate faithfulness of search result",
    dag=DeepAcyclicGraph(
        root_nodes=[
            faithfulness_1,
            faithfulness_2,
            faithfulness_3,
            faithfulness_4,
            faithfulness_5,
        ],
    ),
    verbose_mode=True,
)
accuracy = DAGMetric(
    name="Evaluate accuracy of search result",
    dag=DeepAcyclicGraph(
        root_nodes=[
            accuracy_1,
            accuracy_2,
            accuracy_3,
            accuracy_4,
            accuracy_5,
        ],
    ),
    verbose_mode=True,
)
timeliness = DAGMetric(
    name="Evaluate timeliness of search result",
    dag=DeepAcyclicGraph(
        root_nodes=[
            timeliness_1,
            timeliness_2,
            timeliness_3,
            timeliness_4,
        ],
    ),
    verbose_mode=True,
)
guardian = DAGMetric(
    name="Evaluate based on some Granite Guardian risks.",
    dag=DeepAcyclicGraph(
        root_nodes=[guardian_root],
    ),
    verbose_mode=True,
)

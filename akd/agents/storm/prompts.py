from langchain_core.prompts import ChatPromptTemplate

# =============================================================
# Outline prompts
# =============================================================

gen_initial_outline_inst = """Write an outline for a Wikipedia page.

Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information."""


GEN_INITIAL_OUTLINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", gen_initial_outline_inst),
        (
            "user",
            "Topic you want to write: {topic}\nWrite the Wikipedia page outline:\n",
        ),
    ],
)


REFINE_OUTLINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
You need to make sure that the outline is comprehensive and specific. \
Topic you are writing about: {topic}

Old outline:

{old_outline}""",
        ),
        (
            "user",
            "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
        ),
    ],
)


# =============================================================
# Writer prompts
# =============================================================

SECTION_WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:\n\n"
            "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
        ),
        ("user", "Write the full WikiSection for the {section} section."),
    ],
)


WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:\n\n"
            "{draft}\n\nStrictly follow Wikipedia format guidelines.",
        ),
        (
            "user",
            'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",'
            " avoiding duplicates in the footer. Include URLs in the footer.",
        ),
    ],
)

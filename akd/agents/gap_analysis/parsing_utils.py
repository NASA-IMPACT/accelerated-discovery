import ast
from typing import Dict, List

from bs4 import BeautifulSoup

from .prompts import SECTION_GROUPER_PROMPT
from .structures import Section, SubSection


def parse_html(html_content: str) -> List[Dict[str, str]]:
    """
    Parses the given HTML content to extract sections and their textual content.

    The function identifies all <h2> headers as section titles. For each section, it collects the text
    from sibling elements until the next <h2> is encountered. Supported sibling elements include
    paragraphs (<p>), lists (<ul>, <li>), tables (<table> captions), and figures (<figure> figcaptions).

    Args:
        html_content (str): A string containing the HTML content to be parsed.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary maps a section title (str)
                              to its concatenated content (str).
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception as e:
        raise RuntimeError(f"Failed to parse HTML content: {e}")
    parsed_content = {}
    sections = soup.find_all("h2")
    if not sections:
        raise ValueError("No <h2> section headers found in the HTML content.")
    for section in sections:
        section_title = section.get_text(strip=True)
        section_content = []
        sibling = section.find_next_sibling()
        while sibling and sibling.name != "h2":
            if sibling.name in ["p", "ul", "li"]:
                section_content.append(sibling.get_text(strip=True))
            elif sibling.name == "table":
                caption = sibling.find("caption")
                if caption:
                    caption_text = caption.get_text(strip=True)
                    section_content.append(caption_text)
            elif sibling.name == "figure":
                figcaption = sibling.find("figcaption")
                if figcaption:
                    figure_text = figcaption.get_text(strip=True)
                    section_content.append(figure_text)
            sibling = sibling.find_next_sibling()
        if section_title not in parsed_content:
            parsed_content.update({section_title: " ".join(section_content)})
    return parsed_content


def create_sections_from_parsed_html(parsed_output, section_titles, paper_title):
    """
    Constructs structured Section and SubSection objects from parsed HTML content and section titles.

    Args:
        parsed_output (dict): Mapping of section/subsection titles to their textual content.
        section_titles (list[list[str]]): Nested list where each sublist contains a main section title
                                          followed optionally by subsection titles.
        paper_title (str): Title of the paper used to exclude redundant sections like the paper title or abstract.

    Returns:
        tuple:
            - List[Section]: List of Section objects with populated content and optional subsections.
            - List[list[str]]: Cleaned list of section titles with excluded titles removed.
    """
    sections = []
    sections_to_remove = []
    for section_title_list in section_titles:
        # The first value is the section heading
        main_section_title = section_title_list[0]
        # The title does not have content and abstract is fetched from  S2
        if main_section_title.lower() in [paper_title.lower(), "abstract"]:
            sections_to_remove.append(main_section_title)
            continue
        subsections_data = []
        section_content = parsed_output.get(main_section_title)
        if len(section_title_list) > 1:
            for sub_title in section_title_list[1:]:
                sub_content = parsed_output.get(sub_title)
                if sub_content is not None:
                    subsections_data.append(
                        SubSection(title=sub_title, content=sub_content),
                    )
        if subsections_data:
            sections.append(
                Section(
                    title=main_section_title,
                    content=section_content,
                    subsections=subsections_data,
                ),
            )
        else:
            if section_content is not None:
                sections.append(
                    Section(
                        title=main_section_title,
                        content=section_content,
                        subsections=None,
                    ),
                )
            else:
                sections.append(
                    Section(title=main_section_title, content=None, subsections=None),
                )
    clean_section_titles = [
        title
        for title in section_titles
        if not any(removed_section in title for removed_section in sections_to_remove)
    ]
    return sections, clean_section_titles


async def group_section_titles(parsed_output: list[dict], llm):
    """
    Uses an LLM to group section titles into hierarchical section groupings.

    Args:
        parsed_output (list[dict]): Dictionary mapping section titles to their content.
        llm: Language model instance used to perform the section grouping.

    Returns:
        list[list[str]]: A list of grouped section titles, where each group represents a main section
                         followed by its subsections (if any).
    """
    section_titles = list(parsed_output.keys())
    group_sections_chain = SECTION_GROUPER_PROMPT | llm
    model_output = await group_sections_chain.ainvoke(
        {"input_sections": section_titles},
    )
    parsed_sections = ast.literal_eval(model_output.content)
    return parsed_sections

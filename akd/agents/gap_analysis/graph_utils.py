import ast
import asyncio
import networkx as nx

from typing import List, Tuple, Dict, Union

from .prompts import *
from .structures import ParsedPaper, PaperDataItem


async def add_paper_to_graph(G: nx.Graph, paper: ParsedPaper, llm) -> nx.Graph:
    """
    Adds a paper and its associated metadata to a NetworkX graph.

    This function creates a node for the paper and connects it with the following edges for now:
    - Author nodes via `authored_by` edges
    - Cited and referencing papers via `refers_to` and `cited_by` edges
    - Section and subsection nodes if the paper is a `ParsedPaper`, using LLM classification for section types

    Args:
        paper (ParsedPaper): The paper object to add to the graph.
        G (networkx.Graph): The graph to update with paper data.
        llm: The language model used for classifying section titles (only used for `ParsedPaper`).

    Returns:
        networkx.Graph: The updated graph containing the new paper and its connections.
    """
    paper_data = paper.model_dump(exclude_unset=False)

    paper_id = paper_data.get("paper_id")
    if not paper_id:
        raise ValueError("paper_id is required to add node to the graph.")

    if paper_id not in G:
        # Add the paper node with all fields
        G.add_node(paper_id, **paper_data, node_type='paper')

        # Add author nodes and edges
        authors = paper_data.get("authors") or []
        for author in authors:
            author_id = author.get("authorId")
            author_name = author.get("name")
            if author_id:
                if author_id not in G:
                    G.add_node(author_id, name=author_name, title=author_name, node_type='author')
                G.add_edge(paper_id, author_id, relationship='authored_by')

        # Handle citations and references
        def add_related_papers(related_list, relationship, node_type):
            if not related_list:
                return
            for related in related_list:
                related_id = related.get("paperId") or related.get("paper_id")
                related_title = related.get("title", "")
                if related_id:
                    if related_id not in G:
                        G.add_node(related_id, paperTitle=related_title, title=related_title, node_type=node_type)
                    G.add_edge(paper_id, related_id, relationship=relationship)

        add_related_papers(paper_data.get("references"), relationship="refers_to", node_type="reference")
        add_related_papers(paper_data.get("citations"), relationship="cited_by", node_type="citation")

        # Handle parsed paper sections (specific to ParsedPaper)
        if isinstance(paper, ParsedPaper):
            sections = paper_data.get("sections") or []
            section_to_key = await classify_section_titles(paper, llm)
            for i, section in enumerate(sections, start=1):
                section_title = section.get('title')
                section_content = section.get('content')
                section_subsections = section.get('subsections')

                if section_title not in section_to_key:
                    print(f"Section {section_title} missing for {paper_data.get('title')}")
                    continue

                section_node_type = section_to_key[section_title]
                if section_node_type == 'misc':
                    continue

                section_id = f"{paper_id}_{i}"
                G.add_node(section_id, section_title=section_title, section_content=section_content,
                           node_type=section_node_type)
                G.add_edge(paper_id, section_id, relationship='contains_section')

                if section_subsections:
                    for j, subsection in enumerate(section_subsections, start=1):
                        subsection_id = f"{paper_id}_{i}_{j}"
                        G.add_node(subsection_id, subsection_title=subsection.get('title'),
                                   subsection_content=subsection.get('content'), node_type=section_node_type)
                        G.add_edge(section_id, subsection_id, relationship='contains_subsection')

    return G
                    

def get_nodes_by_type(G: nx.Graph, node_type: str) -> List[str]:
    """
    Retrieves all nodes from the graph that match a specific node type.

    Args:
        G (networkx.Graph): The graph from which to retrieve nodes.
        node_type (str): The type of node to filter by (e.g., 'paper', 'author', 'section').

    Returns:
        List[str]: A list of node identifiers matching the specified node type.
    """
    nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("node_type") == node_type]
    return nodes


def extract_direct_triples(G: nx.Graph, start_node: str) ->List[Tuple[str, str, str]]:
    """
    Extracts all direct relationship triples from the given start node in the graph.

    Args:
        G (networkx.Graph): The graph containing nodes and edges.
        start_node (str): The node ID from which to extract direct relationships.

    Returns:
        List[Tuple[str, str, str]]: A deduplicated list of (node_type_1, relationship, node_type_2) triples.
    """
    triples = []
    if start_node not in G:
        raise ValueError(f"Node {start_node} does not exist in the graph.")
    for neighbor, data in G[start_node].items():
        node_type_1 = G.nodes[start_node].get("node_type", "unknown")
        node_type_2 = G.nodes[neighbor].get("node_type", "unknown")
        relationship = data.get("relationship", "unknown")
        triples.append((node_type_1, relationship, node_type_2))
    return list(set(triples))


def get_connected_nodes(G: nx.Graph, 
                        start_node: str, 
                        target_node_type: str = None,
                        relation: str = None,
                        first_node: bool = False) -> Union[List, Dict, None]:
    """
    Retrieves nodes directly connected to a given node, with optional filtering by node type and relationship.

    Args:
        G (networkx.Graph): The graph containing nodes and edges.
        start_node (str): The node ID from which to search for connected nodes.
        target_node_type (Optional[str]): Filter to return only nodes of this type. If None, all types are considered.
        relation (Optional[str]): Filter to return only edges with this relationship label. If None, all relationships are considered.
        first_node (bool): If True, return only the first matching node's attributes (dict). If False, return all matches.

    Returns:
        Union[List[Tuple[str, str, dict]], dict, None]:
            - If first_node=False: A list of tuples (node_id, relationship, node_attributes) matching the criteria.
            - If first_node=True: A single node's attributes (dict) if found, else None.
    """
    if start_node not in G:
        raise ValueError(f"Node {start_node} does not exist in the graph.")
    connected_nodes = []
    for neighbor, data in G[start_node].items():
        neighbor_node_type = G.nodes[neighbor].get("node_type", "unknown")
        relationship = data.get("relationship", "unknown")
        if (target_node_type is None or neighbor_node_type == target_node_type) and \
           (relation is None or relation == relationship):
            connected_nodes.append((neighbor, relationship, G.nodes[neighbor]))
            if first_node:
                return G.nodes[neighbor]
    return connected_nodes if not first_node else None


async def select_subsections(connected_subsection_nodes: List,
                             query: str,
                             llm) -> List[str]:
    """
    Selects relevant subsection nodes from a list of connected nodes based on a user query using an LLM.

    Args:
        connected_subsection_nodes (List[Tuple[str, str, dict]]): List of tuples representing subsection nodes 
            connected to a parent node. Each tuple contains (node_id, relation, node_data).
        query (str): The user question or query.
        llm: The language model instance used to evaluate relevance.

    Returns:
        List[str]: A list of node IDs corresponding to the selected relevant subsections.
    """
    title_to_node_id_map = {}
    subsection_titles = []
    selected_subsection_nodes = []
    for node_id, relation, data in connected_subsection_nodes:
        subsection_title = f"{data['node_type']} {relation} {data['subsection_title']}"
        subsection_titles.append(subsection_title)
        title_to_node_id_map[subsection_title] = node_id
    if len(subsection_titles) != 0:
        output = await llm.ainvoke(select_subsection_prompt.format_prompt(query=query, titles=subsection_titles))
        selected_subsection_titles = ast.literal_eval(output.content)
        selected_subsection_nodes = [title_to_node_id_map[subsection_title] for subsection_title in selected_subsection_titles]
    return selected_subsection_nodes


async def retrieve_relevant_sections(G: nx.Graph,
                                    paper_node: str,
                                    query: str,
                                    llm) -> List[str]:
    """
    Retrieves relevant section and subsection nodes from a graph based on a user query using an LLM.

    Args:
        G (networkx.Graph): The graph containing paper and section nodes.
        paper_node (str): The node ID of the paper in the graph.
        query (str): The user query for retrieving relevant sections.
        llm: The language model instance used for relevance evaluation.

    Returns:
        List[str]: A list of node IDs representing relevant sections and subsections.
    """
    relations = extract_direct_triples(G, paper_node)
    relation_traversal_output = await llm.ainvoke(traverse_relations_prompt.invoke({"query": query, "relations": relations}))
    selected_sections = ast.literal_eval(relation_traversal_output.content)
    connected_section_nodes = []
    for selected_section in selected_sections:
        _, relation, target_node_type = selected_section
        section_nodes = get_connected_nodes(G, paper_node, target_node_type, relation)
        connected_section_nodes.extend(section_nodes)

    connected_subsection_nodes = []
    for section_node in connected_section_nodes:
        start_node_id, relation, _ = section_node
        subsection_nodes = get_connected_nodes(G, start_node_id, relation="contains_subsection")
        connected_subsection_nodes.extend(subsection_nodes)

    selected_subsection_nodes = await select_subsections(connected_subsection_nodes, query, llm)

    selected_nodes = []
    for conn_section_node in connected_section_nodes:
        node_id, _, data = conn_section_node
        if len(data['section_content']) > 0:
            selected_nodes.append(node_id)
    selected_nodes.extend(selected_subsection_nodes)
    return selected_nodes


async def classify_section_titles(paper: PaperDataItem, llm) -> Dict[str, str]:
    """
    Classifies section titles of a paper into standardized categories using an LLM.

    Args:
        paper (PaperDataItem): The paper object containing grouped section titles.
        llm: Language model instance used for classification.

    Returns:
        dict[str, str]: A mapping from each section title to its classified category label.
    """
    section_titles = paper.section_titles
    main_section_titles = [section_list[0] for section_list in section_titles]
    classify_sections_chain = section_classifier_prompt | llm
    out = await classify_sections_chain.ainvoke({'sections_to_group': main_section_titles})
    grouped_section_titles = ast.literal_eval(out.content)
    section_to_key = {}
    for key, item in grouped_section_titles.items():
        for val in item:
            section_to_key[val] = key
    return section_to_key


async def select_nodes(G: nx.Graph,
                       query: str,
                       llm) -> List[List[str]]:
    """
    Selects relevant section and subsection nodes across all paper nodes in the graph based on a query.

    Args:
        G (networkx.Graph): The graph containing paper and related nodes.
        query (str): The user query to find relevant sections.
        llm: The language model instance used to evaluate relevance.

    Returns:
        List[List[str]]: A list where each element is a list of node IDs relevant to a particular paper.
    """
    paper_nodes = get_nodes_by_type(G, 'paper')
    tasks = [
        retrieve_relevant_sections(G, paper_node, query, llm)
        for paper_node in paper_nodes
    ]
    all_selected_nodes = await asyncio.gather(*tasks)
    return all_selected_nodes


def format_attributed_answers(G: nx.Graph, attributed_answers: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Formats attributed answers with metadata from the graph.

    Args:
        G (networkx.Graph): The graph containing paper and section nodes.
        attributed_answers (Dict[str, str]): A dictionary mapping source node IDs to generated content.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary where each key is a source ID and the value is a dictionary 
        containing the paper title, section title, URL, and the associated content.
    """
    attributed_content = {}
    for source_id, content in attributed_answers.items():
        section_title_id = source_id.strip().split(" ")[-1]
        source_paper_id = section_title_id.split('_')[0]
        title = G.nodes[source_paper_id]['title']
        if section_title_id not in G.nodes:
            continue
        section_data = G.nodes[section_title_id]
        if 'section_title' in section_data.keys():
            section_title = section_data['section_title']
        else:
            section_title = section_data['subsection_title']
        attributed_content[source_id] = {"title": title,
                                "section_title": section_title,
                                "url": G.nodes[source_paper_id]['url'],
                                "content": content}
    return attributed_content


async def generate_final_answer(G: nx.Graph, 
                                query: str,
                                all_selected_nodes: List[List[str]],
                                llm) -> str:
    """
    Generates a consolidated answer for the query based on selected nodes in the graph.

    Args:
        G (networkx.Graph): The graph containing paper and section nodes.
        query (str): The user query for which the answer is generated.
        all_selected_nodes (List[List[str]]): A list of lists of node IDs selected as relevant for the query.
        llm: The language model instance used for generating and summarizing answers.

    Returns:
        Response: The output of the summarization chain containing the final consolidated answer.
    """
    local_answer_chain = gen_answer_prompt | llm
    final_answer_chain = summarise_answer_prompt | llm
    all_answer_prompt_data = []
    for selected_nodes in all_selected_nodes:
        answer_prompt_data = []
        for selected_node in selected_nodes:
            node_data = G.nodes[selected_node]
            answer_prompt_data.append({
                'query': query, 
                'section_title': node_data[list(node_data.keys())[0]], 
                'node_type': node_data['node_type'],
                'section_content': node_data[list(node_data.keys())[1]]
            })
        all_answer_prompt_data.append(answer_prompt_data)
    all_answers = await asyncio.gather(*[
        local_answer_chain.abatch(answer_prompt_data) for answer_prompt_data in all_answer_prompt_data
    ])
    attributed_answers = {}
    for answers, selected_nodes in zip(all_answers, all_selected_nodes):
        for node_id, answer in zip(selected_nodes, answers):
            attributed_answers.update({node_id: answer.content})
    output = await final_answer_chain.ainvoke(input={'query': query, "attributed_answer_list": attributed_answers})
    attributed_source_answers = format_attributed_answers(G, attributed_answers)
    return output, attributed_source_answers


def format_final_answer(G: nx.Graph, output: str) -> str:
    """
    Formats the final answer by appending human-readable citations based on graph node metadata.

    Args:
        G (networkx.Graph): The graph containing paper and section nodes.
        output (str): The generated model output/

    Returns:
        str: A string combining the main answer and a formatted "### Sources" section with numbered citations.
    """
    cited_sources = output.split("### Sources")[-1].strip().split('\n')
    attributed_sources = []
    for i in range(len(cited_sources)):
        section_title_id = cited_sources[i].strip().split(" ")[-1]
        source_paper_id = section_title_id.split('_')[0]
        title = G.nodes[source_paper_id]['title']
        if section_title_id not in G.nodes:
            continue
        section_data = G.nodes[section_title_id]
        if 'section_title' in section_data.keys():
            section_title = section_data['section_title']
        else:
            section_title = section_data['subsection_title']
        attributed_sources.append(f"[{i+1}] {section_title}, {title}")
    sources = "\n".join(attributed_sources)
    answer = output.split("### Sources")[0].strip()
    final_output = f"{answer}\n\n### Sources\n{sources}"
    return final_output
import json
import chromadb
import torch
import nltk
import argparse
import re
import os, shutil

from bert_score import BERTScorer
from operator import itemgetter
from typing import List
from chromadb.utils import embedding_functions

from flask import Flask, jsonify, request

app = Flask(__name__)

COLLECTION_NAME = "wikipedia_en"
DB_PATH = "/home/radu/wiki_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# BERTScore calculation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORER = BERTScorer(model_type='bert-base-uncased', device=DEVICE)

class ChromaReader:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str,
        collection_metadata: dict = None,
    ):
        """
        Initialize the ChromaDB.

        Args:
            collection_name: str
                The collection name in the vector database.
            persist_directory: str
                The directory used for persisting the vector database.
            embedding_model: str
                The embedding model.
            collection_metadata: dict
                A dict containing the collection metadata.
        """
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    def is_empty(self):
        return self.collection.count() == 0

    def query(self, query_texts: str, n_results: int = 5, where_document: dict = None):
        """
        Returns the closests vector to the question vector
        
        Args:
            query_texts: str
                The user query text.
            n_results: int
                The number of results to generate.

        Returns
            The closest result to the given question.
        """
        return self.collection.query(query_texts=query_texts, n_results=n_results, where_document=where_document)
  
def postprocess(text: str) -> List[str]:
    """
    Postprocess a retrieved document by breaking it into passages.

    Args:
        text: str
            A string representing the retrieved document.

    Returns:
        A list of passages (i.e., paragraphs).
    """

    if text.count("\n\n") > 0:
        paragraphs = [p.strip() for p in text.split("\n\n")]
    else:
        paragraphs = [p.strip() for p in text.split("\n")]

    result = []
    paragraphs = [p for p in paragraphs if len(p) > 10]
    for p in paragraphs:
        subpars = [pp.strip() for pp in p.split("\n")]
        new_p = ""
        for pp in subpars:
            sentences = nltk.tokenize.sent_tokenize(pp)
            sentences = [sent for sent in sentences if len(sent)>10]
            if len(sentences) >= 0:
                new_pp = " ".join(sentences)
                new_p += " " + new_pp
        if len(new_p) > 0:
            result.append(new_p.strip())
    return result

def split_paragraphs(text: str) -> List[str]:
    """
    Postprocess a retrieved document by breaking it into paragraphs. A paragraph
    consists is a group of sentences and paragraphs are assumed to be delimited 
    by "\n\n" (2 or more new-line sequences).

    Args:
        text: str
            A string representing the retrieved document.

    Returns:
        A list of passages (i.e., paragraphs).
    """

    no_newlines = text.strip("\n")  # remove leading and trailing "\n"
    split_text = NEWLINES_RE.split(no_newlines)  # regex splitting
    chunks = [p for p in split_text if len(p.split()) > 10]

    paragraphs = []
    for chunk in chunks:
        subpars = [pp.strip() for pp in chunk.split("\n")]
        new_p = ""
        for sp in subpars:
            sentences = nltk.tokenize.sent_tokenize(sp)
            sentences = [sent for sent in sentences if len(sent.split())>5]
            if len(sentences) >= 0:
                new_pp = " ".join(sentences)
                new_p += " " + new_pp
        if len(new_p.strip()) > 0:
            paragraphs.append(new_p.strip())
    return paragraphs

def get_title(text: str) -> str:
    """
    Get the title of the retrived document. By definition, the first line in the
    document is the title (we embedded them like that).
    """
    return text[:text.find("\n")]

def scores(scorer, references: List[str], candidates: List[str]):
    P, R, F1 = scorer.score(candidates, references)
    return F1.numpy()

# Create a persistent ChromaDB client
chroma = ChromaReader(
    collection_name=COLLECTION_NAME, 
    persist_directory=DB_PATH, 
    embedding_model=EMBEDDING_MODEL, 
    collection_metadata={"hnsw:space": "cosine"}
)

def payload_is_valid(payload):
    for key in payload.keys():
        if key not in ["query_text", "top_k", "n_results", "granularity", "relevance", "where_document"]:
            return False
    return True

def cleanup_tmp():
    folder = '/tmp'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route('/query', methods=['POST'])
def query_chromadb():
    """
    Main POST request of the Flask app. The payload is assumed to have the
    following format:
        - query_text: the text of the query
        - top_k: the number of passages to retrieve (if None, then all paragraphs are returned)
        - n_results: the number of relevant chunks retrieved from chromadb
        - granularity: the granularity of the returned passages (document, paragraph)

    Return codes:
        500 - unexpected error
        400 - invalid payload
        200 - success
    """
    
    payload = json.loads(request.data)

    # Check if payload syntax is correct
    if not payload_is_valid(payload):
        return jsonify({ 'error': 'Invalid query payload.' }), 400

    try:
        query_text = payload.get('query_text')
        top_k = payload.get('top_k')
        n_results = payload.get('n_results', 1)
        granularity = payload.get('granularity')
        relevance = payload.get('relevance', False)
        where_document = payload.get('where_document', None)

        print(f"Retrieving {n_results} relevant documents for query: {query_text}")
        print(f"Processing retrieved documents to get top {top_k} passages")
        print(f"Granularity: {granularity}")
        print(f"Relevance: {relevance}")
        print(f"Where document: {where_document}")

        # Retrieve the relevant chunks from the vector store
        relevant_chunks = chroma.query(
            query_texts=[query_text],
            n_results=n_results,
            where_document=where_document
        )

        # Get the chunks (documents)
        docs = relevant_chunks["documents"][0]

        passages = []
        if granularity == "paragraph":
            if relevance == True:
                print(f"Returning paragraphs with bert scores...")
                paragraphs = []
                for i, doc in enumerate(docs):
                    title = get_title(doc)
                    paragraphs.extend([dict(title=title, text=par) for par in split_paragraphs(text=doc)])
                references = [query_text]*len(paragraphs)
                candidates = [par["text"] for par in paragraphs]
                sc = scores(SCORER, references, candidates)
                temp = [(sc[i], paragraphs[i]) for i in range(len(paragraphs))]
                temp = sorted(temp, key=itemgetter(0), reverse=True)
                passages = [p for _,p in temp]
            else:
                print(f"Returning paragraphs without bert scores...")
                passages = []
                for i, doc in enumerate(docs):
                    title = get_title(doc)
                    passages.extend([dict(title=title, text=par) for par in split_paragraphs(text=doc)])
        elif granularity == "document":
            passages = [dict(title=get_title(doc), text=doc) for doc in docs]
        else:
            raise ValueError(f"Unknow granularity level: {granularity}.")

        result = {}
        n = len(passages) if top_k is None else min(top_k, len(passages))
        for i in range(n):
            result[i] = passages[i] # a passage is a dict with title and text as keys

        print(f"Results are: {result}")
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='9.59.197.15', help="IP address of the host where the API runs")
    parser.add_argument('--port', type=int, default=5000, help="Port of the host where the API runs")
    args = parser.parse_args()

    assert args.address is not None
    assert args.port is not None

    app.run(host=args.address, port=args.port)

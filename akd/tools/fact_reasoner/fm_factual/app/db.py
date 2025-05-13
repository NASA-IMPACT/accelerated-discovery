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

# Embeds Wikipedia into a vector store (e.g., Chromadb)

import argparse
import re
import uuid
from typing import Any, Iterable, List, Optional

import chromadb
import nltk
import numpy as np
import torch
from chromadb.utils import embedding_functions
from datasets import load_dataset
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from tqdm import tqdm

from akd.tools.fact_reasoner.fm_factual.utils import set_seed

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

CHARACTER_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=4000,
    chunk_overlap=0
)

TOKEN_SPLITTER = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

def get_passages(text: str) -> List[str]:
    """
    Split a text into a list of passages.
    """
    character_split_texts = CHARACTER_SPLITTER.split_text(text)
    
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += TOKEN_SPLITTER.split_text(text)
    torch.cuda.empty_cache()
    return token_split_texts

def split_paragraphs(text: str) -> List[str]:
    """
    Postprocess a retrieved document by breaking it into paragraphs.

    Args:
        text: str
            A string representing the retrieved document.

    Returns:
        A list of paragraphs.
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

def batched(iterable: Iterable, max_batch_size: int):
    """ Batches an iterable into lists of given maximum size, yielding them one by one. """
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= max_batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

class ChromaBuilder:
    """
    Builds a Chromadb based vector store for Wikipedia.
    """
    
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
        
        # Get the available cuda device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # Fix the seed
        set_seed(42)

        # Initialize the chromadb strore
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

    def load(
        self,
        texts: Iterable[str],
        metadata: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        
        """
        Run more texts through the embeddings and add to the vectorstore.
        
        Args:
            texts: (Iterable[str])
                Texts to add to the vectorstore.
            metadata: (Optional[List[dict]], optional)
                Optional list of metadatas.
            ids: (Optional[List[str]], optional)
                Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        # Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        self.collection.upsert(metadatas=metadata, documents=texts, ids=ids)
        return ids

    def is_empty(self):
        return self.collection.count() == 0

    def build_db(
            self, 
            dataset_path: str, 
            dataset_name: str, 
            granularity: str = "document"
    ):
        """
        Build the vector store from the input dataset.

        Args:
            dataset_path: str
                The dataset path for building the vector store (huggingface).
            dataset_name: str
                The dataset name for building the vector store (huggingface).
            granularity: str
                The granularity of the documents used for embedding. The 
                default value is "document" which means that the entire document
                is embedded. Alternatively, we can use "passage" to embed at
                passage level (the passage is capped at 1000 chars).
        """
        assert granularity in ["document", "passage"], \
            f"Granularity must be passage or document."
        print(f"Building the vector store from dataset: {dataset_path}:{dataset_name}")
        dataset = load_dataset(dataset_path, dataset_name, trust_remote_code=True)

        # Maximum batch size for chroma, cannot submit more than 5,461 embeddings at once.
        batch_size = 5460 
        
        print(f"Embedding is done at granularity: {granularity}")
        if granularity == "document":
            # Embed at document granularity
            documents = dataset['train'].to_pandas()
            documents["indextext"] = documents["title"].astype(str) + "\n" + documents["text"]
            num_batches = int(np.ceil(len(documents)/batch_size))
            print(f"Processing {len(documents)} documents in {num_batches} batches...")

            for j in tqdm(range(num_batches), total=num_batches, desc="Batches"):  
                documents_j = documents[batch_size*j:batch_size*(j+1)]

                _ = self.load(
                    texts=documents_j.indextext.tolist(),
                    # Chroma handles tokenization, embedding, and indexing 
                    # automatically. You can skip that and add your own 
                    # embeddings as well.
                    metadata=[
                        {"title": title, "id": id}
                        for (title, id) in zip(documents_j.title, documents_j.id)
                    ],  # filter on these!
                    ids=[str(i) for i in documents_j.id],  # unique for each doc
                )
        elif granularity == "passage":
            # Split documents into 1000 char passages and embed them separately.
            documents = dataset['train'].to_pandas()
            n = len(documents)
            num_batches = int(np.ceil(len(documents)/batch_size))
            print(f"Processing {len(documents)} documents in {num_batches} batches...")

            for j in tqdm(range(num_batches), total=num_batches, desc="Batches"):
                documents_j = documents[batch_size*j:batch_size*(j+1)]
                titles = documents_j['title'].tolist()
                contents = documents_j['text'].tolist()

                # Create passages in the current batch
                passages = []
                for title, content in list(map(lambda x, y:(x,y), titles, contents)):
                    passages.extend([(str(uuid.uuid1()), title, psg) for psg in get_passages(content)])
                    # for psg in get_passages(content):
                    #     passages.append((str(uuid.uuid1()), title, psg))

                # title = documents['title'].iloc[j]
                # content = documents['text'].iloc[j]
                # passages = get_passages(content)
                # ids = [str(uuid.uuid1()) for _ in passages]
                # metadatas = [{"title": title, "id": id} for id in ids]
                m = int(np.ceil(len(passages)/batch_size))
                print(f"Batch {j}/{num_batches} generated {len(passages)} passages and {m} chunks")
                for batch in tqdm(batched(passages, max_batch_size=batch_size), total=m, desc="Passages"):
                    texts = [t for _,_,t in batch]
                    ids = [i for i,_,_ in batch]
                    metadatas = [{"title": title, "id": i} for i,title,_ in batch]
                    self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

            print(f"Finished embedding {n} documents at passage level.")
        else:
            raise ValueError(f"Unknown granularity level: {granularity}")
        
        print("Finished.")

    def query(self, query_texts: str, n_results: int = 5):
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
        return self.collection.query(query_texts=query_texts, n_results=n_results)
  
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--build_db', action='store_true')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--collection_name', type=str, default='wikipedia_en')
    parser.add_argument('--dataset_path', type=str, default='graelo/wikipedia')
    parser.add_argument('--dataset_name', type=str, default='20230901.en')
    parser.add_argument('--granularity', type=str, default='document')

    args = parser.parse_args()
    assert (args.db_path is not None)

    # Create a persistent ChromaDB client
    chroma = ChromaBuilder(
        collection_name=args.collection_name, 
        persist_directory=args.db_path, 
        embedding_model=args.embedding_model, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    if args.build_db:
        chroma.build_db(
            dataset_path=args.dataset_path, 
            dataset_name=args.dataset_name,
            granularity=args.granularity
        ) 
    else:
        query_text = "What is a bishop?"

        relevant_chunks = chroma.query(
            query_texts=[query_text],
            n_results=5,
        )

        print(f"Retrieving relevant passages for: {query_text}")
        top_k = 10
        docs = relevant_chunks["documents"][0]
        passages = []
        for i, doc in enumerate(docs):
            print(f"Post-processing document {i} with length {len(doc)}...")
            passages.extend(postprocess(doc))
        
        print(f"Retriving top_k = {top_k} passages...")
        for i in range(top_k):
            print(passages[i])
            print("---" * 20)

    print("Done.")

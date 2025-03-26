# Try and improve wikipedia retrieval

from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Initialize the retriever
retriever = WikipediaRetriever(lang="en")

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=False,
    chunk_size=1000,
    chunk_overlap=0
)

# character_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n"],
#     keep_separator=False,
#     chunk_size=1000,
#     chunk_overlap=0
# )

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

# Fetch documents related to a specific topic
results = retriever.invoke("Gerhard Fischer (inventor)")
print(f"Number of results: {len(results)}")

# Process the results
for doc in results:
    print("----------------------------------------------")
    title = doc.metadata["title"]
    summary = doc.metadata["summary"]

    print(f"Title: {title.strip()}")
    print(f"Summary: {summary.strip()}")
    character_split_texts = character_splitter.split_text(doc.page_content)

    print(f"\nTotal chunks: {len(character_split_texts)}")
    for chunk in character_split_texts:
        print(chunk)
        print("=================================================")

    # token_split_texts = []
    # for text in character_split_texts:
    #     token_split_texts += token_splitter.split_text(text)

    # print(token_split_texts[0])
    # print(f"\nTotal chunks: {len(token_split_texts)}")

    # print(doc.page_content)
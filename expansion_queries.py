from pypdf import PdfReader
import os
from openai import OpenAI

from pypdf import PdfReader
import numpy as np
import umap
import chromadb

#load the openai key from the environment variable
openai_key = os.getenv("OPENAI_API_KEY")
#create a openai client object
openai_client = OpenAI(api_key=openai_key)

#create an embedding function using the sentence transformer model
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction()

#create a chroma client object
collection_name = 'microsoft-annual-report'
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

#create a pdf reader object
reader = PdfReader("data/microsoft-annual-report.pdf")
#read the text from the pdf
pages = [p.extract_text().strip() for p in reader.pages]
#filter out empty pages
pages = [page for page in pages if page]

#split the text into chunks using the RecursiveCharacterTextSplitter and SentenceTransformersTokenTextSplitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pages))
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#put the texts into the chroma collection
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(
    ids=ids,
    documents=token_split_texts,
)
#function to get the top k similar documents from the chroma collection for all the given list of queries
def get_similar_documents(queries, k=3):
    results = chroma_collection.query(
        query_texts=queries,
        n_results=k,
        include=["documents", "embeddings", "distances"]
    )
    return results

query = "What details can you provide about the factors that lead to Microsoft's revenue growth in the fiscal year 2023?"
results = get_similar_documents([query], k=3)

# for i, doc in enumerate(results['documents'][0]):
#     id = results['ids'][0][i]
#     distance = results['distances'][0][i]
#     print(f"Document ID: {id}, Distance: {distance} \nContent: {doc}\n")

#function to generate multiple queries based on the initial query
def generate_multi_query(query, model="gpt-4o"):
    system_prompt = f"""
    You are an expert in breaking down complex questions into simpler sub-questions.
    Given the main question, generate a list of 3-5 specific sub-questions that will help in answering the main question comprehensively.
    Ensure that the sub-questions cover different aspects of the main question and are clear and concise.
    Provide the sub-questions in a separate line without numbering or bullets.
    """
    user_prompt = f"create sub questions for the following question: {query}"
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    multi_queries = response.choices[0].message.content.strip()
    return multi_queries

multi_queries = generate_multi_query(query)
print("Generated Sub-Questions:")
#break on new line and print each question
# for q in multi_queries.split("\n"):
#     print(q)

#make a list that contains the original query and the augmented queries
joint_queries = [query] + multi_queries.split("\n")
print(joint_queries)

#get the similar documents for all the joint queries
results = get_similar_documents(joint_queries, k=5)

#de-dup the documents based on the ids
unique_docs = {}
for i, documents in enumerate(results["documents"]):
    for j, doc in enumerate(documents):
        doc_id = results["ids"][i][j]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = {"document":doc, "embedding":results["embeddings"][i][j], "distance":results["distances"][i][j]}

#print the results per query
# for i, query in enumerate(joint_queries):
#     print(f"Query: {query}\n")
#     print("Results:")
#     for j, doc in enumerate(results["documents"][i]):
#         doc_id = results["ids"][i][j]
#         distance = results["distances"][i][j]
#         print(f"Document ID: {doc_id}, Distance: {distance}\nContent: {doc}\n")
#     print("-" * 100)

#plot the embeddings of the unique documents using umap
#create embeddings for the original query
# orginal_query_embedding = embedding_function([query])
# #create embeddings for the joint queries
# joint_query_embeddings = embedding_function(joint_queries)
# #get embeddings for the unique documents
# unique_doc_embeddings = [unique_docs[doc_id]["embedding"] for doc_id in unique_docs]
# #embeddings for all the docs in the database
# all_doc_embeddings = chroma_collection.get(
#     include=["embeddings"]
# )["embeddings"]
# #fit the umap model on all the document embeddings
# umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(all_doc_embeddings)
# #project the original query embedding to 2D
# projected_original_query_embedding = umap_transform.transform(orginal_query_embedding)
# #project the joint query embeddings to 2D
# projected_joint_query_embeddings = umap_transform.transform(joint_query_embeddings)
# #project the unique document embeddings to 2D
# projected_unique_doc_embeddings = umap_transform.transform(unique_doc_embeddings)
# #project all the document embeddings to 2D
# projected_all_doc_embeddings = umap_transform.transform(all_doc_embeddings)

#plot using matplotlib
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 8))
#plot all the document embeddings in light gray
# plt.scatter(
#     projected_all_doc_embeddings[:, 0],
#     projected_all_doc_embeddings[:, 1],
#     c="lightgray",
#     label="All Documents",
#     alpha=0.5,
# )
# #plot the unique document embeddings in blue
# plt.scatter(
#     projected_unique_doc_embeddings[:, 0],
#     projected_unique_doc_embeddings[:, 1],
#     c="blue",
#     label="Retrieved Documents",
#     edgecolors="black",
#     s=100,
# )
# #plot the joint query embeddings in orange
# plt.scatter(
#     projected_joint_query_embeddings[:, 0],
#     projected_joint_query_embeddings[:, 1],
#     c="orange",
#     label="Joint Queries",
#     marker="X",
#     s=200,
#     edgecolors="black",
# )
# #plot the original query embedding in red
# plt.scatter(
#     projected_original_query_embedding[:, 0],
#     projected_original_query_embedding[:, 1],
#     c="red",
#     label="Original Query",
#     marker="*",
#     s=300,
#     edgecolors="black",
# )
# plt.title("UMAP Projection of Document and Query Embeddings")
# plt.legend()
# plt.show()

from collections import defaultdict
from operator import itemgetter

def build_rankings(results):
    # Case 1: Chroma dict (common)
    if isinstance(results, dict) and 'ids' in results:
        ids_nested = results['ids']
        if isinstance(ids_nested, list) and (not ids_nested or isinstance(ids_nested[0], list)):
            return [list(ids_for_q) for ids_for_q in ids_nested]
        raise TypeError("Chroma results['ids'] should be a list-of-lists")

    # Case 2: list of dicts with 'ids'
    if isinstance(results, list) and results and isinstance(results[0], dict) and 'ids' in results[0]:
        return [list(res['ids']) for res in results]

    # Case 3: list of lists of ids
    if isinstance(results, list) and results and isinstance(results[0], (list, tuple)):
        return [list(r) for r in results]

    raise TypeError(f"Unsupported results format: {type(results)}")

def rrf(rankings, K=60, topk=5):
    scores = defaultdict(float)
    for ranking in rankings:
        for r, doc_id in enumerate(ranking, start=1):  # rank starts at 1
            scores[doc_id] += 1.0 / (K + r)
    sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)  # list of (id, score)
    return sorted_scores[:topk]

def unique_docs_from(results):
    """Dedup map {doc_id: doc_text} from any of the supported shapes."""
    uniq = {}
    if isinstance(results, dict) and 'ids' in results and 'documents' in results:
        for ids_for_q, docs_for_q in zip(results['ids'], results['documents']):
            for did, doc in zip(ids_for_q, docs_for_q):
                uniq.setdefault(did, doc)
        return uniq

    if isinstance(results, list):
        for res in results:
            if isinstance(res, dict) and 'ids' in res and 'documents' in res:
                for did, doc in zip(res['ids'], res['documents']):
                    uniq.setdefault(did, doc)
            elif isinstance(res, (list, tuple)):
                # no texts available in this shape; skip
                pass
    return uniq

# --- usage with your Chroma `results` ---
rankings = build_rankings(results)      # -> list of lists of doc_ids
top = rrf(rankings, K=60, topk=5)       # -> list of (doc_id, score)

uniq = unique_docs_from(results)        # -> {doc_id: doc_text}
for doc_id, score in top:
    print(f"{doc_id}  score={score:.4f}\n{uniq.get(doc_id, '')}\n" + "-"*80)


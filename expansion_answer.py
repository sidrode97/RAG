from pypdf import PdfReader
import os 
from openai import OpenAI
from helper_utils import word_wrap
import numpy as np

from pypdf import PdfReader
#import the text splitters from langchain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

#import the embedding model from langchain
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

#import umap for dimensionality reduction
import umap

#get the OpenAI API key from the environment variable
openai_key = os.getenv("OPENAI_API_KEY")

#setup the pdf reader to read the document
reader = PdfReader("/Users/siddharthraodeb/Documents/GitHub/RAG/data/microsoft-annual-report.pdf")
#extract the text from the pdf
pdf_texts = [p.extract_text().strip() for p in reader.pages]

#get rid of empty pages
pdf_texts = [p for p in pdf_texts if p]

#pretty print the first page
# print(
#     word_wrap(pdf_texts[0], width=100)
# )

#initialise the character splitter object to split the text into chunks  #recursively split by new lines, sentences, spaces and characters in that #sequence upto a chunk size of 1000 characters. This preserves natural #boundaries as much as possible
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " ",""], chunk_size=1000, chunk_overlap=0)

#put the whole document together and split the text into chunks
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

#print the number of chunks created
# print(f"Number of chunks created: {len(character_split_texts)}")
# print(character_split_texts[0])

#initialise the sentence transformer text splitter to split the text into #chunks based on token count
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

#split the chunks further to align with token window of the model
token_split_text = []
for text in character_split_texts:
    token_split_text.extend(token_splitter.split_text(text))

#print the number of chunks created
# print(f"Number of chunks created after token split: {len(token_split_text)}")

#create the sentence transformer embedding model object
embedding_function = SentenceTransformerEmbeddingFunction()
#test the embedding function
# print(embedding_function(token_split_text[0]))

#initialise the vector database to store the embeddings
collection_name = "microsoft-annual-report"
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

#store everything in the vector database
#create ids for each chunk
ids = [str(i) for i in range(len(token_split_text))]
#add the chunks to the vector database
chroma_collection.add(
    ids=ids,
    documents=token_split_text,
)

#create a query function to get the relevant chunks from the vector database
def query_collection(query, n_results=5):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents","embeddings","distances"],
    )
    return results
# test the query function
query = "what was the revenue of microsoft in 2023 and how does it compare to the previous year?"

results = query_collection(query)
print("number of results retrieved: " + str(len(results["documents"][0])))
# #print the results
# for i, res in enumerate(results["documents"][0]):
#     #print the id of the result
#     id = results["ids"][0][i]
#     #print the distance of the result
#     distance = results["distances"][0][i]
#     print(f"Result id {id} at distance {distance}:")
#     #print the result
#     print(res)
#     print("\n")

#create an openai client to create a pseudo answer for augmentation
def pseudo_answer(query):
    openai_client = OpenAI(api_key=openai_key)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role":"system",
                "content":"You are a helpful assistant that helps to answer questions"
            },
            {
                "role":"user",
                "content":f"answer the following question in one to two sentences {query}"
            }
        ],
    )
    return response.choices[0].message.content

# test the pseudo answer function
answer = pseudo_answer(query)

# print(f"the question is : {query} and the answer is : {answer}")

#create a joint query using the pseudo answer and the original query
joint_query = f"{query} {answer}"

#query the collection using the joint query to get the relevant chunks
results_enhanced_query = query_collection(joint_query)

#print the results
retrieved_docs_enhanced_query = results_enhanced_query["documents"][0]
# for i, res in enumerate(retrieved_docs):
#     #print the id of the result
#     id = results["ids"][0][i]
#     #print the distance of the result
#     distance = results["distances"][0][i]
#     print(f"Result id {id} at distance {distance}:")
#     #print the result
#     print(res)
#     print("\n")

#evaluate the query to see if it is closer to the embedding space
#get the embeddings for all the documents in the collection
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
# get the embedding for the original query
original_query_embedding = embedding_function([query])
#print the shape of the un-enhanced query embeddings
print(f"shape of the embeddings: {np.array(original_query_embedding).shape}")
#get the embedding for the joint query
enhanced_query_embedding = embedding_function([joint_query])
#print the shape of the enhanced query embeddings
print(f"shape of the enhanced query embedding: {np.array(enhanced_query_embedding).shape}")
#use umap to reduce the dimensionality of the embeddings to 2D
umap_transform = umap.UMAP(random_state=42, transform_seed=42).fit(embeddings)
#project the dataset embeddings to 2D
projected_dataset_embeddings = umap_transform.transform(embeddings)
#project the original query's embeddings to 2D
projected_original_query_embedding = umap_transform.transform(original_query_embedding)
#project the enhanced query's embeddings to 2D
projected_enhanced_query_embedding = umap_transform.transform(enhanced_query_embedding)
#project the retrieved embeddings from the original query to 2D
projected_retrieved_embeddings = umap_transform.transform(results["embeddings"][0])
#project the retrieved embeddings from the enhanced query to 2D
projected_retrieved_embeddings_enhanced_query = umap_transform.transform(results_enhanced_query["embeddings"][0])

#make a graph to show the embeddings
import matplotlib.pyplot as plt

plt.figure()

#plot the dataset embeddings
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    c="lightgrey",
    label="Document Embeddings",
)

#plot retrieved embeddings
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="blue",
    label="Retrieved Embeddings",
)

plt.scatter(
    projected_retrieved_embeddings_enhanced_query[:, 0],
    projected_retrieved_embeddings_enhanced_query[:, 1],
    s=200,
    facecolors="none",
    edgecolors="orange",
    label="Retrieved Embeddings (Enhanced Query)",
)

#plot original query embedding
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    c="red",
    label="Original Query",
)

#plot enhanced query embedding
plt.scatter(
    projected_enhanced_query_embedding[:, 0],
    projected_enhanced_query_embedding[:, 1],
    s=150,
    marker="X",
    c="green",
    label="Enhanced Query",
)

plt.gca().set_aspect("equal", "datalim")
plt.title("plot of original and enhanced query in the embedding space")
plt.axis("off")
plt.show()  # display the plot

# #use the enhanced results to generate the final answer
# def generate_final_answer(docs, query):
#     openai_client = OpenAI(api_key=openai_key)
#     context = "\n".join(docs)
#     system_prompt = (
#         "you are a helpful assistant for question-answering based on provided" "context.If you dont know the answer say I dont know. Use maximum of"
#         "3 sentences and keep the answer concise.\n\n"
#     )
#     user_prompt = (
#         "use only the context below to answer the question.\n\n"
#         f"Context: {context}\n\n"
#         f"Question: {query}\n\n"
#     )
#     response = openai_client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#     )
#     answer = response.choices[0].message.content
#     return answer

# final_answer = generate_final_answer(retrieved_docs_enhanced_query, query)
# print(f"the question is : {query} \n" + f"the final answer is : {final_answer}")

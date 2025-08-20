import os
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

#get the openai api key
openai_key = os.getenv("OPENAI_API_KEY")

#get the embedding function that is going to be used to create document embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

#initialise the chromadb client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
#give the collection a name
collection_name = "document_qa_collection"
#check if the collection exists, if not create it with the embedding function
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

#intialise the OpenAI client
openai_client = OpenAI(api_key=openai_key)
#get a response
# resp = openai_client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of India?"}
#     ]
# )

#load the documents from the directory of news articles
def load_documents_from_directory(directory_path):
    print("Loading documents from directory:", directory_path)
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({"id": filename, "text": content})
    return documents

#function to split the documents into chunks
#we have overlapping characters so that context is preserved between chunks. #the greater the overlap, the more context is preserved but the more tokens are #used
def split_text(text, chunk_size=1000, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

#load the documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print("loaded " + str(len(documents)) + " documents from the directory")

#split the documents into chunks
print("splitting the document into chunks")
chunked_documents = []
for document in documents:
    chunked_document = split_text(document['text'])
    #we need to save the sequence of the chunk within the document as part of #the key as well
    for index, chunk in enumerate(chunked_document):
        chunked_documents.append({
            "id":f"{document['id']}_chunk{index+1}", 
            "text":chunk})

#create a function to get openai embeddings for the chunks
def get_openai_embeddings(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

#generate embeddings for the chunked documents and save it in the dict
#thus the dict will now contain the key, the text and the embedding
for chunked_document in chunked_documents:
    chunked_document["embedding"] = get_openai_embeddings(chunked_document['text'])

#enter the chunked documents into the collection to save in a structured way.
#Note:upsert adds a document if id does not exist, or updates it if it does.
for chunked_document in chunked_documents:
    collection.upsert(
        ids=[chunked_document['id']],
        documents=[chunked_document['text']],
        embeddings=[chunked_document['embedding']],
    )

print("Chunked documents have been upserted into the collection.")

#function to query the collection.
def query_collection(query, top_k=5):
    #the response will contain the top_k most relevant documents to the query
    #it will return ids, documents and metadatas.
    #for more info see docs.(how to get the embeddings)
    response = collection.query(query_texts=[query], n_results=top_k)
    #extract all the relevant document chunks from the response
    relevant_document_chunks = [doc_chunk for doc in response['documents']for doc_chunk in doc]
    #take a peek at the top matching chunks
    # for idx, doc_chunk in enumerate(response["documents"][0]):
    #     chunk_id = response["ids"][0][idx]
    #     distance = response["distances"][0][idx]
    #     print(f"Found document chunk: {doc_chunk} with ID: {chunk_id}, Distance: {distance}")
    print("returning the top f{top_k} relevant document chunks")
    return relevant_document_chunks

#create a function to generate a response using the OpenAI client
def generate_openai_response(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = (
        "you are a helpful assistant for question-answering based on provided" "context. Use the following retrieved context to answer the question"
        ".If you dont know the answer say I dont know. Use maximum of"
        "3 sentences and keep the answer concise.\n\n"
        f"context:\n{context}\n\n"
        f"question:\n {question}"
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    return answer

#example question
question = "Can you tell me something about Hugging Face and its contributions to the field of AI?"
relevant_chunks = query_collection(question)
answer = generate_openai_response(question, relevant_chunks)

print(f"Question: {question}")
print(f"Answer: {answer}")



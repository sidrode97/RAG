from pypdf import PdfReader
import os 
from openai import OpenAI
from helper_utils import word_wrap

from pypdf import PdfReader

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

#import the text splitters from langchain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

#initialise the character splitter object to split the text into chunks  #recursively split by new lines, sentences, spaces and characters in that #sequence upto a chunk size of 1000 characters. This preserves natural #boundaries as much as possible
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " ",""], chunk_size=1000, chunk_overlap=0)

#put the whole document together and split the text into chunks
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

#print the number of chunks created
print(f"Number of chunks created: {len(character_split_texts)}")
# print(character_split_texts[0])

# #initialise the sentence transformer text splitter to split the text into #chunks based on token count
# token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

# #split the chunks further to align with token window of the model
# token_split_text = []
# for text in character_split_texts:
#     token_split_text.extend(token_splitter.split_text(text))

# #print the number of chunks created
# print(f"Number of chunks created after token split: {len(token_split_text)}")

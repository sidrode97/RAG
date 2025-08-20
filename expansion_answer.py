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

#get rid of empty strings
pdf_texts = [p for p in pdf_texts if p]

print(
    word_wrap(pdf_texts[0], width=100)
)

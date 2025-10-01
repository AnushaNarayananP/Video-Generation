from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import time
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(quantize=True, bits=4)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", quantization_config=quantization_config,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
import transformers
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    repetition_penalty=1.2,
    return_full_text=True,
    max_new_tokens=1000)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
from langchain_community.document_loaders import TextLoader

# load the document and split it into chunks
loader = TextLoader("./demo.txt", encoding='utf-8')
docs = loader.load()
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, separators=['\n\n', '\n', '.']
)
document_chunks = text_splitter.split_documents(docs)
embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-en-v1.5')
chroma_db = Chroma.from_documents(document_chunks, embedding_model)
import warnings
warnings.filterwarnings('ignore')
# Use Prompt only once
import time

retriever = chroma_db.as_retriever()

# Create question answer chain
qa_chain = RetrievalQA.from_chain_type(mistral_llm, retriever=retriever)

while True:
    # Ask questions to chatbot
    # Do you know language DtsDummyLanguage?
    # How to use it for web development?
    question = input("Please enter your question (or 'quit' to stop): ")

    if question.lower() == 'quit':
        break

    start_time = time.time()
    response = qa_chain({"query": question})
    end_time = time.time()

    total_time = int(end_time - start_time)

    print(response['result'])
    print(f"Total calculation time: {total_time} seconds")
    # Use Prompt twice


while True:
    # Ask questions to chatbot
    # Do you know language DtsDummyLanguage?
    # How to use it for web development?
    question = input("Please enter your question (or 'quit' to stop): ")

    if question.lower() == 'quit':
        break

    start_time = time.time()
    similar_search_result = chroma_db.similarity_search(question)
    chroma_db_for_prompt = Chroma.from_documents(similar_search_result, embedding_model)
    retriever = chroma_db_for_prompt.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(mistral_llm, retriever=retriever)
    response = qa_chain({"query": question})

    end_time = time.time()
    total_time = int(end_time - start_time)

    print(response['result'])
    print(f"Total calculation time: {total_time} seconds")
    
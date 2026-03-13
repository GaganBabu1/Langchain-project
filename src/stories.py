

import warnings
warnings.filterwarnings("ignore")

pdf_name = "data/sample.pdf"

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_name)
documents=loader.load()

print("Pages: ",len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =300,
    chunk_overlap=50
)
docs=text_splitter.split_documents(documents)
print("chunks: ",len(docs))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=50,
    temperature=0.0,
    do_sample=False,
    pad_token_id=2,
    eos_token_id=2
)

llm = HuggingFacePipeline(pipeline=pipe)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
Use the context below to answer the question.

Context:
{context}

Question: {question}

Provide one clear sentence describing the document.
""",
    input_variables=["context", "question"]
)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

response = qa_chain.invoke({"query": "what is this pdf about?"})

print(response["result"])
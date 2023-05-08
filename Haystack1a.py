import streamlit as st
import os
import docx
import haystack
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever,TfidfRetriever,PromptNode, PromptTemplate
from haystack.nodes import FARMReader,TransformersReader
from haystack.pipelines import ExtractiveQAPipeline,DocumentSearchPipeline,GenerativeQAPipeline,Pipeline
from haystack.utils import print_answers
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack import Document
from haystack.utils import print_answers
import pdfplumber
from haystack.nodes import OpenAIAnswerGenerator
from haystack.nodes.prompt import PromptTemplate

document_store = InMemoryDocumentStore(use_bm25=True)
documents=[]
def add_document(document_store, file):
    if file.type == 'text/plain':
        text = file.getvalue().decode("utf-8")
        # document_store.write_documents(dicts)
        # st.write(file.name)
        # st.write(text)
    elif file.type == 'application/pdf':
        with pdfplumber.open(file) as pdf:
            text = "\n\n".join([page.extract_text() for page in pdf.pages])
            # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
            # st.write(text)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
        # st.write(text)
    else:
        st.warning(f"{file.name} is not a supported file format")
    dicts = {
            'content': text,
            'meta': {'name': file.name}
            }  
    documents.append(dicts) 

    
# create Streamlit app
# st.title("click to add documents to Document Storage")
API_KEY = st.secrets['OPENAI_API_KEY']
# API_KEY='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'
# create file uploader
uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

# loop through uploaded files and add them to document store
if uploaded_files:
    for file in uploaded_files:
        add_document(document_store, file)
document_store.write_documents(documents)
# display number of documents in document store
st.write(f"Number of documents in document store: {document_store.get_document_count()}")

if (document_store.get_document_count()!=0):
    question = st.text_input('Ask a question')
    retriever = TfidfRetriever(document_store=document_store)
     # QA pipeline using prompt node 
    if question != '':  
        prompt = PromptTemplate(name="lfqa",
                             prompt_text="""Given the context and the given question,provide a clear and concise response from the relevant information presented in the paragraphs.
                             If the question cannot be answered from the context, reply with 'No relevant information present in attached documents'.
                             \n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:""") 
        node = PromptNode("gpt-3.5-turbo", default_prompt_template=prompt, api_key=API_KEY)#
        candidate_documents = retriever.retrieve(query=question,top_k=2)
        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        output = pipe.run(query=question, documents=candidate_documents)
        st.write(output["results"][0])





        
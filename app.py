import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",  # Add this line
        temperature=0.7,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512  # Remove quotes around 512 (should be an integer)
        }
    )
    return llm


# Custom prompt template for QA
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Below is a context extracted from documents:

{context}

Answer the question based on the context above:

Question: {question}
Answer:
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Load Database
DB_FAISS_PATH = "Data/vectorstores/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Streamlit UI setup
st.title("RAG-based Chatbot")

# Create chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**User:** {chat['user']}")
    st.markdown(f"**Assistant:** {chat['assistant']}")

# Input query box
user_query = st.text_input("Write your question here:")

# If a query is entered, get the response
if user_query:
    response = qa_chain.invoke({'query': user_query})
    assistant_response = response["result"]

    # Add to chat history
    st.session_state.history.append({"user": user_query, "assistant": assistant_response})

    # Display the assistant's response on the left side
    st.markdown(f"**Assistant:** {assistant_response}")

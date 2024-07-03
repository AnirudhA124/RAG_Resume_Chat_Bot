import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from tempfile import NamedTemporaryFile

# Initialize LangChain components
llm = ChatCohere(cohere_api_key="uml0lVi8lxTjTL10Bkb42inOlNFk3zDf7sELxPDN", model="command-r")
prompt = hub.pull("rlm/rag-prompt")

# Function to save and process document
def save_and_process_document(uploaded_file):
    # Save the uploaded PDF file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process the document using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    embeddings_model = CohereEmbeddings(cohere_api_key="uml0lVi8lxTjTL10Bkb42inOlNFk3zDf7sELxPDN",
                                        model="embed-english-light-v3.0")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    db = FAISS.from_documents(splits, embeddings_model)
    retriever = db.as_retriever(kwargs={"score_threshold": 0.5})

    return retriever

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app
def main():
    st.title("RChat Bot")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        retriever = save_and_process_document(uploaded_file)
        
        query = st.text_input("Ask a question about the document:")
        
        if st.button("Ask"):
            if query:
                result = chat(query, retriever)
                st.text_area("Response:", value=result, height=200)
            else:
                st.warning("Please enter a question.")

# Function to handle chat interaction
def chat(query, retriever):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)

if __name__ == "__main__":
    main()

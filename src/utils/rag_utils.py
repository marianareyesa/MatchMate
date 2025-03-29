from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import os
import streamlit as st


api_key = os.environ["OPENAI_API_KEY"]

def extract_and_process_pdf(pdf_file_paths):
    all_data = []
    for pdf_file_path in pdf_file_paths:
        # Extract text from the PDF
        pdf_text = ""
        with open(pdf_file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

        # Create a temporary text file
        temp_txt_filename = pdf_file_path.replace('.pdf', '.txt')
        with open(temp_txt_filename, 'w', encoding='utf-8') as f:
            f.write(pdf_text)

        # Use the extracted text file for further processing
        loader = TextLoader(file_path=temp_txt_filename, encoding="utf-8")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data = text_splitter.split_documents(data)
        all_data.extend(data)
    
    return all_data

def create_conversation_chain(processed_texts, api_key):
    # Set the API key as an environment variable
    # os.environ["OPENAI_API_KEY"] = api_key
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(processed_texts, embedding=embeddings)

    # Create conversation chain
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# TODO: Read from index the names of the pdf files instead of inputting them manually
# Define the list of PDF files
pdf_files = [
    "forehand.pdf",
]

processed_texts = extract_and_process_pdf(pdf_files)
conversation_chain = create_conversation_chain(processed_texts, api_key)

# Open the latest report which is a pdf file
latest_report = "timestamp_report.pdf" #path
raw_latest_report = extract_and_process_pdf([latest_report])

#TODO: have a report that contains the history of the things the player has worked on
history_report = "history_report.pdf" #path
raw_history_report = extract_and_process_pdf([history_report])

query = f"""
You are a professional tennis coach and you will be analyzing a summary report of a player's training session.

Make sure to identify the strengths and weaknesses of the player's technique.
Also take into consideration the player's history of training.

Once all of the background information has been analyzed, provide a detailed plan for the next training session

Latest Report: {raw_latest_report}
History Report: {history_report}
"""

def generate_conversation(user_input, conversation_chain):
    # Check if the user wants to exit the conversation
    if user_input.lower() == "exit":
        return "Goodbye!"

    # Send the user input to the conversation chain
    result = conversation_chain({"question": user_input})

    # Get the answer from the result
    answer = result["answer"]

    # Return the answer directly
    return answer


if __name__ == "__main__":
    generate_conversation(query, conversation_chain)

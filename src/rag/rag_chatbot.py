"""
RAG-based Chatbot for Admissions Data
This script implements a Retrieval-Augmented Generation (RAG) chatbot
using local models (qwen:14b and nomic-embed-text).
"""

import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from tqdm import tqdm

class AdmissionsChatbot:
    def __init__(
        self,
        embedding_model_name: str = "nomic-embed-text:latest",
        llm_model_name: str = "qwen3:14b",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the Admissions Chatbot.
        """
        print("Initializing Admissions Chatbot...")
        
        # Step 1: Set up text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        print("✓ Text splitter initialized")
        
        # Step 2: Set up embedding model
        self.embeddings = OllamaEmbeddings(
            model=embedding_model_name,
            base_url="http://localhost:11434"
        )
        print("✓ Embedding model initialized")
        
        # Step 3: Set up language model
        self.llm = Ollama(
            model=llm_model_name,
            base_url="http://localhost:11434"
        )
        print("✓ Language model initialized")
        
        self.vector_store = None

    def process_section(self, lines: List[str]) -> str:
        """
        Process a section of text, combining heading with its content.
        """
        if not lines:
            return ""
        
        # Clean and join lines
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines:
            return ""
        
        # If first line is a heading, format it properly
        if cleaned_lines[0].isupper() or ':' in cleaned_lines[0]:
            heading = cleaned_lines[0]
            content = ' '.join(cleaned_lines[1:])
            return f"{heading}\n{content}"
        
        return ' '.join(cleaned_lines)
        
    def load_and_process_documents(self, file_path: str, max_documents: int = 100) -> None:
        """
        Load and process the cleaned admissions data.
        """
        print(f"\nLoading documents from {file_path}...")
        
        # Read the cleaned data file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Process the content into meaningful sections
        sections = []
        current_section = []
        
        # Split content into lines and process
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new section
            is_new_section = (
                line[0].isdigit() or  # Starts with number
                line.isupper() or     # All caps (likely heading)
                ':' in line or        # Contains colon
                re.match(r'^[A-Z][a-z]+', line) is not None  # Starts with capitalized word
            )
            
            if is_new_section and current_section:
                # Process and add the previous section
                section_text = self.process_section(current_section)
                if section_text:
                    sections.append(section_text)
                current_section = []
            
            current_section.append(line)
        
        # Add the last section
        if current_section:
            section_text = self.process_section(current_section)
            if section_text:
                sections.append(section_text)
        
        # Limit the number of sections for initial setup
        sections = sections[:max_documents]
        print(f"Processing {len(sections)} sections")
        
        # Split sections into chunks and create Document objects
        print("Splitting sections into chunks...")
        chunks = []
        for section in tqdm(sections, desc="Processing sections"):
            # Only split if section is longer than chunk_size
            if len(section) > self.text_splitter._chunk_size:
                section_chunks = self.text_splitter.create_documents([section])
                chunks.extend(section_chunks)
            else:
                # Create Document object for shorter sections
                chunks.append(Document(page_content=section))
        
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./src/data/chroma_db"
        )
        print("✓ Vector store created and saved")
        
    def setup_qa_chain(self) -> RetrievalQA:
        """
        Set up the question-answering chain.
        """
        prompt_template = """You are a helpful admissions assistant. Use the following pieces of context 
        to answer the question at the end. If you don't know the answer, just say that you don't know, 
        don't try to make up an answer. Be specific and detailed in your response when the information is available.
        
        When answering:
        1. First, identify the relevant information from the context
        2. Then, provide a clear and structured response
        3. If the information is incomplete, mention what specific details are missing
        4. Always cite the source of your information
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get top 5 most relevant documents
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def ask_question(self, question: str) -> Dict:
        """
        Ask a question to the chatbot.
        """
        if not self.vector_store:
            raise ValueError("Please load and process documents first using load_and_process_documents()")
        
        print(f"\nProcessing question: {question}")
        
        qa_chain = self.setup_qa_chain()
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

def main():
    """
    Main function to demonstrate the chatbot usage.
    """
    # Initialize the chatbot
    chatbot = AdmissionsChatbot()
    
    # Load and process the cleaned admissions data
    chatbot.load_and_process_documents(
        "src/data/pu_admissions_combined_cleaned.txt",
        max_documents=100
    )
    
    print("\nChatbot is ready! You can ask questions about admissions.")
    print("Example questions:")
    print("1. What are the admission requirements?")
    print("2. How can I apply for admission?")
    print("3. What documents do I need to submit?")
    
    # Interactive question-answering loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        try:
            result = chatbot.ask_question(question)
            print("\nAnswer:", result["answer"])
            print("\nSource Documents:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n{i}. {doc.page_content[:200]}...")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
"""
Optimized RAG-based Chatbot for Admissions Data
Uses local models with efficient document processing and retrieval.
"""

import os
from typing import List, Dict
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
        persist_directory: str = "./src/data/chroma_db"
    ):
        print("Initializing Admissions Chatbot...")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model_name,
            base_url="http://localhost:11434"
        )
        self.llm = Ollama(
            model=llm_model_name,
            base_url="http://localhost:11434",
            temperature=0.1
        )
        self.persist_directory = persist_directory
        self.vector_store = None
        print("✓ Components initialized")

    def load_prechunked_documents(self, file_path: str) -> None:
        """Load pre-chunked data from a file and create the vector store."""
        print(f"\nLoading pre-chunked data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split on chunk headers or separator lines
        import re
        raw_chunks = re.split(r'Chunk \d+\n|^#{10,}\n', content, flags=re.MULTILINE)
        # Remove empty and whitespace-only chunks
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        print(f"Loaded {len(chunks)} chunks.")
        # Create Document objects
        docs = [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]
        # Create or load vector store
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vector_store.persist()
        print("✓ Vector store ready")

    def setup_qa_chain(self) -> RetrievalQA:
        prompt_template = """You are a helpful admissions assistant. Use the following context to answer the question.\nIf you don't know the answer, say so. Be specific and cite sources.\n\nContext: {context}\n\nQuestion: {question}\n\nProvide a clear, structured answer with these elements:\n1. Direct answer to the question\n2. Supporting details from the context\n3. Source of information\n4. Any missing information\n\nAnswer:"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please load documents first.")
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask_question(self, question: str) -> Dict:
        if not self.vector_store:
            raise ValueError("Please load documents first")
        print(f"\nProcessing question: {question}")
        qa_chain = self.setup_qa_chain()
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

def main():
    chatbot = AdmissionsChatbot()
    chatbot.load_prechunked_documents("src/data/all_chunks.txt")
    print("\nChatbot is ready! Ask questions about admissions.")
    print("Example questions:")
    print("1. What are the admission requirements?")
    print("2. How can I apply for admission?")
    print("3. What documents do I need to submit?")
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ").strip()
        if question.lower() == 'quit':
            break
        try:
            result = chatbot.ask_question(question)
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n{i}. {doc.page_content[:150]}...")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
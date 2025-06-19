from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import os

# 1 Loading your text first
with open("src/data/pu_admissions_combined_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Text length :", len(text))

# 2 Initialize Ollama embedding wrapper
embed = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# 3 Initialize SemanticChunker with Ollama embedding
chunker = SemanticChunker(embeddings=embed)

# 4 Perform semantic chunking
chunks = chunker.split_text(text)

print(f"Generated {len(chunks)} semantic chunks.\n")



# 5 Define output file
file_name = "src/data/all_chunks.txt"

# 6 Open the file for writing
with open(file_name, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"Chunk {i}\n")
        f.write(chunk + '\n')
        f.write('\n' * 5)  # 5 newlines to create large space between chunks
        f.write("#" * 80 + '\n\n')  # separator for clarity

print(f"All {len(chunks)} semantic chunks have been successfully saved in '{file_name}'.")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read the dataset file
with open('example/dataset.txt', 'r', encoding='utf-8') as file:
    entire_text = file.read()

# Create text splitter with appropriate chunk size for conversations
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increased chunk size to accommodate conversation pairs
    chunk_overlap=50,  # Increased overlap to maintain context between chunks
    length_function=len,
    is_separator_regex=False,
)

# Create chunks from the text
texts = text_splitter.create_documents([entire_text])

print(f"Total chunks: {len(texts)}")

# Print first few chunks to verify
for i in range(min(3, len(texts))):
    print(f"\nChunk {i+1}:")
    print(texts[i].page_content)
    print("-" * 50)
    
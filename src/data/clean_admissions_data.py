import pandas as pd
import re

def clean_text(text):
    """
    Clean text by removing extra whitespaces and normalizing spaces.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def clean_admissions_data(input_file, output_file):
    """
    Clean the admissions data file and save the cleaned version.
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to save the cleaned output file
    """
    try:
        # Read the file in chunks to handle large files efficiently
        chunk_size = 10000  # Adjust this based on your system's memory
        
        # Initialize an empty list to store cleaned chunks
        cleaned_chunks = []
        
        # Read and clean the file in chunks
        for chunk in pd.read_csv(input_file, 
                               chunksize=chunk_size,
                               encoding='utf-8',
                               on_bad_lines='warn'):
            
            # Clean each column
            for column in chunk.columns:
                if chunk[column].dtype == 'object':  # Only clean string columns
                    chunk[column] = chunk[column].apply(clean_text)
            
            cleaned_chunks.append(chunk)
        
        # Combine all chunks
        cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        
        # Save the cleaned data
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Successfully cleaned data and saved to {output_file}")
        print(f"Total rows processed: {len(cleaned_df)}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    input_file = "src/data/pu_admissions_combined.txt"
    output_file = "src/data/pu_admissions_combined_cleaned.txt"
    
    clean_admissions_data(input_file, output_file) 
import os
from dotenv import load_dotenv

#Read the .env file and load it's value as environment variables
load_dotenv(override=True)

# --- Set the OpenAI API key from the environment variable ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAF Settings ---
CHUNK_SIZE = 800           #Character length of each chunk
CHUNK_OVERLAP = 200         #Character between between chunks
TOP_K = 5                   #Number of top results to return from the vector database
MIN_SCORE = 0.3              #Minimum similarity score for a result to be considered relevant

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"  #Model used for generating embeddings
CHAT_MODEL = "gpt-4o-mini"                      #Model used for chat completions

# --- Storage ---
CHROMA_DIR = "chroma_db"            #Directory where the Chroma vector database will be stored
COLLECTION_NAME = "knowledge_base"      #Name of the collection in the Chroma vector database

if __name__ == "__main__":
    # Quick self-check. We never print the key itself, only whether it was loaded successfully.
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        print("OpenAI API key loaded successfully.")
    else:
        print("OpenAI API key not found or invalid. Please check your .env file.")
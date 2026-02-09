import os

def initialize_clients():
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec
    """
    Loads API keys from Colab Secrets and initializes OpenAI and Pinecone clients.
    Returns the initialized clients.
    """
    print("\n🔑 Initializing API clients...")
    try:
        # Load OpenAI API Key
        os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
        openai_client = OpenAI()
        print("   - OpenAI client initialized.")

        # Load Pinecone API Key and initialize client
        pinecone_api_key = os.getenv("COLLECTION_NAME")
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        print("   - Pinecone client initialized.")

        print("✅ Clients initialized successfully.")
        return openai_client, pinecone_client

    except Exception as e:
        print(f"An error occurred during client initialization: {e}")
        return None, None

if __name__ == "__main__":
    initialize_clients()
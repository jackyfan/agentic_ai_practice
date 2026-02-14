from utils import initialize_clients
import logging
from openai import APIError
import textwrap
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken

# === Configure Production-Level Logging ===
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def create_mcp_message(sender, content, metadata=None):
    """
    Create a standardized message for the MCP.
    """
    return {
        "protocol_version": "1.0",
        "sender": sender,
        "content": content,
        "metadata": metadata or {}
    }


def call_llm(system_prompt, user_prompt):
    """
    Call the LLM with the given prompt.
    """
    try:
        client, _ = initialize_clients()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the API call: {e}"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_llm(system_prompt, user_prompt, client, temperature=1, json_mode=False):
    """A centralized function to handle all LLM interactions with retries."""
    try:
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        response = client.chat.completions.create(
            model="qwen-plus",
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return f"LLM Error: {e}"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_llm_robust(system_prompt, user_prompt, client, generation_model='qwen-plus', json_mode=False):
    """
    A centralized function to handle all LLM interactions with retries.
    UPGRADE: Now requires the 'client' and 'generation_model' objects to be passed in.
    """
    logging.info("Attempting to call LLM...")
    try:
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        # UPGRADE: Uses the passed-in client and model name for the API call.
        response = client.chat.completions.create(
            model=generation_model,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        logging.info("LLM call successful.")
        return response.choices[0].message.content.strip()
    except APIError as e:
        logging.error(f"OpenAI API Error in call_llm_robust: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred in call_llm_robust: {e}")
        raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, client, embedding_model='text-embedding-v2'):
    """
    Generates embeddings for a single text query with retries.
    """
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=embedding_model)
        return response.data[0].embedding
    except APIError as e:
        logging.error(f"LLM API Error in get_embedding: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_embedding: {e}")
        raise e


def display_mcp(message, title="MCP Message"):
    """Helper function to display MCP messages clearly during the trace."""
    logging.info(f"\n--- {title} (Sender: {message['sender']}) ---")
    # Display content snippet or keys if content is complex
    if isinstance(message['content'], dict):
        logging.info(f"Content Keys: {list(message['content'].keys())}")
    else:
        logging.info(f"Content: {textwrap.shorten(str(message['content']), width=100)}")
    # Display metadata keys
    print(f"Metadata Keys: {list(message['metadata'].keys())}")
    print("-" * (len(title) + 25))


def query_pinecone(query_text, namespace, top_k, index, client, embedding_model):
    """Embeds the query text and searches the specified Pinecone namespace."""
    try:
        query_embedding = get_embedding(query_text,client, embedding_model)
        response = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        return response['matches']
    except Exception as e:
        logging.error(f"Error querying Pinecone (Namespace: {namespace}): {e}")
        return []


def count_tokens(text, model="gpt-5"):
    """Counts the number of tokens in a text string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for models that might not be in the tiktoken registry
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

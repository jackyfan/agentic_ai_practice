from utils import initialize_clients
import time
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
        client,_ = initialize_clients()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the API call: {e}"


def call_llm_robust(system_prompt, user_content, retries=3, delay=5):
    """
    A more robust helper function to call the OpenAI API with retries.
    """
    for i in range(retries):
        try:
            client,_ = initialize_clients()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_content}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed on attempt {i+1}/{retries}. Error: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retries failed.")
            return None

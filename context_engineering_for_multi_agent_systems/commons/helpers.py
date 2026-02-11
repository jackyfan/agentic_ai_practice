

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


def call_llm(system_prompt, user_prompt,client):
    """
    Call the LLM with the given prompt.
    """
    try:
        response =  client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content":system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the API call: {e}"
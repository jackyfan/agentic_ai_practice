from commons.engine import context_engine
from commons.utils import initialize_clients
import logging

config = {
    "index_name": 'genai-mas-mcp-ch3',
    "generation_model": "qwen-plus",
    "embedding_model": "text-embedding-v2",
    "namespace_context": 'ContextLibrary',
    "namespace_knowledge": 'KnowledgeStore'
}

goal = "What are the primary scientific objectives of the Juno mission, and what makes its design unique? Please cite your sources."

logging.info(f"******** Starting Engine for Goal: '{goal}' **********\n")
client, pc = initialize_clients()
result, trace = context_engine(
    goal,
    client=client,
    pc=pc,
    **config  # Unpack the config dictionary into keyword arguments
)
logging.info(result)
logging.info([step for step in trace.steps])

import agents
import logging


class AgentRegistry:
    def __init__(self):
        self.agents = {
            "Researcher": agents.researcher_agent,
            "Writer": agents.writer_agent,
            "Librarian": agents.agent_context_librarian
        }

    def get_agent(self, agent_name, client, index, generation_model,
                  embedding_model, namespace_context, namespace_knowledge):
        agent = self.agents.get(agent_name)
        if not agent:
            logging.error(f"Agent '{agent_name}' not found in registry.")
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        if 'Librarian' in agent_name:
            return lambda mcp_message: agent(mcp_message, client=client, index=index,
                                             embedding_model=embedding_model,
                                             namespace_context=namespace_context
                                             )
        elif 'Researcher' in agent_name:
            return lambda mcp_message: agent(mcp_message, client=client, index=index,
                                             generation_model=generation_model,
                                             embedding_model=embedding_model,
                                             namespace_knowledge=namespace_knowledge
                                             )
        elif 'Writer' in agent_name:
            return lambda mcp_message: agent(mcp_message, client=client, generation_model=generation_model)

        else:
            return agent

    def get_capabilities_description(self):
        """Returns a structured description of the agents for the Planner LLM."""
        return """
    Available Agents and their required inputs.
    CRITICAL: You MUST use the exact input key names provided for each agent.

    1. AGENT: Librarian
       ROLE: Retrieves Semantic Blueprints (style/structure instructions).
       INPUTS:
         - "intent_query": (String) A descriptive phrase of the desired style.
       OUTPUT: The blueprint structure (JSON string).

    2. AGENT: Researcher
       ROLE: Retrieves and synthesizes factual information on a topic.
       INPUTS:
         - "topic_query": (String) The subject matter to research.
       OUTPUT: Synthesized facts (String).

    3. AGENT: Writer
       ROLE: Generates or rewrites content by applying a Blueprint to source material.
       INPUTS:
         - "blueprint": (String/Reference) The style instructions (usually from Librarian).
         - "facts": (String/Reference) Factual information (usually from Researcher).
         - "previous_content": (String/Reference) Existing text for rewriting.
       OUTPUT: The final generated text (String).
    """

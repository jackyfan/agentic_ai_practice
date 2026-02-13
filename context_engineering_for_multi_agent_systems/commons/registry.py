from agents import researcher_agent, writer_agent, agent_context_librarian


class AgentRegistry:
    def __init__(self):
        self.agents = {
            "Researcher": researcher_agent,
            "Writer": writer_agent,
            "Librarian": agent_context_librarian
        }

    def get_agent(self, agent_name):
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
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

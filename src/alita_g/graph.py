from typing import Annotated, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from alita_g.mcp_box import MCPBox


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    mcp_box_path: str
    selected_mcps: List[str]  # Names of selected MCPs


def get_model(model_name: str = "claude-3-5-sonnet-20240620"):
    # Using LangChain's ChatOpenAI but it can also be configured for Anthropic
    # since API keys are available in .bashrc. The paper mentions Sonnet-4.
    # We'll use GPT-4o as a strong default if Sonnet isn't directly available via ChatOpenAI wrapper
    # unless we specifically use ChatAnthropic.
    return ChatOpenAI(model=model_name)


class AlitaGAgent:
    def __init__(self, mcp_box_path: str = "mcp_box.json"):
        self.mcp_box = MCPBox(mcp_box_path)
        self.model = ChatOpenAI(model="gpt-4o") # Master agent model
        self.mcp_box_path = mcp_box_path

    def task_analyzer(self, state: AgentState) -> Dict:
        """Analyzes the task and retrieves relevant MCPs."""
        last_message = state["messages"][-1].content
        relevant_mcps = self.mcp_box.retrieve(last_message, threshold=0.7)

        mcp_names = [item.name for item in relevant_mcps]

        # In a real implementation, we would dynamically load these MCPs as tools.
        # For now, we'll just track them in the state.
        return {"selected_mcps": mcp_names}

    def reasoner(self, state: AgentState) -> Dict:
        """Core reasoning node."""
        messages = state["messages"]
        # Injected information about available MCPs
        if state.get("selected_mcps"):
            mcp_info = "\n".join([f"- {name}" for name in state["selected_mcps"]])
            system_msg = f"You have access to the following specialized MCP tools:\n{mcp_info}"
            if not any(isinstance(m, HumanMessage) and "specialized MCP tools" in m.content for m in messages):
                messages = [HumanMessage(content=system_msg)] + list(messages)

        response = self.model.invoke(messages)
        return {"messages": [response]}

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze", self.task_analyzer)
        workflow.add_node("reason", self.reasoner)

        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "reason")
        workflow.add_edge("reason", END)

        return workflow.compile()


# Export the compiled graph for LangGraph CLI/Studio
agent = AlitaGAgent()
graph = agent.build_graph()

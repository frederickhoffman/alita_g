from typing import Annotated, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages

from alita_g.mcp_box import MCPBox


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    mcp_box_path: str
    selected_mcps: List[str]  # Names of selected MCPs


class AlitaGAgent:
    def __init__(
        self,
        mcp_box: Optional[MCPBox] = None,
        model: Optional[ChatOpenAI] = None,
        mcp_box_path: str = "mcp_box.json",
    ):
        self.mcp_box = mcp_box or MCPBox(mcp_box_path)
        self.model = model or ChatOpenAI(model="gpt-4o")
        self.mcp_box_path = mcp_box_path

    def task_analyzer(self, state: AgentState) -> Dict:
        """Analyzes the task and retrieves relevant MCPs from the MCP Box."""
        last_message = state["messages"][-1].content
        if not isinstance(last_message, str):
            # Fallback if content is a list of blocks
            last_message = str(last_message)

        # Dynamic retrieval based on paper's recommended threshold
        relevant_mcps = self.mcp_box.retrieve(last_message, threshold=0.7)

        mcp_names = [item.name for item in relevant_mcps]
        return {"selected_mcps": mcp_names}

    def reasoner(self, state: AgentState) -> Dict:
        """Core reasoning node that leverages selected MCP tools."""
        messages = state["messages"]
        selected_mcps = state.get("selected_mcps", [])

        if selected_mcps:
            mcp_info = "\n".join([f"- {name}" for name in selected_mcps])
            system_msg = (
                f"You have access to the following specialized MCP tools:\n{mcp_info}\n"
                "Incorporate their logic into your reasoning to provide a precise answer."
            )
            # Avoid duplicate system messages
            has_mcp_info = any(
                isinstance(m, HumanMessage) and "specialized MCP tools" in m.content
                for m in messages
            )
            if not has_mcp_info:
                messages = [HumanMessage(content=system_msg)] + list(messages)

        response = self.model.invoke(messages)
        return {"messages": [response]}

    def build_graph(self) -> CompiledGraph:
        """Compiles the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze", self.task_analyzer)
        workflow.add_node("reason", self.reasoner)

        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "reason")
        workflow.add_edge("reason", END)

        return workflow.compile()


# Default instance for LangGraph CLI/Studio
agent = AlitaGAgent()
graph = agent.build_graph()

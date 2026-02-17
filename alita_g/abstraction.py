from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from alita_g.mcp_box import MCPItem


class MCPToolSchema(BaseModel):
    name: str = Field(description="The name of the generalized tool")
    description: str = Field(description="Functional description of the tool")
    use_case: str = Field(description="Concise use case summary")
    code: str = Field(description="The generalized Python code for the tool")


class MCPAbstractor:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = ChatOpenAI(model=model_name)
        self.structured_model = self.model.with_structured_output(MCPToolSchema)

    def abstract(self, raw_code: str, task_context: str) -> MCPItem:
        """
        Abstracts raw code into a generalized MCP tool using structured output.
        """
        prompt = f"""
        You are an expert software engineer.
        Abstract the following Python code into a generalized, reusable MCP tool.

        Raw Code:
        ```python
        {raw_code}
        ```

        Task Context:
        {task_context}

        Requirements:
        1. Replace hard-coded values with configurable parameters.
        2. Remove task-specific references.
        3. Standardize the interface using the FastMCP pattern (decorator style).
        4. Add comprehensive docstrings and type annotations.
        5. Provide a functional description and a concise use case summary.
        """

        try:
            result = self.structured_model.invoke(prompt)
            if result is None:
                raise ValueError("Abstraction returned None")

            # Explicit typing for mypy since structured_output can return Dict or Pydantic
            typed_result: MCPToolSchema = result  # type: ignore
            return MCPItem(
                name=typed_result.name,
                code=typed_result.code,
                description=typed_result.description,
                use_case=typed_result.use_case,
            )
        except Exception as e:
            print(f"Error during abstraction: {e}")
            # Fallback to a basic item if abstraction fails
            return MCPItem(
                name="unknown_tool",
                code=raw_code,
                description="Auto-generated tool",
                use_case=task_context,
            )

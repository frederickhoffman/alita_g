from langchain_openai import ChatOpenAI

from alita_g.mcp_box import MCPItem


class MCPAbstractor:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")

    def abstract(self, raw_code: str, task_context: str) -> MCPItem:
        """
        Abstracts raw code into a generalized MCP tool.
        Performs:
        - Parameter Generalization
        - Context Removal
        - Interface Standardization (FastMCP)
        - Documentation Enhancement
        """
        prompt = f"""
        You are an expert software engineer. Abstract the following Python code into a generalized, reusable MCP tool.
        
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
        
        Output format:
        NAME: [tool_name]
        DESCRIPTION: [functional_description]
        USE_CASE: [use_case_summary]
        CODE:
        ```python
        [abstracted_code]
        ```
        """

        response = self.model.invoke(prompt).content

        # Simple parsing logic
        try:
            name = response.split("NAME:")[1].split("DESCRIPTION:")[0].strip()
            description = response.split("DESCRIPTION:")[1].split("USE_CASE:")[0].strip()
            use_case = response.split("USE_CASE:")[1].split("CODE:")[0].strip()
            code = response.split("```python")[1].split("```")[0].strip()

            return MCPItem(name=name, code=code, description=description, use_case=use_case)
        except Exception as e:
            print(f"Error parsing abstraction response: {e}")
            # Fallback to a basic item if parsing fails
            return MCPItem(name="unknown_tool", code=raw_code, description="Auto-generated tool", use_case=task_context)

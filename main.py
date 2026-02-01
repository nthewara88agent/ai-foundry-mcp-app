"""
Azure AI Foundry + MCP Integration

A Python application that uses Azure AI Foundry (GPT-5) with
Microsoft Learn MCP Server for tool calling.
"""

import os
import json
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
)
from azure.identity import DefaultAzureCredential

from mcp_client import MCPClient

load_dotenv()


class AIFoundryMCPAgent:
    """Agent that combines Azure AI Foundry with MCP tools."""

    def __init__(self):
        # Azure AI Foundry setup
        endpoint = os.getenv("AZURE_AI_ENDPOINT")
        deployment = os.getenv("AZURE_AI_DEPLOYMENT", "gpt-5.1")
        
        if not endpoint:
            raise ValueError("AZURE_AI_ENDPOINT environment variable is required")

        # Use DefaultAzureCredential for authentication
        credential = DefaultAzureCredential()
        
        self.ai_client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
        )
        self.deployment = deployment

        # MCP Server setup
        mcp_url = os.getenv("MCP_SERVER_URL", "https://learn.microsoft.com/api/mcp")
        self.mcp_client = MCPClient(mcp_url)
        self.mcp_client.initialize()
        self.mcp_client.list_tools()

        # Conversation history
        self.messages = []

    def _get_tools(self) -> list[ChatCompletionsToolDefinition]:
        """Get MCP tools formatted for Azure AI."""
        tools = []
        for tool in self.mcp_client._tools:
            tools.append(
                ChatCompletionsToolDefinition(
                    function=FunctionDefinition(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=tool.get("inputSchema", {"type": "object", "properties": {}})
                    )
                )
            )
        return tools

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute an MCP tool and return the result."""
        print(f"  🔧 Calling tool: {tool_name}")
        print(f"     Arguments: {json.dumps(arguments, indent=2)}")
        
        result = self.mcp_client.call_tool(tool_name, arguments)
        content = self.mcp_client._extract_text_content(result)
        
        # Truncate for display
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"     Result preview: {preview}\n")
        
        return content

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, with tool calling support."""
        # Add user message to history
        self.messages.append(UserMessage(content=user_message))

        # System message for context
        system_msg = SystemMessage(content="""You are a helpful AI assistant with access to Microsoft Learn documentation.
Use the available tools to search and fetch official Microsoft documentation when users ask about:
- Azure services and configurations
- .NET, C#, Python SDKs
- Microsoft 365, Power Platform
- Developer tools and best practices

Always cite your sources with documentation URLs when providing information from Microsoft docs.""")

        # Get completion with tools
        tools = self._get_tools()
        
        response = self.ai_client.complete(
            model=self.deployment,
            messages=[system_msg] + self.messages,
            tools=tools if tools else None,
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        while assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.messages.append(AssistantMessage(
                content=assistant_message.content or "",
                tool_calls=assistant_message.tool_calls
            ))

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                result = self._execute_tool(tool_name, arguments)
                
                # Add tool result to messages
                self.messages.append(ToolMessage(
                    tool_call_id=tool_call.id,
                    content=result
                ))

            # Get next response
            response = self.ai_client.complete(
                model=self.deployment,
                messages=[system_msg] + self.messages,
                tools=tools,
            )
            assistant_message = response.choices[0].message

        # Add final assistant message
        final_content = assistant_message.content or ""
        self.messages.append(AssistantMessage(content=final_content))

        return final_content

    def close(self):
        """Clean up resources."""
        self.mcp_client.close()


def main():
    """Interactive chat loop."""
    print("=" * 60)
    print("Azure AI Foundry + Microsoft Learn MCP Agent")
    print("=" * 60)
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

    agent = AIFoundryMCPAgent()
    
    # Show available tools
    print("📚 Available MCP Tools:")
    for tool in agent.mcp_client._tools:
        print(f"   - {tool['name']}: {tool.get('description', 'N/A')[:60]}...")
    print()

    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                agent.messages = []
                print("Conversation cleared.\n")
                continue

            print("\n🤖 Processing...\n")
            
            try:
                response = agent.chat(user_input)
                print(f"Assistant: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")

    finally:
        agent.close()


if __name__ == "__main__":
    main()

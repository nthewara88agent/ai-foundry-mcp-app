"""
Microsoft Learn MCP Client

Implements Streamable HTTP transport for the Microsoft Learn MCP Server.
Based on MCP specification for HTTP+SSE transport.
"""

import json
import uuid
from typing import Any
import httpx


class MCPClient:
    """Client for Microsoft Learn MCP Server using Streamable HTTP transport."""

    def __init__(self, server_url: str = "https://learn.microsoft.com/api/mcp"):
        self.server_url = server_url.rstrip("/")
        self.session_id = None
        self.client = httpx.Client(timeout=60.0)
        self._tools: list[dict] = []

    def _make_request(self, method: str, params: dict | None = None) -> dict:
        """Make a JSON-RPC request to the MCP server."""
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = self.client.post(self.server_url, json=payload, headers=headers)
        response.raise_for_status()

        # Check for session ID in response headers
        if "Mcp-Session-Id" in response.headers:
            self.session_id = response.headers["Mcp-Session-Id"]

        # Handle SSE response
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse_response(response.text)

        return response.json()

    def _parse_sse_response(self, sse_text: str) -> dict:
        """Parse SSE response and extract JSON-RPC result."""
        result = {}
        for line in sse_text.split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data.strip():
                    try:
                        parsed = json.loads(data)
                        if "result" in parsed:
                            result = parsed
                    except json.JSONDecodeError:
                        continue
        return result

    def initialize(self) -> dict:
        """Initialize the MCP session."""
        result = self._make_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "ai-foundry-mcp-client",
                "version": "1.0.0"
            }
        })
        # Send initialized notification
        self._make_request("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        result = self._make_request("tools/list")
        if "result" in result and "tools" in result["result"]:
            self._tools = result["result"]["tools"]
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server."""
        result = self._make_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return result.get("result", result)

    def search_docs(self, query: str) -> str:
        """Search Microsoft documentation."""
        result = self.call_tool("microsoft_docs_search", {"query": query})
        return self._extract_text_content(result)

    def fetch_doc(self, url: str) -> str:
        """Fetch a specific documentation page."""
        result = self.call_tool("microsoft_docs_fetch", {"url": url})
        return self._extract_text_content(result)

    def search_code_samples(self, query: str, language: str | None = None) -> str:
        """Search for code samples."""
        args = {"query": query}
        if language:
            args["language"] = language
        result = self.call_tool("microsoft_code_sample_search", args)
        return self._extract_text_content(result)

    def _extract_text_content(self, result: dict) -> str:
        """Extract text content from MCP tool result."""
        if isinstance(result, dict) and "content" in result:
            contents = result["content"]
            text_parts = []
            for item in contents:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)
        return str(result)

    def get_tools_for_llm(self) -> list[dict]:
        """Get tools formatted for Azure AI Inference SDK."""
        if not self._tools:
            self.list_tools()
        
        formatted_tools = []
        for tool in self._tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            })
        return formatted_tools

    def close(self):
        """Close the client connection."""
        self.client.close()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

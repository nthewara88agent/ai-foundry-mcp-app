# Azure AI Foundry + MCP Python Application

A Python application that integrates **Azure AI Foundry** (GPT-5) with the **Microsoft Learn MCP Server** for intelligent documentation search and retrieval.

## Features

- 🤖 **Azure AI Foundry Integration** - Uses GPT-5.1 model via Azure AI Inference SDK
- 📚 **Microsoft Learn MCP Server** - Access official Microsoft documentation
- 🔧 **Tool Calling** - Automatic tool invocation for documentation queries
- 💬 **Interactive Chat** - Conversational interface with memory

## Prerequisites

- Python 3.10+
- Azure CLI logged in (`az login`)
- Access to Azure AI Foundry with GPT-5 deployed

## Setup

1. **Create virtual environment:**
   ```bash
   cd projects/ai-foundry-mcp-app
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure AI Foundry endpoint
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_AI_ENDPOINT` | Azure AI Foundry endpoint URL | Required |
| `AZURE_AI_DEPLOYMENT` | Model deployment name | `gpt-5.1` |
| `MCP_SERVER_URL` | Microsoft Learn MCP endpoint | `https://learn.microsoft.com/api/mcp` |

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `microsoft_docs_search` | Semantic search across Microsoft documentation |
| `microsoft_docs_fetch` | Fetch full content from a documentation URL |
| `microsoft_code_sample_search` | Search for official code samples |

## Example Usage

```
You: How do I create an Azure Function with Python?

🤖 Processing...

  🔧 Calling tool: microsoft_docs_search
     Arguments: {"query": "create Azure Function Python"}
     Result preview: ...
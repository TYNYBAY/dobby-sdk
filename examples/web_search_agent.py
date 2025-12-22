#!/usr/bin/env python
"""CLI agent with web search.

A simple streaming agent that can search the web using Tavily.
Tool calls are logged to the CLI.

Usage:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
    export TAVILY_API_KEY=tvly-...
    python web_search_agent.py "What are the latest AI news?"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from dobby import AgentExecutor, OpenAIProvider
from dobby.common_tools import TavilySearchTool
from dobby.types import (
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolUseEndEvent,
    ToolUseEvent,
    UserMessagePart,
)

# Load environment variables from .env file
load_dotenv()


# ANSI colors for CLI
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


async def run_agent(query: str) -> None:
    """Run the web search agent with streaming output."""

    # Get API keys from environment
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not azure_key:
        print(f"{Colors.RED}Error: AZURE_OPENAI_API_KEY not set{Colors.RESET}")
        sys.exit(1)

    if not azure_endpoint:
        print(f"{Colors.RED}Error: AZURE_OPENAI_ENDPOINT not set{Colors.RESET}")
        sys.exit(1)

    if not tavily_key:
        print(f"{Colors.RED}Error: TAVILY_API_KEY not set{Colors.RESET}")
        sys.exit(1)

    # Initialize Azure OpenAI provider
    provider = OpenAIProvider(
        base_url=azure_endpoint,
        azure_deployment_id=azure_deployment,
        api_key=azure_key,
    )

    search_tool = TavilySearchTool(api_key=tavily_key)

    # Create executor with tools
    executor = AgentExecutor(
        provider="azure-openai",
        llm=provider,
        tools=[search_tool],
    )

    # Create user message
    messages = [UserMessagePart(parts=[TextPart(text=query)])]

    system_prompt = """You are a helpful research assistant with web search capabilities.
When the user asks a question, use the tavily_search tool to find relevant information.
After searching, provide a comprehensive answer based on the results.
Always cite your sources by mentioning the URLs."""

    print(f"\n{Colors.BOLD}Query:{Colors.RESET} {query}\n")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}\n")

    try:
        # Stream the response
        async for event in executor.run_stream(
            messages,
            system_prompt=system_prompt,
            reasoning_effort="low",
        ):
            match event:
                case StreamStartEvent(model=model):
                    print(f"{Colors.DIM}Model: {model}{Colors.RESET}\n")

                case TextDeltaEvent(delta=delta):
                    print(delta, end="", flush=True)

                case ReasoningStartEvent():
                    print(f"\n{Colors.BLUE}💭 Reasoning:{Colors.RESET} ", end="")

                case ReasoningDeltaEvent(delta=delta):
                    print(f"{Colors.DIM}{delta}{Colors.RESET}", end="", flush=True)

                case ReasoningEndEvent():
                    print("\n")  # End reasoning block

                case ToolUseEvent(name=name, inputs=inputs):
                    print(
                        f"\n\n{Colors.YELLOW}🔧 Tool Call:{Colors.RESET} "
                        f"{Colors.BOLD}{name}{Colors.RESET}"
                    )
                    print(f"{Colors.DIM}   Inputs: {inputs}{Colors.RESET}\n")

                case ToolResultEvent(name=name, result=result):
                    output = str(result)
                    if len(output) > 200:
                        output = output[:200] + "..."
                    print(f"{Colors.GREEN}✓ Tool Result ({name}):{Colors.RESET}")
                    print(f"{Colors.DIM}   {output}{Colors.RESET}\n")

                case ToolUseEndEvent():
                    pass  # Tool finished, result already shown

                case StreamEndEvent(usage=usage):
                    if usage:
                        print(f"\n\n{Colors.DIM}{'─' * 50}{Colors.RESET}")
                        print(
                            f"{Colors.DIM}Tokens: {usage.total_tokens} "
                            f"(in: {usage.input_tokens}, "
                            f"out: {usage.output_tokens}){Colors.RESET}"
                        )

    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        raise


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} \"<your query>\"")
        print("\nExample:")
        print(f'  python {sys.argv[0]} "What are the latest developments in AI?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(run_agent(query))


if __name__ == "__main__":
    main()

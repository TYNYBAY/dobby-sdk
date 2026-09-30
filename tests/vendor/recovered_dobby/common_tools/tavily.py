"""Tavily web search tool.

Uses Tavily API to search the web for information.
Requires tavily-python package: pip install tavily-python
"""

from typing import Annotated, Literal

from pydantic import BaseModel

from dobby.tools import Tool

try:
    from tavily import AsyncTavilyClient
except ImportError as e:
    raise ImportError(
        "Please install 'tavily-python' to use the Tavily search tool: pip install tavily-python"
    ) from e


class TavilySearchResult(BaseModel):
    """A Tavily search result.

    Attributes:
        title: The title of the search result.
        url: The URL of the search result.
        content: A short description/snippet of the content.
        score: The relevance score of the result.
    """

    title: str
    url: str
    content: str
    score: float


class TavilySearchTool(Tool):
    """Tavily web search tool.

    Searches the web using Tavily API and returns structured results.

    Usage:
        tool = TavilySearchTool(api_key="your-api-key")
        agent = AgentExecutor(tools=[tool], ...)

    Attributes:
        api_key: Your Tavily API key from https://app.tavily.com/home
    """

    name = "tavily_search"
    description = "Search the web using Tavily and return relevant results"

    def __init__(self, api_key: str):
        """Initialize the Tavily search tool.

        Args:
            api_key: Your Tavily API key.
        """
        self.api_key = api_key
        self.client = AsyncTavilyClient(api_key)

    async def __call__(
        self,
        query: Annotated[str, "The search query to execute"],
        search_depth: Annotated[
            Literal["basic", "advanced"],
            "Depth of search - 'basic' for quick results, 'advanced' for more thorough search",
        ] = "basic",
        topic: Annotated[
            Literal["general", "news"],
            "Category of search - 'general' for web search, 'news' for recent news",
        ] = "general",
        time_range: Annotated[
            Literal["day", "week", "month", "year"] | None,
            "Time range to filter results - None for no time filter",
        ] = None,
    ) -> list[TavilySearchResult]:
        """Search the web using Tavily.

        Args:
            query: The search query to execute.
            search_depth: Depth of search ('basic' or 'advanced').
            topic: Category of search ('general' or 'news').
            time_range: Time range filter (None, 'day', 'week', 'month', 'year').

        Returns:
            A list of search results with title, URL, content, and score.
        """
        response = await self.client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,  # type: ignore[reportUnknownMemberType]
        )
        return [
            TavilySearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
            )
            for result in response.get("results", [])
        ]

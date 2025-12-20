"""Common pre-built tools for Dobby agents.

This module provides ready-to-use tools that can be added to any agent.
"""

from .tavily import TavilySearchResult, TavilySearchTool

__all__ = ["TavilySearchResult", "TavilySearchTool"]

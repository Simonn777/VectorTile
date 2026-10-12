"""Exhaustive top-k retrieval with explicit, reproducible backends."""

from .search import BackendUnavailableError, SearchResult, search

__all__ = ["BackendUnavailableError", "SearchResult", "search"]
__version__ = "0.1.0"

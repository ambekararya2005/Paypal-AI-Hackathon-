"""Provider-generic LLM chat layer with a committed disk cache (HARD RULE 3).

Used by the shopping agent, the L3 injection judge and the mandate compiler.
Sits directly above ``core`` in the dependency order, so ``defense`` and ``agent``
can both import it without importing each other. The provider SDK is imported
lazily, only on a live cache miss.
"""

from mantis.llm.client import CacheMissError, ChatResult, LlmChat, LlmConfigError, ToolCall

__all__ = ["CacheMissError", "ChatResult", "LlmChat", "LlmConfigError", "ToolCall"]

"""Declare and export a tool collection for Anthropic and OpenAI APIs.

A :class:`ToolManifest` is a registry of :class:`ToolDefinition` objects.
Each definition carries the tool name, description, JSON input schema, and
optional tags.  The manifest can export the collection in Anthropic or
OpenAI wire format.

Example::

    from llm_tool_manifest import ToolManifest, ToolDefinition

    manifest = ToolManifest()

    manifest.add(ToolDefinition(
        name="search",
        description="Search the web for a query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        tags=["web", "read"],
    ))

    # Export for Anthropic API
    tools = manifest.to_anthropic()

    # Export for OpenAI API
    tools = manifest.to_openai()

    # Filter by tag
    read_tools = manifest.filter_by_tag("read")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolManifestError(Exception):
    """Raised for invalid tool definitions or manifest operations."""


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
    """A single tool declaration.

    Attributes:
        name:         Tool name (must be non-empty, no spaces).
        description:  Human-readable description.
        input_schema: JSON Schema dict for the tool's input.  Must have
                      ``"type": "object"``.
        tags:         Optional list of string labels.
        metadata:     Arbitrary key/value store for application use.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ToolManifestError("ToolDefinition.name must be non-empty")
        if " " in self.name:
            raise ToolManifestError(
                f"ToolDefinition.name must not contain spaces: {self.name!r}"
            )

    def has_tag(self, tag: str) -> bool:
        """Return ``True`` if *tag* is in this tool's tag list."""
        return tag in self.tags

    def required_fields(self) -> list[str]:
        """Return the ``required`` list from the input schema (or empty list)."""
        return list(self.input_schema.get("required", []))

    def property_names(self) -> list[str]:
        """Return sorted list of property names from the input schema."""
        props = self.input_schema.get("properties", {})
        return sorted(props.keys())

    def to_anthropic(self) -> dict[str, Any]:
        """Return an Anthropic-format tool dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict[str, Any]:
        """Return an OpenAI ``function``-style tool dict."""
        func: dict[str, Any] = {
            "name": self.name,
            "parameters": self.input_schema,
        }
        if self.description:
            func["description"] = self.description
        return {"type": "function", "function": func}

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# ToolManifest
# ---------------------------------------------------------------------------


class ToolManifest:
    """Registry of :class:`ToolDefinition` objects.

    Example::

        manifest = ToolManifest(name="my-agent-tools", version="1.0.0")
        manifest.add(ToolDefinition(name="search", description="Web search"))

        # Export
        manifest.to_anthropic()   # list of Anthropic tool dicts
        manifest.to_openai()      # list of OpenAI tool dicts

        # Query
        manifest.get("search")
        manifest.filter_by_tag("web")
        manifest.names()
    """

    def __init__(
        self,
        name: str = "",
        version: str = "0.1.0",
        description: str = "",
    ) -> None:
        self._name = name
        self._version = version
        self._description = description
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, tool: ToolDefinition) -> None:
        """Register *tool* in the manifest.

        Raises:
            ToolManifestError: if a tool with the same name is already
                               registered.
        """
        if tool.name in self._tools:
            raise ToolManifestError(
                f"Tool {tool.name!r} is already registered in the manifest"
            )
        self._tools[tool.name] = tool

    def add_or_replace(self, tool: ToolDefinition) -> None:
        """Register *tool*, replacing any existing tool with the same name."""
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        """Remove the tool named *name*.

        Raises:
            KeyError: if the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} not found in manifest")
        del self._tools[name]

    def clear(self) -> None:
        """Remove all tools from the manifest."""
        self._tools.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition:
        """Return the :class:`ToolDefinition` for *name*.

        Raises:
            KeyError: if not found.
        """
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} not found in manifest")
        return self._tools[name]

    def contains(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._tools

    def names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools)

    def tools(self) -> list[ToolDefinition]:
        """Return all registered tools (sorted by name)."""
        return [self._tools[n] for n in self.names()]

    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def filter_by_tag(self, tag: str) -> list[ToolDefinition]:
        """Return all tools that have *tag* in their tag list."""
        return [t for t in self.tools() if t.has_tag(tag)]

    def filter_by_tags(self, tags: list[str]) -> list[ToolDefinition]:
        """Return tools that have ALL of the given *tags*."""
        tag_set = set(tags)
        return [t for t in self.tools() if tag_set.issubset(set(t.tags))]

    def all_tags(self) -> list[str]:
        """Return sorted list of all distinct tags across all tools."""
        tags: set[str] = set()
        for t in self._tools.values():
            tags.update(t.tags)
        return sorted(tags)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_anthropic(self, *, tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Export tools in Anthropic API format.

        Args:
            tags: If provided, export only tools that have ALL listed tags.
        """
        tools = self.filter_by_tags(tags) if tags else self.tools()
        return [t.to_anthropic() for t in tools]

    def to_openai(self, *, tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Export tools in OpenAI API format.

        Args:
            tags: If provided, export only tools that have ALL listed tags.
        """
        tools = self.filter_by_tags(tags) if tags else self.tools()
        return [t.to_openai() for t in tools]

    def to_dict(self) -> dict[str, Any]:
        """Return a full manifest dict."""
        return {
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "tool_count": self.count(),
            "tools": {n: t.to_dict() for n, t in self._tools.items()},
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation warnings for all tools.

        An empty list means the manifest is clean.
        """
        warnings: list[str] = []
        for t in self.tools():
            schema = t.input_schema
            if schema.get("type") != "object":
                warnings.append(
                    f"Tool {t.name!r}: input_schema.type should be 'object'"
                )
            if not t.description:
                warnings.append(f"Tool {t.name!r}: description is empty")
        return warnings

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return (
            f"ToolManifest(name={self._name!r}, tools={self.count()}, "
            f"version={self._version!r})"
        )

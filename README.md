# llm-tool-manifest

Declare and export a tool collection for Anthropic and OpenAI APIs. Zero dependencies.

Define your tools once as a manifest, then export them in the format each provider expects. Supports tag-based filtering, validation, and introspection.

## Install

```bash
pip install llm-tool-manifest
```

## Usage

```python
from llm_tool_manifest import ToolManifest, ToolDefinition

manifest = ToolManifest(name="my-agent-tools", version="1.0.0")

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

manifest.add(ToolDefinition(
    name="write_file",
    description="Write content to a file.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    tags=["fs", "write"],
))

# Export for Anthropic
client.messages.create(
    model="claude-sonnet-4-6",
    tools=manifest.to_anthropic(),
    ...
)

# Export for OpenAI
client.chat.completions.create(
    model="gpt-4",
    tools=manifest.to_openai(),
    ...
)

# Filter by tag
read_tools = manifest.to_anthropic(tags=["read"])
```

## Querying

```python
manifest.get("search")                    # ToolDefinition
manifest.contains("search")              # True/False
manifest.names()                          # sorted list
manifest.filter_by_tag("web")            # list of ToolDefinitions
manifest.filter_by_tags(["web", "read"]) # tools with ALL tags
manifest.all_tags()                       # sorted list of all tags
manifest.count()                          # number of tools
```

## Validation

```python
warnings = manifest.validate()
# Returns list of warning strings:
# - "description is empty"
# - "input_schema.type should be 'object'"
```

## Serialisation

```python
d = manifest.to_dict()
# {name, version, description, tool_count, tools: {...}}
```

## ToolDefinition fields

| Field | Description |
|-------|-------------|
| `name` | Tool name (no spaces, required) |
| `description` | Human-readable description |
| `input_schema` | JSON Schema dict (`"type": "object"`) |
| `tags` | Optional list of string labels |
| `metadata` | Arbitrary key/value store |

## License

MIT

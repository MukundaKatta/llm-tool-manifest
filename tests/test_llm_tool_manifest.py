"""Tests for llm_tool_manifest."""

from __future__ import annotations

import pytest

from llm_tool_manifest import ToolDefinition, ToolManifest, ToolManifestError

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


def make_search():
    return ToolDefinition(
        name="search",
        description="Search the web.",
        input_schema=SEARCH_SCHEMA,
        tags=["web", "read"],
    )


def make_read():
    return ToolDefinition(
        name="read_file",
        description="Read a file.",
        input_schema=READ_SCHEMA,
        tags=["fs", "read"],
    )


def make_write():
    return ToolDefinition(
        name="write_file",
        description="Write a file.",
        input_schema={"type": "object"},
        tags=["fs", "write"],
    )


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


def test_definition_basic():
    t = make_search()
    assert t.name == "search"
    assert t.description == "Search the web."
    assert t.tags == ["web", "read"]


def test_definition_empty_name_raises():
    with pytest.raises(ToolManifestError):
        ToolDefinition(name="")


def test_definition_name_with_space_raises():
    with pytest.raises(ToolManifestError):
        ToolDefinition(name="my tool")


def test_definition_has_tag():
    t = make_search()
    assert t.has_tag("web") is True
    assert t.has_tag("write") is False


def test_definition_required_fields():
    t = make_search()
    assert t.required_fields() == ["query"]


def test_definition_required_fields_empty():
    t = ToolDefinition(name="noop")
    assert t.required_fields() == []


def test_definition_property_names():
    t = make_search()
    assert t.property_names() == ["query"]


def test_definition_to_dict():
    t = make_search()
    d = t.to_dict()
    assert d["name"] == "search"
    assert d["tags"] == ["web", "read"]
    assert "input_schema" in d


def test_definition_defaults():
    t = ToolDefinition(name="t")
    assert t.description == ""
    assert t.input_schema == {"type": "object"}
    assert t.tags == []
    assert t.metadata == {}


def test_definition_metadata():
    t = ToolDefinition(name="t", metadata={"cost": 0.01})
    assert t.metadata["cost"] == 0.01


# ---------------------------------------------------------------------------
# ToolDefinition — to_anthropic
# ---------------------------------------------------------------------------


def test_to_anthropic_structure():
    t = make_search()
    d = t.to_anthropic()
    assert d["name"] == "search"
    assert d["description"] == "Search the web."
    assert d["input_schema"] == SEARCH_SCHEMA


def test_to_anthropic_no_tags():
    t = make_search()
    d = t.to_anthropic()
    assert "tags" not in d


# ---------------------------------------------------------------------------
# ToolDefinition — to_openai
# ---------------------------------------------------------------------------


def test_to_openai_structure():
    t = make_search()
    d = t.to_openai()
    assert d["type"] == "function"
    func = d["function"]
    assert func["name"] == "search"
    assert func["description"] == "Search the web."
    assert func["parameters"] == SEARCH_SCHEMA


def test_to_openai_no_description_omitted():
    t = ToolDefinition(name="t", input_schema={"type": "object"})
    d = t.to_openai()
    assert "description" not in d["function"]


def test_to_openai_no_tags():
    t = make_search()
    d = t.to_openai()
    assert "tags" not in d["function"]


# ---------------------------------------------------------------------------
# ToolManifest — empty state
# ---------------------------------------------------------------------------


def test_empty_manifest():
    m = ToolManifest()
    assert m.count() == 0
    assert m.names() == []
    assert m.tools() == []
    assert len(m) == 0


def test_repr_empty():
    m = ToolManifest(name="test", version="1.0")
    r = repr(m)
    assert "test" in r
    assert "tools=0" in r


def test_manifest_name_version():
    m = ToolManifest(name="my-agent", version="2.0.0")
    d = m.to_dict()
    assert d["name"] == "my-agent"
    assert d["version"] == "2.0.0"


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------


def test_add_tool():
    m = ToolManifest()
    m.add(make_search())
    assert m.count() == 1
    assert "search" in m


def test_add_duplicate_raises():
    m = ToolManifest()
    m.add(make_search())
    with pytest.raises(ToolManifestError, match="already registered"):
        m.add(make_search())


def test_add_or_replace():
    m = ToolManifest()
    m.add(make_search())
    new = ToolDefinition(name="search", description="Updated")
    m.add_or_replace(new)
    assert m.get("search").description == "Updated"


def test_remove_tool():
    m = ToolManifest()
    m.add(make_search())
    m.remove("search")
    assert "search" not in m


def test_remove_nonexistent_raises():
    m = ToolManifest()
    with pytest.raises(KeyError):
        m.remove("nope")


def test_clear():
    m = ToolManifest()
    m.add(make_search())
    m.add(make_read())
    m.clear()
    assert m.count() == 0


# ---------------------------------------------------------------------------
# contains / get
# ---------------------------------------------------------------------------


def test_contains():
    m = ToolManifest()
    m.add(make_search())
    assert m.contains("search") is True
    assert m.contains("nope") is False


def test_dunder_contains():
    m = ToolManifest()
    m.add(make_search())
    assert "search" in m
    assert "nope" not in m


def test_get_returns_tool():
    m = ToolManifest()
    t = make_search()
    m.add(t)
    assert m.get("search") is t


def test_get_missing_raises():
    m = ToolManifest()
    with pytest.raises(KeyError):
        m.get("nope")


# ---------------------------------------------------------------------------
# names / tools / count
# ---------------------------------------------------------------------------


def test_names_sorted():
    m = ToolManifest()
    m.add(make_write())
    m.add(make_search())
    m.add(make_read())
    assert m.names() == ["read_file", "search", "write_file"]


def test_tools_sorted():
    m = ToolManifest()
    m.add(make_write())
    m.add(make_search())
    names = [t.name for t in m.tools()]
    assert names == ["search", "write_file"]


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def test_filter_by_tag():
    m = ToolManifest()
    m.add(make_search())
    m.add(make_read())
    m.add(make_write())
    read_tools = m.filter_by_tag("read")
    names = {t.name for t in read_tools}
    assert names == {"search", "read_file"}


def test_filter_by_tag_none():
    m = ToolManifest()
    m.add(make_search())
    assert m.filter_by_tag("nonexistent") == []


def test_filter_by_tags_all_required():
    m = ToolManifest()
    m.add(make_search())   # tags: web, read
    m.add(make_read())     # tags: fs, read
    # Only search has both "web" and "read"
    result = m.filter_by_tags(["web", "read"])
    assert len(result) == 1
    assert result[0].name == "search"


def test_all_tags():
    m = ToolManifest()
    m.add(make_search())  # web, read
    m.add(make_read())    # fs, read
    m.add(make_write())   # fs, write
    tags = m.all_tags()
    assert tags == ["fs", "read", "web", "write"]


# ---------------------------------------------------------------------------
# to_anthropic / to_openai
# ---------------------------------------------------------------------------


def test_to_anthropic_all():
    m = ToolManifest()
    m.add(make_search())
    m.add(make_read())
    result = m.to_anthropic()
    assert len(result) == 2
    names = {d["name"] for d in result}
    assert names == {"search", "read_file"}


def test_to_anthropic_with_tag_filter():
    m = ToolManifest()
    m.add(make_search())   # web, read
    m.add(make_read())     # fs, read
    m.add(make_write())    # fs, write
    result = m.to_anthropic(tags=["web"])
    assert len(result) == 1
    assert result[0]["name"] == "search"


def test_to_openai_all():
    m = ToolManifest()
    m.add(make_search())
    result = m.to_openai()
    assert len(result) == 1
    assert result[0]["type"] == "function"


def test_to_openai_with_tag_filter():
    m = ToolManifest()
    m.add(make_search())   # read
    m.add(make_read())     # read
    m.add(make_write())    # write
    result = m.to_openai(tags=["write"])
    assert len(result) == 1
    assert result[0]["function"]["name"] == "write_file"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_clean():
    m = ToolManifest()
    m.add(make_search())
    m.add(make_read())
    assert m.validate() == []


def test_validate_missing_description():
    m = ToolManifest()
    m.add(ToolDefinition(name="t"))
    warnings = m.validate()
    assert any("description is empty" in w for w in warnings)


def test_validate_non_object_schema():
    m = ToolManifest()
    m.add(ToolDefinition(
        name="t",
        description="desc",
        input_schema={"type": "string"},
    ))
    warnings = m.validate()
    assert any("should be 'object'" in w for w in warnings)


def test_validate_multiple_issues():
    m = ToolManifest()
    m.add(ToolDefinition(
        name="t",
        description="",
        input_schema={"type": "array"},
    ))
    warnings = m.validate()
    assert len(warnings) == 2


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_full():
    m = ToolManifest(name="demo", version="1.0.0", description="test tools")
    m.add(make_search())
    d = m.to_dict()
    assert d["name"] == "demo"
    assert d["tool_count"] == 1
    assert "search" in d["tools"]


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len():
    m = ToolManifest()
    assert len(m) == 0
    m.add(make_search())
    assert len(m) == 1


def test_to_anthropic_empty():
    m = ToolManifest()
    assert m.to_anthropic() == []


def test_to_openai_empty():
    m = ToolManifest()
    assert m.to_openai() == []

"""Tests for the heritage extraction phase (Phase 6)."""

from __future__ import annotations

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from synaptiq.core.ingestion.heritage import process_heritage
from synaptiq.core.ingestion.parser_phase import FileParseData
from synaptiq.core.ingestion.symbol_lookup import build_name_index
from synaptiq.core.parsers.base import ParseResult

_HERITAGE_LABELS = (NodeLabel.CLASS, NodeLabel.INTERFACE, NodeLabel.MODULE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with Class and Interface nodes.

    Layout:
    - Class:src/models.py:Animal
    - Class:src/models.py:Dog
    - Interface:src/types.ts:Serializable
    - Class:src/models.ts:User
    """
    g = KnowledgeGraph()

    # Python class nodes
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.CLASS, "src/models.py", "Animal"),
            label=NodeLabel.CLASS,
            name="Animal",
            file_path="src/models.py",
        )
    )
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.CLASS, "src/models.py", "Dog"),
            label=NodeLabel.CLASS,
            name="Dog",
            file_path="src/models.py",
        )
    )

    # TypeScript interface node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.INTERFACE, "src/types.ts", "Serializable"),
            label=NodeLabel.INTERFACE,
            name="Serializable",
            file_path="src/types.ts",
        )
    )

    # TypeScript class node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.CLASS, "src/models.ts", "User"),
            label=NodeLabel.CLASS,
            name="User",
            file_path="src/models.ts",
        )
    )

    return g


def _make_parse_data(
    file_path: str,
    heritage: list[tuple[str, str, str]],
) -> FileParseData:
    """Create a FileParseData with only heritage tuples populated."""
    return FileParseData(
        file_path=file_path,
        language="python",
        parse_result=ParseResult(heritage=heritage),
    )


# ---------------------------------------------------------------------------
# build_name_index tests (heritage labels)
# ---------------------------------------------------------------------------


class TestBuildSymbolIndex:
    """build_name_index produces a correct mapping from name to node ID."""

    def test_build_symbol_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _HERITAGE_LABELS)

        assert "Animal" in index
        assert "Dog" in index
        assert "Serializable" in index
        assert "User" in index

    def test_index_values_are_node_ids(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _HERITAGE_LABELS)

        for name, node_ids in index.items():
            assert isinstance(node_ids, list)
            for node_id in node_ids:
                node = graph.get_node(node_id)
                assert node is not None
                assert node.name == name

    def test_index_excludes_non_heritage_labels(self, graph: KnowledgeGraph) -> None:
        # Add a function node -- it should NOT appear in the index.
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FUNCTION, "src/models.py", "helper"),
                label=NodeLabel.FUNCTION,
                name="helper",
                file_path="src/models.py",
            )
        )
        index = build_name_index(graph, _HERITAGE_LABELS)
        assert "helper" not in index


# ---------------------------------------------------------------------------
# process_heritage — extends
# ---------------------------------------------------------------------------


class TestProcessHeritageExtends:
    """Dog extends Animal creates an EXTENDS relationship."""

    def test_process_heritage_extends(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Dog", "extends", "Animal")],
            ),
        ]
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 1

        rel = extends_rels[0]
        assert rel.source == generate_id(NodeLabel.CLASS, "src/models.py", "Dog")
        assert rel.target == generate_id(NodeLabel.CLASS, "src/models.py", "Animal")

    def test_extends_relationship_id_format(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Dog", "extends", "Animal")],
            ),
        ]
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        rel = extends_rels[0]
        assert rel.id.startswith("extends:")
        assert "->" in rel.id


# ---------------------------------------------------------------------------
# process_heritage — implements
# ---------------------------------------------------------------------------


class TestProcessHeritageImplements:
    """User implements Serializable creates an IMPLEMENTS relationship."""

    def test_process_heritage_implements(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.ts",
                [("User", "implements", "Serializable")],
            ),
        ]
        process_heritage(parse_data, graph)

        impl_rels = graph.get_relationships_by_type(RelType.IMPLEMENTS)
        assert len(impl_rels) == 1

        rel = impl_rels[0]
        assert rel.source == generate_id(NodeLabel.CLASS, "src/models.ts", "User")
        assert rel.target == generate_id(NodeLabel.INTERFACE, "src/types.ts", "Serializable")

    def test_implements_relationship_type(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.ts",
                [("User", "implements", "Serializable")],
            ),
        ]
        process_heritage(parse_data, graph)

        impl_rels = graph.get_relationships_by_type(RelType.IMPLEMENTS)
        assert impl_rels[0].type == RelType.IMPLEMENTS


# ---------------------------------------------------------------------------
# process_heritage — unresolved parent
# ---------------------------------------------------------------------------


class TestProcessHeritageUnresolvedParent:
    """Heritage referencing an unknown parent is silently skipped."""

    def test_process_heritage_unresolved_parent(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Dog", "extends", "UnknownBase")],
            ),
        ]
        # Should not raise.
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 0

    def test_unresolved_child_also_skipped(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Phantom", "extends", "Animal")],
            ),
        ]
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 0


# ---------------------------------------------------------------------------
# process_heritage — multiple heritage
# ---------------------------------------------------------------------------


class TestProcessHeritageMultiple:
    """A class with one extends and two implements produces 3 relationships."""

    def test_multiple_heritage(self, graph: KnowledgeGraph) -> None:
        # Add a second interface for the test.
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.INTERFACE, "src/types.ts", "Printable"),
                label=NodeLabel.INTERFACE,
                name="Printable",
                file_path="src/types.ts",
            )
        )
        # User extends Animal (cross-file), implements Serializable, implements Printable
        # For cross-file extends to work we need Animal in the graph (it is).
        parse_data = [
            _make_parse_data(
                "src/models.ts",
                [
                    ("User", "extends", "Animal"),
                    ("User", "implements", "Serializable"),
                    ("User", "implements", "Printable"),
                ],
            ),
        ]
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        impl_rels = graph.get_relationships_by_type(RelType.IMPLEMENTS)

        assert len(extends_rels) == 1
        assert len(impl_rels) == 2

        # Total: 3 heritage relationships.
        total = len(extends_rels) + len(impl_rels)
        assert total == 3

    def test_multiple_heritage_sources_are_correct(self, graph: KnowledgeGraph) -> None:
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.INTERFACE, "src/types.ts", "Printable"),
                label=NodeLabel.INTERFACE,
                name="Printable",
                file_path="src/types.ts",
            )
        )
        parse_data = [
            _make_parse_data(
                "src/models.ts",
                [
                    ("User", "extends", "Animal"),
                    ("User", "implements", "Serializable"),
                    ("User", "implements", "Printable"),
                ],
            ),
        ]
        process_heritage(parse_data, graph)

        user_id = generate_id(NodeLabel.CLASS, "src/models.ts", "User")

        all_rels = graph.get_relationships_by_type(
            RelType.EXTENDS
        ) + graph.get_relationships_by_type(RelType.IMPLEMENTS)

        for rel in all_rels:
            assert rel.source == user_id


# ---------------------------------------------------------------------------
# Protocol annotation tests
# ---------------------------------------------------------------------------


class TestProtocolAnnotation:
    """Heritage with unresolvable Protocol parent annotates the child."""

    def test_protocol_parent_annotates_child(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Animal", "extends", "Protocol")],
            ),
        ]
        process_heritage(parse_data, graph)

        animal_id = generate_id(NodeLabel.CLASS, "src/models.py", "Animal")
        animal = graph.get_node(animal_id)
        assert animal is not None
        assert animal.properties.get("is_protocol") is True

    def test_abc_parent_annotates_child(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Animal", "extends", "ABC")],
            ),
        ]
        process_heritage(parse_data, graph)

        animal_id = generate_id(NodeLabel.CLASS, "src/models.py", "Animal")
        animal = graph.get_node(animal_id)
        assert animal is not None
        assert animal.properties.get("is_protocol") is True

    def test_non_protocol_parent_not_annotated(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Dog", "extends", "UnknownBase")],
            ),
        ]
        process_heritage(parse_data, graph)

        dog_id = generate_id(NodeLabel.CLASS, "src/models.py", "Dog")
        dog = graph.get_node(dog_id)
        assert dog is not None
        assert dog.properties.get("is_protocol") is None

    def test_protocol_annotation_does_not_create_edge(self, graph: KnowledgeGraph) -> None:
        """Protocol annotation should NOT create an EXTENDS edge."""
        parse_data = [
            _make_parse_data(
                "src/models.py",
                [("Animal", "extends", "Protocol")],
            ),
        ]
        process_heritage(parse_data, graph)

        extends_rels = graph.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 0


# ---------------------------------------------------------------------------
# process_heritage — Ruby mixins (MIXES_IN)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ruby_graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph with Ruby Class and Module nodes.

    Layout:
    - Module:app/concerns/trackable.rb:Trackable
    - Class:app/models/user.rb:User
    - Class:app/models/base.rb:Base
    - Module:app/models/user.rb:LocalMixin  (same-file as User)
    - Module:app/concerns/other.rb:LocalMixin  (cross-file duplicate name)
    """
    g = KnowledgeGraph()

    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.MODULE, "app/concerns/trackable.rb", "Trackable"),
            label=NodeLabel.MODULE,
            name="Trackable",
            file_path="app/concerns/trackable.rb",
        )
    )
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.CLASS, "app/models/user.rb", "User"),
            label=NodeLabel.CLASS,
            name="User",
            file_path="app/models/user.rb",
        )
    )
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.CLASS, "app/models/base.rb", "Base"),
            label=NodeLabel.CLASS,
            name="Base",
            file_path="app/models/base.rb",
        )
    )
    # Same-name module in two files to exercise same-file preference.
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.MODULE, "app/concerns/other.rb", "LocalMixin"),
            label=NodeLabel.MODULE,
            name="LocalMixin",
            file_path="app/concerns/other.rb",
        )
    )
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.MODULE, "app/models/user.rb", "LocalMixin"),
            label=NodeLabel.MODULE,
            name="LocalMixin",
            file_path="app/models/user.rb",
        )
    )

    return g


def _make_ruby_parse_data(
    file_path: str,
    heritage: list[tuple[str, str, str]],
) -> FileParseData:
    """Create a Ruby FileParseData with only heritage tuples populated."""
    return FileParseData(
        file_path=file_path,
        language="ruby",
        parse_result=ParseResult(heritage=heritage),
    )


class TestProcessHeritageMixin:
    """Ruby include/extend/prepend produce MIXES_IN relationships."""

    def test_include_creates_mixes_in(self, ruby_graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_ruby_parse_data(
                "app/models/user.rb",
                [("User", "mixin", "Trackable")],
            ),
        ]
        process_heritage(parse_data, ruby_graph)

        mixin_rels = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)
        assert len(mixin_rels) == 1

        rel = mixin_rels[0]
        assert rel.type == RelType.MIXES_IN
        assert rel.source == generate_id(NodeLabel.CLASS, "app/models/user.rb", "User")
        assert rel.target == generate_id(NodeLabel.MODULE, "app/concerns/trackable.rb", "Trackable")

    def test_mixin_relationship_id_format(self, ruby_graph: KnowledgeGraph) -> None:
        parse_data = [
            _make_ruby_parse_data(
                "app/models/user.rb",
                [("User", "mixin", "Trackable")],
            ),
        ]
        process_heritage(parse_data, ruby_graph)

        rel = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)[0]
        assert rel.id.startswith("mixin:")
        assert "->" in rel.id

    def test_class_inheritance_still_extends(self, ruby_graph: KnowledgeGraph) -> None:
        """``class User < Base`` must still produce an EXTENDS edge, not MIXES_IN."""
        parse_data = [
            _make_ruby_parse_data(
                "app/models/user.rb",
                [("User", "extends", "Base")],
            ),
        ]
        process_heritage(parse_data, ruby_graph)

        extends_rels = ruby_graph.get_relationships_by_type(RelType.EXTENDS)
        mixin_rels = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)

        assert len(extends_rels) == 1
        assert len(mixin_rels) == 0
        assert extends_rels[0].target == generate_id(NodeLabel.CLASS, "app/models/base.rb", "Base")

    def test_mixin_prefers_same_file_module_target(self, ruby_graph: KnowledgeGraph) -> None:
        """A mixin target defined in the importing file wins over a cross-file dup."""
        parse_data = [
            _make_ruby_parse_data(
                "app/models/user.rb",
                [("User", "mixin", "LocalMixin")],
            ),
        ]
        process_heritage(parse_data, ruby_graph)

        rel = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)[0]
        assert rel.target == generate_id(NodeLabel.MODULE, "app/models/user.rb", "LocalMixin")

    def test_unresolved_external_mixin_skipped(self, ruby_graph: KnowledgeGraph) -> None:
        """An include of an external/unknown module is skipped without error."""
        parse_data = [
            _make_ruby_parse_data(
                "app/models/user.rb",
                [("User", "mixin", "Comparable")],
            ),
        ]
        # Should not raise.
        process_heritage(parse_data, ruby_graph)

        mixin_rels = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)
        assert len(mixin_rels) == 0

    def test_module_can_be_mixin_source(self, ruby_graph: KnowledgeGraph) -> None:
        """A module including another module produces MIXES_IN (module as source)."""
        parse_data = [
            _make_ruby_parse_data(
                "app/concerns/trackable.rb",
                [("Trackable", "mixin", "LocalMixin")],
            ),
        ]
        process_heritage(parse_data, ruby_graph)

        mixin_rels = ruby_graph.get_relationships_by_type(RelType.MIXES_IN)
        assert len(mixin_rels) == 1
        assert mixin_rels[0].source == generate_id(
            NodeLabel.MODULE, "app/concerns/trackable.rb", "Trackable"
        )

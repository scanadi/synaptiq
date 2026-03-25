"""Tests for TypeScript/TSX parser extensions — JSX, re-exports, callbacks."""

from __future__ import annotations

from synaptiq.core.parsers.typescript import TypeScriptParser

# ---------------------------------------------------------------------------
# JSX element references (Fix 1)
# ---------------------------------------------------------------------------


class TestJsxElementReferences:
    """JSX usage of components should emit CallInfo entries."""

    def test_self_closing_jsx_emits_call(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            'const App = () => <MyComponent prop="value" />;',
            "App.tsx",
        )

        call_names = [c.name for c in result.calls]
        assert "MyComponent" in call_names

    def test_opening_jsx_emits_call(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            "const App = () => <MyComponent>child</MyComponent>;",
            "App.tsx",
        )

        call_names = [c.name for c in result.calls]
        assert "MyComponent" in call_names

    def test_lowercase_jsx_tag_skipped(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            "const App = () => <div className='x'>text</div>;",
            "App.tsx",
        )

        call_names = [c.name for c in result.calls]
        assert "div" not in call_names

    def test_member_expression_jsx(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            "const App = () => <Foo.Bar />;",
            "App.tsx",
        )

        matching = [c for c in result.calls if c.name == "Bar"]
        assert len(matching) == 1
        assert matching[0].receiver == "Foo"


# ---------------------------------------------------------------------------
# JSX attribute callbacks (Fix 2)
# ---------------------------------------------------------------------------


class TestJsxAttributeCallbacks:
    """JSX expression references like {handleClick} should emit CallInfo."""

    def test_jsx_expression_identifier(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            "const App = () => <button onClick={handleClick}>Go</button>;",
            "App.tsx",
        )

        call_names = [c.name for c in result.calls]
        assert "handleClick" in call_names

    def test_jsx_expression_member(self) -> None:
        parser = TypeScriptParser(dialect="tsx")
        result = parser.parse(
            "const App = () => <button onClick={handlers.onClick}>Go</button>;",
            "App.tsx",
        )

        matching = [c for c in result.calls if c.name == "onClick"]
        assert len(matching) >= 1
        assert matching[0].receiver == "handlers"


# ---------------------------------------------------------------------------
# Export default identifier (Fix 4)
# ---------------------------------------------------------------------------


class TestExportDefaultIdentifier:
    """export default Foo should add Foo to exports."""

    def test_export_default_identifier(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "function MyComponent() {}\nexport default MyComponent;",
            "component.ts",
        )

        assert "MyComponent" in result.exports

    def test_export_default_function(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "export default function handler() { return 1; }",
            "handler.ts",
        )

        assert "handler" in result.exports


# ---------------------------------------------------------------------------
# Re-exports and barrel files (Fix 6)
# ---------------------------------------------------------------------------


class TestReExports:
    """Re-export statements should emit ImportInfo entries."""

    def test_named_re_export(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "export { Foo, Bar } from './module';",
            "index.ts",
        )

        # Should have exports
        assert "Foo" in result.exports
        assert "Bar" in result.exports

        # Should also have an import for the re-exported module
        re_import = [i for i in result.imports if i.module == "./module"]
        assert len(re_import) == 1
        assert "Foo" in re_import[0].names
        assert "Bar" in re_import[0].names
        assert re_import[0].is_relative is True

    def test_wildcard_re_export(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "export * from './utils';",
            "index.ts",
        )

        re_import = [i for i in result.imports if i.module == "./utils"]
        assert len(re_import) == 1
        assert re_import[0].names == []  # wildcard has no specific names
        assert re_import[0].is_relative is True


# ---------------------------------------------------------------------------
# Object property callbacks (Fix 7)
# ---------------------------------------------------------------------------


class TestObjectPropertyCallbacks:
    """Object properties with identifier values should emit CallInfo."""

    def test_pair_callback(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "const config = { onSuccess: handleSuccess, retries: 3 };",
            "config.ts",
        )

        call_names = [c.name for c in result.calls]
        assert "handleSuccess" in call_names

    def test_shorthand_property(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "const obj = { handleClick, doThing };",
            "handlers.ts",
        )

        call_names = [c.name for c in result.calls]
        assert "handleClick" in call_names
        assert "doThing" in call_names


# ---------------------------------------------------------------------------
# Dynamic import() (Fix 8)
# ---------------------------------------------------------------------------


class TestDynamicImport:
    """Dynamic import() should emit ImportInfo."""

    def test_dynamic_import_emits_import_info(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "const mod = import('./Component');",
            "lazy.ts",
        )

        matching = [i for i in result.imports if i.module == "./Component"]
        assert len(matching) == 1
        assert matching[0].is_relative is True


# ---------------------------------------------------------------------------
# Object literal class_name resolution
# ---------------------------------------------------------------------------


class TestObjectLiteralClassName:
    """Methods in object literals should get the variable name as class_name."""

    def test_object_literal_method_has_class_name(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "const EmailService = {\n  sendOTP() { return true; }\n};",
            "service.ts",
        )

        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "sendOTP"
        assert methods[0].class_name == "EmailService"

    def test_class_method_still_works(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        result = parser.parse(
            "class Pool {\n  acquire() { return null; }\n}",
            "pool.ts",
        )

        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "acquire"
        assert methods[0].class_name == "Pool"

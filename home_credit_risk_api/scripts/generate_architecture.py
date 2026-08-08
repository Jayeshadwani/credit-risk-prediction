from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "app"
OUTPUT_FILE = PROJECT_ROOT / "architecture.mmd"


def get_function_name(
    node: ast.Call,
) -> str | None:
    """
    Extracts the called function or method name from
    a Python AST call node.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id

    if isinstance(node.func, ast.Attribute):
        return node.func.attr

    return None


def collect_defined_functions(
    source_dir: Path,
) -> dict[str, str]:
    """
    Finds functions defined in the project and maps each
    function name to its source module.
    """
    functions: dict[str, str] = {}

    for file_path in source_dir.rglob("*.py"):
        tree = ast.parse(file_path.read_text())

        module = (
            file_path.relative_to(PROJECT_ROOT)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )

        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                functions[node.name] = module

    return functions


def collect_function_calls(
    source_dir: Path,
    known_functions: set[str],
) -> set[tuple[str, str]]:
    """
    Builds caller-to-callee relationships for functions
    defined within the local project.
    """
    relationships: set[tuple[str, str]] = set()

    for file_path in source_dir.rglob("*.py"):
        tree = ast.parse(file_path.read_text())

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            caller = node.name

            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue

                callee = get_function_name(child)

                if (
                    callee
                    and callee in known_functions
                    and callee != caller
                ):
                    relationships.add(
                        (caller, callee)
                    )

    return relationships


def sanitize_mermaid_id(
    value: str,
) -> str:
    """
    Converts Python function names into Mermaid-safe
    node identifiers.
    """
    return "".join(
        character
        if character.isalnum()
        else "_"
        for character in value
    )


def build_mermaid(
    relationships: set[tuple[str, str]],
) -> str:
    """
    Converts discovered function-call relationships into
    a Mermaid flowchart definition.
    """
    lines = [
        "flowchart TD",
    ]

    nodes = {
        function
        for relationship in relationships
        for function in relationship
    }

    for function in sorted(nodes):
        node_id = sanitize_mermaid_id(function)

        lines.append(
            f'    {node_id}["{function}"]'
        )

    lines.append("")

    for caller, callee in sorted(relationships):
        caller_id = sanitize_mermaid_id(caller)
        callee_id = sanitize_mermaid_id(callee)

        lines.append(
            f"    {caller_id} --> {callee_id}"
        )

    return "\n".join(lines)


def main() -> None:
    """
    Scans the application code and generates a Mermaid
    function-call architecture graph.
    """
    functions = collect_defined_functions(
        SOURCE_DIR
    )

    relationships = collect_function_calls(
        SOURCE_DIR,
        known_functions=set(functions),
    )

    mermaid = build_mermaid(
        relationships
    )

    OUTPUT_FILE.write_text(
        mermaid,
        encoding="utf-8",
    )

    print(
        f"Generated {OUTPUT_FILE}"
    )

    print(
        f"Functions discovered: {len(functions)}"
    )

    print(
        f"Connections discovered: {len(relationships)}"
    )


if __name__ == "__main__":
    main()
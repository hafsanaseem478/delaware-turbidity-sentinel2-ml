from pathlib import Path
import ast

ROOT = Path("04_src")

for path in ROOT.rglob("*.py"):
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"SKIPPED {path}: {e}")
        continue

    ranges = []

    nodes = [tree]
    nodes += [
        n for n in ast.walk(tree)
        if isinstance(
            n,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    ]

    for node in nodes:
        body = getattr(node, "body", [])

        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            doc = body[0]
            ranges.append(
                (
                    doc.lineno - 1,
                    doc.end_lineno - 1,
                )
            )

    for start, end in sorted(
        ranges,
        reverse=True,
    ):
        del lines[start:end + 1]

    text = "".join(lines)

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    path.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Cleaned: {path}")
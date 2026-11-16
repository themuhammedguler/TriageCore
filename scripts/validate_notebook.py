import ast
import json

nb = json.load(open("notebooks/TriageCore_QLoRA_Colab.ipynb", encoding="utf-8"))
flags = getattr(ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0)

all_ok = True
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        raw = "".join(cell["source"])
        clean_lines = [line for line in raw.splitlines() if not line.strip().startswith(("!", "%"))]
        code = "\n".join(clean_lines)
        try:
            compile(code, f"Cell_{i}", "exec", flags=flags)
            print(f"Cell {i} ({cell['cell_type']}): COMPILES OK (lines: {len(clean_lines)})")
        except SyntaxError as e:
            print(f"Cell {i} SYNTAX ERROR: {e}")
            all_ok = False

if all_ok:
    print("\n[+] TUM KOD HUCRELERI SOZDIZIMI (SYNTAX) BAKIMINDAN KUSURSUZ!")

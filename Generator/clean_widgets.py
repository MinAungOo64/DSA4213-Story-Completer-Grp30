import nbformat

INPUT = "Generator Training.ipynb"
OUTPUT = "Generator Training Clean.ipynb"

nb = nbformat.read(INPUT, as_version=4)

# Remove top-level widgets metadata
nb.metadata.pop("widgets", None)

for cell in nb.cells:
    # Remove cell-level widget metadata
    cell.metadata.pop("widgets", None)

    # Remove widget outputs (but keep other outputs)
    if "outputs" in cell:
        new_outputs = []
        for out in cell.outputs:
            data = out.get("data", {})
            if "application/vnd.jupyter.widget-view+json" in data:
                # Skip this output (widget view)
                continue
            new_outputs.append(out)
        cell.outputs = new_outputs

nbformat.write(nb, OUTPUT)
print(f"Cleaned notebook saved as: {OUTPUT}")


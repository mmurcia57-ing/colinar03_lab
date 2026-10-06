"""Exporta el notebook ejecutado a PDF (celdas + salidas) vía LaTeX.

Uso: python scripts/export_pdf.py
Requiere xelatex. Ejecuta antes el notebook (ver README).
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

from nbconvert import LatexExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "laboratorio_eliminacion_anomalias_ejecutado.ipynb"
OUT = ROOT / "deliverables" / "laboratorio_eliminacion_anomalias.pdf"

# pandoc >= 3 emite \setcounter{none} en listas; además se reducen márgenes y título.
PREAMBLE_FIX = r"\newcounter{none}" "\n" r"\setcounter{secnumdepth}{0}" "\n" r"\geometry{margin=1.6cm}" "\n"


def main() -> None:
    body, resources = LatexExporter().from_filename(str(NOTEBOOK))
    body = body.replace(r"\begin{document}", PREAMBLE_FIX + r"\begin{document}", 1)
    body = body.replace(r"\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}", "")
    body = body.replace(r"\maketitle", "")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "notebook.tex").write_text(body, encoding="utf-8")
        for name, data in resources.get("outputs", {}).items():
            (tmp / name).write_bytes(data)
        for _ in range(2):
            subprocess.run(["xelatex", "-interaction=nonstopmode", "notebook.tex"], cwd=tmp, check=False,
                           stdout=subprocess.DEVNULL)
        pdf = tmp / "notebook.pdf"
        if not pdf.exists():
            raise SystemExit("xelatex no generó el PDF; revisa notebook.log")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, OUT)
    print(f"PDF escrito en {OUT}")


if __name__ == "__main__":
    main()

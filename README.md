# Eliminación de anomalías de la imagen

Laboratorio de Percepción Computacional para detectar y restaurar ruido impulsivo en imágenes.

El repositorio contiene una implementación reproducible y explicada de un detector contextual propio. La solución se prueba sobre dos imágenes distintas contaminadas con la misma anomalía y genera un notebook ejecutado junto con un PDF de su ejecución.

## Contenido

```text
src/anomaly_removal.py      algoritmo propio (detección contextual + restauración)
src/metrics.py              métricas de evaluación (precisión, recall, F1, PSNR, SSIM)
src/visualization.py        figuras y tablas del notebook
scripts/generate_data.py    prepara 3 fotos reales (skimage.data) y les añade 10 % de sal y pimienta
scripts/export_pdf.py       exporta el notebook ejecutado a PDF (requiere xelatex)
notebooks/                  notebook fuente y notebook ejecutado
deliverables/               PDF con toda la ejecución del notebook (6 páginas)
data/                       originales, contaminadas, máscaras reales y salidas
tests/                      pruebas automáticas del algoritmo
```

## Ejecución

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py          # opcional: los datos ya están en data/
jupyter nbconvert --to notebook --execute notebooks/laboratorio_eliminacion_anomalias.ipynb \
  --output laboratorio_eliminacion_anomalias_ejecutado.ipynb
python scripts/export_pdf.py             # genera deliverables/laboratorio_eliminacion_anomalias.pdf
pytest                                   # pruebas
```

También se puede abrir el notebook en Jupyter o VS Code (kernel `.venv`) y ejecutar todas las celdas.

## Entrega

Todos los integrantes entregan el notebook (`notebooks/laboratorio_eliminacion_anomalias.ipynb`), los ficheros de `src/`, `scripts/` y `data/`, y el PDF de `deliverables/`.

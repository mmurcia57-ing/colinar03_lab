# Eliminación de anomalías de la imagen

Laboratorio de Percepción Computacional para detectar y restaurar ruido impulsivo en imágenes.

El repositorio contiene una implementación reproducible y explicada de un detector contextual propio. La solución se prueba sobre dos imágenes distintas contaminadas con la misma anomalía y genera un notebook ejecutado junto con un PDF de su ejecución.

## Estado

El repositorio de GitHub de partida estaba vacío. El diseño y la implementación se desarrollan aquí de forma incremental, con commits que separan requisitos, diseño, algoritmo, validación y entrega.

## Estructura prevista

```text
docs/
  design/solution-design.md
  requirements.md
src/
  anomaly_removal.py
  metrics.py
  visualization.py
data/
  original/
  corrupted/
  output/
notebooks/
  laboratorio_eliminacion_anomalias.ipynb
tests/
  test_algorithm.py
requirements.txt
```

## Ejecución

El notebook es la entrada principal. Las funciones reutilizables viven en `src/` para que el experimento sea auditable y no esconda la operación principal en celdas.


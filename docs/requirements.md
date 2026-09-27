# Requisitos del laboratorio

Fuente: guía UNIR `colinar03_lab.docx`, adjunta a la conversación de trabajo.

## Requisitos funcionales

1. Identificar una anomalía o artefacto en imágenes y aplicar operaciones de procesado para eliminarlo.
2. Entregar un notebook Python con estas secciones: integrantes, descripción del problema, al menos dos imágenes con la anomalía, solución propuesta y ejecución paso a paso comentada.
3. Aplicar exactamente el mismo algoritmo a al menos dos imágenes distintas con la misma anomalía.
4. Mostrar los resultados de los pasos principales y adjuntar un PDF que contenga toda la ejecución del notebook.

## Restricciones académicas

1. La solución no puede ser ad hoc: debe extrapolarse a otras imágenes con la misma anomalía.
2. La solución no puede limitarse a repetir una solución básica conocida o una llamada a una librería.
3. Se pueden usar librerías de apoyo, pero la operación principal debe estar implementada por el grupo.
4. No se permite copiar código de Internet. Las ideas externas reutilizadas deben referenciarse.
5. La memoria tiene un límite máximo de seis páginas.

## Mapping requisito a evidencia

| Requisito o criterio | Artefacto | Evidencia verificable |
|---|---|---|
| Describir el problema | Notebook, sección 1 | Tipo de anomalía, hipótesis y motivación |
| Dos imágenes con la misma anomalía | `data/original/`, `data/corrupted/`, notebook | Dos pares con parámetros de corrupción comunes |
| Algoritmo no ad hoc | `src/anomaly_removal.py`, notebook | Mismos parámetros y misma función para ambos pares |
| Operación principal propia | `src/anomaly_removal.py` | Recorrido de vecindad, detector y restaurador implementados sin `medianBlur`/equivalente |
| Ejecución paso a paso | Notebook ejecutado | Celdas de carga, detección, mapa, restauración y resultados |
| Validación de eliminación | `src/metrics.py`, notebook | PSNR, SSIM y métricas de detección usando la imagen limpia como referencia |
| Código claro | `src/`, tests | Funciones documentadas, nombres explícitos y prueba automatizada |
| PDF completo y memoria <= 6 páginas | `deliverables/` | Exportación del notebook y comprobación de páginas |
| No plagio | `docs/design/solution-design.md`, notebook | Bibliografía e identificación de decisiones propias |


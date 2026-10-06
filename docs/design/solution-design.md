# Diseño de solución

## 1. Problema elegido

Se estudiará ruido impulsivo tipo sal y pimienta: algunos píxeles toman valores extremos, normalmente 0 o 255, mientras el resto de la imagen conserva su estructura. Se usará la misma familia de corrupción en dos imágenes de contenido diferente.

La elección es adecuada porque la anomalía es contextual: un píxel extremo no siempre es anómalo en una región de alto contraste, por lo que el detector debe contrastar el píxel con su vecindad y proteger bordes.

## 2. Alternativas evaluadas

| Alternativa | Ventaja | Riesgo frente a la rúbrica | Decisión |
|---|---|---|---|
| Umbral global de intensidad | Muy simple y fácil de explicar | Confunde objetos claros/oscuros con ruido; es ad hoc para una imagen | Descartada |
| Filtro de mediana de librería | Bueno para impulsos y disponible | Repite una solución conocida y no demuestra operación propia | Solo baseline de referencia, no solución |
| Mediana implementada a mano en toda la imagen | Operación propia | Sigue siendo una réplica básica del filtro estándar; puede borrar bordes | No prioritaria |
| Detector local por desviación respecto a la vecindad + reemplazo fijo | Más contextual | Un umbral fijo puede fallar entre imágenes y en bordes | Parcial, como baseline |
| Detector adaptativo por mediana, dispersión robusta y contraste local + restauración ponderada | Explicable, generalizable y propia; detecta solo outliers compatibles con impulso y restaura con vecinos no anómalos | Requiere justificar parámetros y tratar bordes | Elegida |
| Modelos aprendidos o autoencoders | Potencialmente potentes | Necesitan datos, entrenamiento y complican la trazabilidad; no son necesarios para el objetivo | Descartados |

## 3. Algoritmo propuesto

Por canal, con parámetros fijos para todas las imágenes (`src/anomaly_removal.py`):

1. **Candidatos:** píxeles con valor entre 0 y 10 o entre 245 y 255.
2. **Decisión contextual** para cada candidato:
   - ventana adaptativa de 3×3 a 9×9 hasta reunir al menos 4 vecinos *fiables* (no saturados);
   - **región saturada real** si tiene al menos 3 vecinos con su misma saturación en 3×3 y al menos 8 en 5×5 (la esquina de un objeto cumple ambas condiciones): se conserva;
   - **impulso** si su valor queda fuera de [mín − 20, máx + 20] de los vecinos fiables; si no, se conserva.
3. **Restauración** solo de los impulsos: media de los vecinos no marcados, con pesos espaciales gaussianos multiplicados por pesos de rango respecto a la mediana local.

Una primera versión usaba un umbral proporcional a la MAD (identificador de Hampel). En fotografías reales con textura, la MAD inflaba el umbral y se escapaba más del 30 % de los impulsos (recall 0.65 en *astronaut*). Calcular el contexto solo con vecinos fiables, junto con la envolvente local, resolvió el problema (recall 0.89–0.98 y PSNR por encima de la mediana de librería en todos los casos probados).

La operación principal propia es el recorrido de ventanas, la decisión contextual y la restauración ponderada. No se usan `cv2.medianBlur`, `scipy.ndimage.median_filter` ni ninguna función equivalente como solución.

## 4. Generalización y validación

El algoritmo recibirá solamente la imagen, tamaño de ventana y parámetros definidos antes de ver los resultados de cada imagen. Se aplica sin cambios a tres fotografías reales (dos en color y una en gris) y a densidades de ruido del 5 % al 30 %. Como las imágenes limpias se conservan antes de inyectar la anomalía, se podrá medir:

- detección: precision, recall y F1 sobre la máscara de píxeles alterados;
- restauración: PSNR, SSIM y error absoluto medio frente a la imagen limpia;
- efecto visual: imagen corrupta, máscara detectada, imagen restaurada y diferencias.

Se comparará contra dos baselines de interpretación: no restaurar y una mediana de referencia de librería, dejando claro que el baseline no es la contribución principal.

## 5. Referencias y autoría

La especificación y los criterios proceden de `colinar03_lab.docx`. Las nociones de ruido impulsivo, mediana y MAD se usarán como fundamentos de procesamiento robusto; la combinación concreta del detector contextual, la protección por contraste y la restauración ponderada se implementará de forma propia en este repositorio. La bibliografía del notebook identificará las fuentes teóricas consultadas, sin copiar código externo.


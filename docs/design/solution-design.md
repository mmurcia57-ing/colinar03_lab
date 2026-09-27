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

Para cada píxel y canal:

1. Extraer una ventana cuadrada impar alrededor del píxel.
2. Excluir el centro y calcular mediana local y desviación absoluta mediana (MAD). La MAD aporta una escala robusta que no queda dominada por unos pocos impulsos.
3. Marcar el centro como candidato si su diferencia frente a la mediana supera `max(k_mad * 1.4826 * MAD, tau_min)` y además es compatible con una saturación impulsiva o con un residuo extremo respecto a la vecindad.
4. Calcular un contraste local con gradientes simples. En un borde fuerte, exigir una evidencia más estricta para no convertir el borde en anomalía.
5. Restaurar solo los candidatos: usar los vecinos no marcados, ponderados por similitud a la mediana local y distancia espacial. Si no hay suficientes vecinos válidos, usar la mediana local.
6. Repetir una segunda pasada opcional con el mapa de la primera pasada fijado, para tratar grupos pequeños de impulsos sin propagar valores corruptos.

La operación principal propia es el recorrido de ventanas, la estadística robusta, el criterio contextual y la restauración ponderada. No se usará `cv2.medianBlur`, `scipy.ndimage.median_filter` ni una función equivalente como solución.

## 4. Generalización y validación

El algoritmo recibirá solamente la imagen, tamaño de ventana y parámetros definidos antes de ver los resultados de cada imagen. Se aplicará sin cambios a dos imágenes diferentes. Como las imágenes limpias se conservan antes de inyectar la anomalía, se podrá medir:

- detección: precision, recall y F1 sobre la máscara de píxeles alterados;
- restauración: PSNR, SSIM y error absoluto medio frente a la imagen limpia;
- efecto visual: imagen corrupta, máscara detectada, imagen restaurada y diferencias.

Se comparará contra dos baselines de interpretación: no restaurar y una mediana de referencia de librería, dejando claro que el baseline no es la contribución principal.

## 5. Referencias y autoría

La especificación y los criterios proceden de `colinar03_lab.docx`. Las nociones de ruido impulsivo, mediana y MAD se usarán como fundamentos de procesamiento robusto; la combinación concreta del detector contextual, la protección por contraste y la restauración ponderada se implementará de forma propia en este repositorio. La bibliografía del notebook identificará las fuentes teóricas consultadas, sin copiar código externo.


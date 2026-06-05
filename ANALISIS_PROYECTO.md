# Análisis técnico del proyecto — ProyectoJic2026

---

## 1. Scripts de dataset1 — estado y comentarios

### `medir_brightness_dataset.py`
**Estado:** correcto.
Mide el brillo promedio de píxeles (canal gris, 0–255) de cada imagen y lo exporta a CSV.
El promedio de intensidad gris es la métrica estándar de brillo perceptual para este tipo de análisis.

### `generar_niveles_iluminacion.py`
**Estado:** correcto.
Función `ajustar_brillo(img, brillo_objetivo)`:
- Escala linealmente todos los píxeles por `factor = objetivo / actual`.
- Hace `clip(0, 255)` para evitar overflow.
- Retorna uint8.
Este método de escalado es determinista y reproducible, lo que es una ventaja para el paper.

### `comparar_ear_iluminacion.py`
**Estado:** correcto.
Implementa la fórmula EAR de Soukupova & Čech (2016):

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Los índices de landmarks de MediaPipe son correctos para ojos izquierdo y derecho.
El EAR promedio bilateral `(ear_izq + ear_der) / 2` es la práctica estándar.

---

## 2. Métodos de mejora de iluminación — estado

### CLAHE (`CLAHE/metodo_CLAHE.py`)
**Estado:** correcto.

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_mejorado = clahe.apply(l)
```

Aplicar CLAHE al canal L en espacio LAB es la implementación estándar de la literatura.
Parámetros: `clipLimit=4.0`, `tileGridSize=(8,8)`.

### GAMMA (`GAMMA/metodo_gamma.py`)
**Estado:** correcto.
Usa gamma adaptativo lineal: mapea brillo_medio/128 → gamma entre 0.4 y 1.0.
Aplica transformación por LUT (eficiente, ~30+ fps).

### RETINEX (`RETINEX/metodo_RETINEX.py`)
**Estado:** correcto.
Single-Scale Retinex (SSR): estima iluminación con blur gaussiano grande (sigma=15),
divide para obtener reflectancia, suaviza con filtro bilateral, normaliza y aplica GAIN=1.2.

### LiteIE (`LITEIE/metodo_LITEIE.py`)
**Estado:** correcto.
Red neuronal ligera (LiteIE). Requiere checkpoint `snapshots/epoth_199.pth`.
Resize a múltiplo de 16 antes de inferencia. Soporta GPU si disponible.

### `metricas_ofc/iluminacion.py`
**Estado:** correcto.
Evalúa iluminación con 3 métricas combinadas:
- Brillo promedio del canal gris
- Desviación estándar (contraste)
- Porcentaje de píxeles oscuros (< intensidad 80)
Clasifica en CRITICA o BIEN según umbrales configurables.

---

## 3. Por qué CLAHE es el peor método — no es un bug

### Los datos (promedio de 673 frames evaluados)

| Método   | illum_org | illum_mej | darkpct_mej | level_mej | PSNR   |
|----------|-----------|-----------|-------------|-----------|--------|
| CLAHE    | ~12       | ~28       | ~99 %       | CRITICA   | ~24 dB |
| GAMMA    | ~12       | ~55       | ~87 %       | CRITICA   | ~15 dB |
| RETINEX  | ~12       | ~80       | ~50 %       | BIEN      | ~10 dB |
| LiteIE   | ~12       | ~95       | ~35 %       | BIEN      | ~9 dB  |

### La razón técnica
CLAHE es **redistribución de histograma local** (contraste), no una transformación de brillo.
Cuando el brillo medio es 3–15/255 con 100% de píxeles oscuros, no hay contraste que redistribuir.
GAMMA, RETINEX y LiteIE aplican transformaciones que elevan el valor absoluto de los píxeles.

### Por qué los papers dicen que CLAHE es mejor
Los benchmarks favorables a CLAHE usan baja luz **moderada** (brillo ~60–100/255),
no oscuridad extrema como el vídeo de este proyecto (~3–15/255).
En condiciones moderadas, CLAHE preserva mejor el detalle sin artefactos.
En oscuridad extrema, falla por diseño.

---

## 4. Problema crítico con las métricas PSNR/SSIM

### Lo que mide tu código actualmente
```
PSNR = psnr(frame_original_oscuro, frame_mejorado)
```

### El problema
- **PSNR alto en CLAHE (~24 dB)** = imagen mejorada muy similar al original oscuro = CLAHE NO cambió nada
- **PSNR bajo en RETINEX/LiteIE (~9 dB)** = imagen muy diferente al original = SÍ mejoró el brillo

**La interpretación convencional está invertida en tu setup.**
En la literatura, PSNR se mide contra una imagen de referencia bien iluminada (ground truth).
En tu caso no hay ground truth, por lo que PSNR mide distorsión respecto al original oscuro.

### Qué debes aclarar en tu investigación
Especificar explícitamente que PSNR/SSIM se calcula como `psnr(original, mejorada)`,
lo que cuantifica cuánto cambia la imagen (no qué tan buena es la calidad visual).
PSNR alto → poca modificación. PSNR bajo → mucha modificación.

---

## 5. Bug real encontrado — GAMMA EAR inválido

### Síntoma
En `GAMMA/resultados_gamma.csv`, la columna `ear_mej` es **idéntica** a `ear_org`
en todos los 673 frames. Esto es estadísticamente imposible si ambos se calculan
sobre imágenes diferentes.

### Causa raíz
En `probar_gamma.py` (líneas 183-246), MediaPipe se usa en modo tracking
(`static_image_mode=False`). Se hacen dos llamadas consecutivas al mismo objeto `face_mesh`:

```python
# 1ª llamada: frame original
fl, bbox = obtener_landmarks(frame_original, face_mesh)

# 2ª llamada: frame mejorado (en el MISMO objeto face_mesh en tracking)
fl_mej, _ = obtener_landmarks(frame_mejorado, face_mesh)
```

En modo tracking, la segunda llamada devuelve los landmarks cacheados del frame anterior
si no detecta un cambio suficiente. Como la mejora solo se aplica al ROI facial (el resto
del frame sigue oscuro), MediaPipe no detecta cambio y retorna los mismos landmarks.
Resultado: `ear_mej == ear_org` exacto.

### Estado del fix — APLICADO
Se añadió `face_mesh_eval` con `static_image_mode=True` en `probar_gamma.py`.
La línea `fl_mej, _ = obtener_landmarks(frame_mejorado, face_mesh)` ahora usa
`face_mesh_eval` en su lugar. Ambos objetos se cierran al final del script.

**Nota de rendimiento:** `static_image_mode=True` no usa tracking entre frames,
lo que puede bajar ligeramente los FPS al procesar el frame mejorado.
Es el mismo tradeoff que tiene RETINEX de facto (su mejora es tan intensa
que MediaPipe re-detecta de cero en cada frame igualmente).

**El CSV `resultados_gamma.csv` es inválido hasta que se ejecute `probar_gamma.py` de nuevo.**

---

## 6. Inconsistencia metodológica entre scripts evaluadores

| Script           | Aplica mejora a     | ear_mej distinto de ear_org |
|-----------------|--------------------|-----------------------------|
| probar_CLAHE.py  | Solo ROI (cara)     | Sí                          |
| probar_gamma.py  | Solo ROI (cara)     | No (bug corregido)          |
| probar_retinex   | Frame completo      | Sí                          |
| probar_liteie    | Frame completo      | Sí                          |

CLAHE y GAMMA aplican la mejora solo a la cara; RETINEX y LiteIE a todo el frame.
Esto hace que MediaPipe vea condiciones distintas entre métodos.
Para una comparación justa hay que estandarizar: todos al frame completo O todos al ROI.

---

## 7. Qué regenerar después del fix de GAMMA

| Archivo                          | Estado          | Acción necesaria                        |
|----------------------------------|-----------------|------------------------------------------|
| `GAMMA/resultados_gamma.csv`     | **Inválido**    | Regenerar ejecutando `probar_gamma.py`   |
| `CLAHE/resultados_clahe.csv`     | Válido          | Ninguna                                  |
| `RETINEX/resultados_retinex.csv` | Válido          | Ninguna                                  |
| `LITEIE/resultados_liteie.csv`   | Válido          | Ninguna                                  |

---

## 8. Referencias bibliográficas para el paper

### Obligatorias (usadas directamente en el código)

- **EAR (Eye Aspect Ratio):**
  Soukupova, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*.
  21st Computer Vision Winter Workshop (CVWW 2016).

- **MediaPipe Face Mesh:**
  Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*.
  arXiv:1906.08172.

- **CLAHE:**
  Pizer, S.M. et al. (1987). *Adaptive histogram equalization and its variations*.
  Computer Vision, Graphics, and Image Processing, 39(3), 355–368.
  (o la versión aplicada de Zuiderveld, K., 1994, en *Graphics Gems IV*)

- **Retinex:**
  Land, E.H. & McCann, J.J. (1971). *Lightness and Retinex Theory*.
  Journal of the Optical Society of America, 61(1), 1–11.

- **OpenCV:**
  Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.

### Recomendadas para contexto

- Sobre detección de somnolencia con EAR:
  Dwivedi, K. et al. (2014). *Drowsy driver detection using representation learning*. IEEE.

- Sobre efectos de iluminación en detección facial:
  Buscar en Google Scholar: "face detection low illumination conditions deep learning" (2020–2024).

---

## 9. Sugerencias para el proyecto

1. **Agregar script de visualización**: gráfico `EAR vs nivel de brillo` y
   `tasa de detección vs nivel de brillo` — es el resultado más impactante para el paper.

2. **Estandarizar scope de mejora**: decidir entre ROI o frame completo para todos los métodos.

3. **Agregar umbral EAR en análisis**: la literatura usa EAR < 0.2 como criterio de ojo cerrado.
   Analizar a qué nivel de brillo ese umbral empieza a producir falsos negativos.

4. **Registrar brillo real en comparacion_EAR_iluminacion.csv**:
   calcular `np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))` además del nivel objetivo,
   para correlacionar brillo medido (no deseado) con el EAR y la tasa de detección.

5. **Renombrar columna en comparar_ear_iluminacion.py**:
   la columna `brightness` del CSV guarda el nivel objetivo (5, 10, 25...) no el brillo real.
   Considera renombrarla `nivel_brillo` para evitar confusión al leer el CSV.

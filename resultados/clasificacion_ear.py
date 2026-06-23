"""
clasificacion_ear.py

Evalúa qué tan bien cada método de mejora de iluminación preserva la
clasificación cerrado/abierto del ojo usando el EAR con umbral fijo.

Lógica:
  - Clase real   : columna `estado_ojo`  (cerrado / abierto)
  - Clase predicha: umbral aplicado a `ear_mejorada`
        < UMBRAL  → cerrado
        >= UMBRAL → abierto
        no detectado → no_det

Resultados por fila:
  TP  cerrado real + cerrado predicho  (somnolencia detectada correctamente)
  TN  abierto real  + abierto predicho  (alerta detectada correctamente)
  FP  abierto real  + cerrado predicho  (falsa alarma)
  FN  cerrado real  + abierto predicho  (somnolencia NO detectada — el peor caso)
  no_det  no se detectó cara en la imagen mejorada

Accuracy = (TP + TN) / (TP + TN + FP + FN)   (excluye no_det del denominador)
"""

import csv
import os

# ── configuración ──────────────────────────────────────────────────────────
UMBRAL   = 0.155          # < umbral → cerrado,  >= umbral → abierto
METODOS  = ['CLAHE', 'GAMMA', 'RETINEX', 'LiteIE']
DIR_IN   = os.path.dirname(os.path.abspath(__file__))
CSV_OUT  = os.path.join(DIR_IN, 'clasificacion_ear_resultados.csv')


# ── helpers ────────────────────────────────────────────────────────────────
def _parse(val):
    try:
        v = float(val)
        return None if v != v else v      # NaN → None
    except (ValueError, TypeError):
        return None


def _clase(ear):
    if ear is None:
        return 'no_det'
    return 'cerrado' if ear < UMBRAL else 'abierto'


def _resultado(real, pred):
    if pred == 'no_det':
        return 'no_det'
    tabla = {
        ('cerrado', 'cerrado'): 'TP',
        ('abierto',  'abierto'):  'TN',
        ('abierto',  'cerrado'): 'FP',
        ('cerrado', 'abierto'):  'FN',
    }
    return tabla[(real, pred)]


# ── procesar todos los métodos ─────────────────────────────────────────────
filas_out = []

for metodo in METODOS:
    ruta = os.path.join(DIR_IN, f'resultados_{metodo}.csv')
    with open(ruta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ear_ref = _parse(r['ear_ref'])
            ear_mej = _parse(r['ear_mejorada'])

            clase_real  = r['estado_ojo']            # etiqueta del dataset
            clase_ref   = _clase(ear_ref)            # umbral sobre referencia
            clase_pred  = _clase(ear_mej)            # umbral sobre mejorada
            resultado   = _resultado(clase_real, clase_pred)

            filas_out.append({
                'metodo':         metodo,
                'imagen_id':      r['imagen_id'],
                'nivel_brillo':   r['nivel_brillo'],
                'estado_ojo':     clase_real,
                'ear_ref':        round(ear_ref, 6) if ear_ref is not None else '',
                'clase_ref':      clase_ref,
                'ear_mejorada':   round(ear_mej, 6) if ear_mej is not None else '',
                'clase_predicha': clase_pred,
                'resultado':      resultado,
            })


# ── guardar CSV ────────────────────────────────────────────────────────────
CAMPOS = ['metodo', 'imagen_id', 'nivel_brillo', 'estado_ojo',
          'ear_ref', 'clase_ref', 'ear_mejorada', 'clase_predicha', 'resultado']

with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=CAMPOS)
    w.writeheader()
    w.writerows(filas_out)

print(f'CSV guardado: {CSV_OUT}')
print(f'Total filas : {len(filas_out)}\n')


# ── resumen por método y nivel ─────────────────────────────────────────────
print(f'Umbral EAR = {UMBRAL}  (< umbral = cerrado,  >= umbral = abierto)\n')
print(f"{'Metodo':<10} {'Nivel':<6} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'no_det':>7} {'Acc_det':>9} {'Acc_total':>10}")
print('-' * 75)

filas_resumen = []

for metodo in METODOS:
    niveles = sorted(set(r['nivel_brillo'] for r in filas_out if r['metodo'] == metodo))
    for nivel in niveles:
        sub = [r for r in filas_out if r['metodo'] == metodo and r['nivel_brillo'] == nivel]
        tp = sum(1 for r in sub if r['resultado'] == 'TP')
        tn = sum(1 for r in sub if r['resultado'] == 'TN')
        fp = sum(1 for r in sub if r['resultado'] == 'FP')
        fn = sum(1 for r in sub if r['resultado'] == 'FN')
        nd = sum(1 for r in sub if r['resultado'] == 'no_det')

        denom_det   = tp + tn + fp + fn
        denom_total = denom_det + nd

        acc_det   = (tp + tn) / denom_det   * 100 if denom_det   > 0 else 0
        acc_total = (tp + tn) / denom_total * 100 if denom_total > 0 else 0

        print(f"{metodo:<10} {nivel:<6} {tp:>5} {tn:>5} {fp:>5} {fn:>5} {nd:>7} {acc_det:>8.1f}% {acc_total:>9.1f}%")

        filas_resumen.append({
            'metodo':      metodo,
            'nivel_brillo': nivel,
            'umbral_ear':  UMBRAL,
            'TP':          tp,
            'TN':          tn,
            'FP':          fp,
            'FN':          fn,
            'no_det':      nd,
            'acc_det_pct':   round(acc_det, 2),
            'acc_total_pct': round(acc_total, 2),
        })
    print()

# ── guardar resumen como CSV ───────────────────────────────────────────────
CSV_RESUMEN = os.path.join(DIR_IN, 'clasificacion_ear_resumen.csv')
CAMPOS_RES = ['metodo', 'nivel_brillo', 'umbral_ear',
              'TP', 'TN', 'FP', 'FN', 'no_det',
              'acc_det_pct', 'acc_total_pct']

with open(CSV_RESUMEN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=CAMPOS_RES)
    w.writeheader()
    w.writerows(filas_resumen)

print(f'Resumen guardado: {CSV_RESUMEN}')

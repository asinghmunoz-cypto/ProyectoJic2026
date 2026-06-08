import csv

def avg(rows, col):
    vals = []
    for r in rows:
        try:
            v = float(r[col])
            if v == v:
                vals.append(v)
        except:
            pass
    return sum(vals)/len(vals) if vals else None

def stats(file):
    rows = []
    with open(file, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    niveles = sorted(set(r['nivel_brillo'] for r in rows))
    print(f"\n=== {file} ===")
    print(f"{'nivel':<6} {'brillo_mej':>11} {'dark%':>7} {'psnr_ref':>9} {'ssim_ref':>9} {'ear_err_mej%':>13} {'det_mej%':>9} {'fps':>8}")

    for nv in niveles:
        sub = [r for r in rows if r['nivel_brillo'] == nv]
        det = sum(1 for r in sub if r['deteccion_mejorada'] == '1') / len(sub) * 100
        bm = avg(sub, 'brillo_mejorada')
        dp = avg(sub, 'darkpct_mejorada')
        pr = avg(sub, 'psnr_vs_ref')
        sr = avg(sub, 'ssim_vs_ref')
        ee = avg(sub, 'ear_error_mej_pct')
        fp = avg(sub, 'fps')
        print(f"{nv:<6} {bm:>11.2f} {dp:>7.2f} {pr:>9.3f} {sr:>9.4f} {ee or 0:>13.2f} {det:>9.1f}% {fp or 0:>8.1f}")

for m in ['resultados_CLAHE.csv', 'resultados_GAMMA.csv', 'resultados_RETINEX.csv', 'resultados_LiteIE.csv']:
    stats(m)

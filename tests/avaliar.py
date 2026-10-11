"""Compara a leitura com o gabarito das fotos em amostras/.

Uso (dentro de backend/): python tests/avaliar.py
Para incluir uma foto nova, adicione-a em amostras/ e a resposta correta em tests/gabarito.json
(equipamento, v = valores na ordem do modelo, f = flags por índice da linha, h = campos do cabeçalho).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ocr import Reader  # noqa: E402

gt = json.loads((ROOT / "tests" / "gabarito.json").read_text())
reader = Reader()
tv = tf = th = nv = nf = nh = wrong_hi = 0
for name, g in gt.items():
    t = time.time()
    raw = (ROOT.parent / "amostras" / name).read_bytes()
    out = reader.read(raw, g["equipamento"])
    auto = reader.read(raw)["equipamento"]
    if auto != g["equipamento"]:
        print(f"  deteccao automatica errou: {auto}")
    dt = time.time() - t
    errs = []
    for i, x in enumerate(out["resultados"]):
        ok = x["valor"] == g["v"][i]
        tv += ok
        nv += 1
        if not ok:
            errs.append(f"{x['parametro']}: {x['valor']} != {g['v'][i]} ({x['confianca']})")
            wrong_hi += x["confianca"] == "alta"
        expected = g["f"].get(str(i), "")
        tf += x["flag"] == expected
        nf += 1
        if x["flag"] != expected:
            errs.append(f"flag {x['parametro']}: {x['flag'] or '-'} != {expected or '-'}")
    hdr = {h["campo"]: h["valor"] for h in out["cabecalho"]}
    for k, v in g["h"].items():
        th += hdr.get(k) == v
        nh += 1
        if hdr.get(k) != v:
            errs.append(f"cabecalho {k}: {hdr.get(k)!r} != {v}")
    print(f"{name} ({dt:.2f}s): {'; '.join(errs) or 'OK'}")
print(f"\nvalores {tv}/{nv} | flags {tf}/{nf} | cabecalho {th}/{nh} | erros com confianca alta: {wrong_hi}")

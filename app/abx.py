"""Leitura da tela de resultados do ABX Micros ES 60.

Layout: três painéis (WBC, RBC, PLT). Cada linha tem rótulo, unidade, valor numa caixa branca
e, quando fora da referência, a letra L (valor em azul) ou H (valor em vermelho).

  1. OCR na imagem em tons de cinza (texto escuro sobre fundo claro já lê bem).
  2. Rótulos lidos servem de âncora para a posição das linhas de cada grupo.
  3. Valor = número puro à direita do rótulo na mesma linha; a célula é relida ampliada e as duas
     leituras precisam concordar para confiança alta.
  4. Flag pela cor do texto do valor; a letra H/L lida pelo OCR confirma.
"""

import re
from difflib import SequenceMatcher

import cv2
import numpy as np

# grupos de linhas igualmente espaçadas: (nome, unidade, casas decimais, faixa plausível)
GROUPS = [
    [
        ("WBC", "10³/mm³", 1, (0, 300)),
        ("LIN%", "%", 1, (0, 100)),
        ("MON%", "%", 1, (0, 100)),
        ("GRA%", "%", 1, (0, 100)),
    ],
    [
        ("LIN#", "10³/mm³", 2, (0, 300)),
        ("MON#", "10³/mm³", 2, (0, 300)),
        ("GRA#", "10³/mm³", 2, (0, 300)),
    ],
    [
        ("RBC", "10⁶/mm³", 2, (0, 12)),
        ("HGB", "g/dL", 1, (0, 30)),
        ("HCT", "%", 1, (0, 90)),
        ("VCM", "µm³", 0, (30, 160)),
        ("HCM", "pg", 1, (5, 60)),
        ("CHCM", "g/dL", 1, (15, 50)),
        ("RDW", "%", 1, (3, 50)),
    ],
    [
        ("PLT", "10³/mm³", 0, (0, 3000)),
        ("VPM", "µm³", 1, (2, 30)),
        ("PCT", "%", 3, (0, 3)),
        ("PDW", "%", 1, (2, 40)),
    ],
]

HEADER_KEYS = ["ID amostra", "ID paciente", "Sobrenome", "Data", "Tempo"]

PURE_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _key(s: str) -> str:
    s = s.replace("º", "o").replace("°", "o").upper()
    return re.sub(r"[^A-Z0-9%#]", "", s)


def _match_label(text: str) -> tuple[int, int] | None:
    """Retorna (grupo, linha) se o texto começa com um dos rótulos."""
    k = _key(text)
    best, best_sc = None, 0.0
    for g, rows in enumerate(GROUPS):
        for i, (name, *_rest) in enumerate(rows):
            n = _key(name)
            sc = 1.0 if k.startswith(n) else SequenceMatcher(None, k[: len(n)], n).ratio()
            if sc > best_sc:
                best, best_sc = (g, i), sc
    return best if best_sc >= 0.75 else None


def _text_color_flag(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    """Valor em azul = L, em vermelho = H, preto = normal."""
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = gray < np.percentile(gray, 10)
    if ink.sum() < 5:
        return ""
    # diferença tinta - fundo da caixa: anula o tom (balanço de branco) da foto
    bg = np.median(crop.reshape(-1, 3), axis=0)
    db, dg, dr = crop[ink].astype(float).mean(axis=0) - bg
    if db - dr > 15:
        return "L"
    if dr - db > 15 and dr - dg > 15:
        return "H"
    return ""


def read_abx(reader, img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    boxes = reader.detect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    slopes = [
        (b.pts[1, 1] - b.pts[0, 1]) / (b.pts[1, 0] - b.pts[0, 0])
        for b in boxes
        if b.x1 - b.x0 > 60
    ]
    slope = float(np.median(slopes)) if slopes else 0.0

    def yp(x, y):
        return y - slope * x

    # âncoras: rótulos reconhecidos
    anchors: dict[int, dict[int, tuple[float, float]]] = {}
    for b in boxes:
        m = _match_label(b.text)
        if m and not PURE_NUM.fullmatch(b.text.strip()):
            g, i = m
            anchors.setdefault(g, {}).setdefault(i, (yp(b.x0, b.cy), b.x0))

    # espaçamento entre linhas a partir de pares de rótulos do mesmo grupo
    steps = []
    for rows in anchors.values():
        items = sorted(rows.items())
        for (i1, (y1, _)), (i2, (y2, _)) in zip(items, items[1:]):
            steps.append((y2 - y1) / (i2 - i1))
    spacing = float(np.median(steps)) if steps else 0.03 * H

    numbers = [b for b in boxes if PURE_NUM.fullmatch(b.text.strip())]
    letters = [b for b in boxes if b.text.strip() in ("H", "L")]

    resultados = []
    for g, rows in enumerate(GROUPS):
        ga = anchors.get(g, {})
        if ga:
            idx = np.array(sorted(ga), float)
            ys = np.array([ga[int(i)][0] for i in idx])
            if len(idx) >= 2:
                coef = np.polyfit(idx, ys, 1)
            else:
                coef = np.array([spacing, ys[0] - spacing * idx[0]])
            label_x = float(np.median([ga[i][1] for i in ga]))
        for i, (name, unit, dec, (lo, hi)) in enumerate(rows):
            item = {"parametro": name, "valor": "", "unidade": unit, "flag": "", "confianca": "baixa"}
            resultados.append(item)
            if not ga:
                continue
            yrow = float(np.polyval(coef, i))
            vx0, vx1 = label_x + 0.10 * W, label_x + 0.27 * W

            def on_row(b):
                return abs(yp(b.x0, b.cy) - yrow) < 0.5 * spacing

            cands = []
            row_nums = [b for b in numbers if vx0 <= b.cx <= vx1 and on_row(b)]
            box = min(row_nums, key=lambda b: abs(yp(b.x0, b.cy) - yrow)) if row_nums else None
            if box is not None:
                v = reader.parse_number(box.text, dec)
                if v:
                    cands.append((v, box.score, "det"))
                bx0, bx1 = int(box.x0), int(box.x1)
                by0, by1 = int(box.pts[:, 1].min()), int(box.pts[:, 1].max())
                pad = int(0.15 * (by1 - by0))
                cx0, cx1, cy0, cy1 = bx0 - pad, bx1 + pad, by0 - pad, by1 + pad
            else:
                ymid = yrow + slope * (vx0 + vx1) / 2
                bx0, bx1 = int(vx0), int(vx1)
                by0, by1 = int(ymid - 0.45 * spacing), int(ymid + 0.45 * spacing)
                cx0, cx1, cy0, cy1 = bx0, bx1, by0, by1

            # releitura da célula ampliada: segunda opinião independente da detecção
            crop = gray[max(0, cy0):min(H, cy1), max(0, cx0):min(W, cx1)]
            if crop.size and crop.shape[0] > 8:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                txt, sc = reader.recognize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
                v = reader.parse_number(txt, dec)
                if v:
                    cands.append((v, sc, "rec"))

            cands = [reader.fix_decimal(c, dec, lo, hi) for c in cands]
            if cands:
                def rank(c):
                    nd = len(c[0].split(".")[1]) if "." in c[0] else 0
                    return (c[1] + 0.2 * sum(o[0] == c[0] for o in cands)
                            + 0.3 * (nd == dec) + 0.3 * (lo <= float(c[0]) <= hi))
                v, sc, src = max(cands, key=rank)
                nd = len(v.split(".")[1]) if "." in v else 0
                ok = lo <= float(v) <= hi and nd == dec and src != "fix"
                agree = len(cands) > 1 and all(c[0] == v for c in cands)
                item["valor"] = v
                if ok and agree and sc > 0.9:
                    item["confianca"] = "alta"
                elif ok and sc > 0.7:
                    item["confianca"] = "media"

            # flag: cor do valor; a letra lida pelo OCR na mesma linha tem prioridade
            letter = [b for b in letters if bx1 < b.cx < bx1 + 0.12 * W and on_row(b)]
            if letter:
                item["flag"] = letter[0].text.strip()
            elif item["valor"]:
                item["flag"] = _text_color_flag(img, bx0, by0, bx1, by1)

    return {
        "cabecalho": reader._header(boxes, HEADER_KEYS),
        "resultados": resultados,
        "texto": [b.text for b in sorted(boxes, key=lambda b: (round(b.cy / 15), b.cx))],
    }

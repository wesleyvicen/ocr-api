"""Leitura de telas de analisador hematológico ("Rev. histog. amostras").

Pipeline 100% local, sem serviço externo:
  1. Retifica perspectiva usando a área azul da tela.
  2. Realça texto (canal vermelho invertido + CLAHE): texto amarelo/branco sobre azul vira preto sobre branco.
  3. RapidOCR (modelos PaddleOCR em ONNX) detecta e lê o texto.
  4. Rótulos encontrados servem de âncora para um modelo de posição das linhas.
  5. Cada célula de valor é relida isoladamente; fica a leitura de maior confiança.
  6. Flags H/L (vermelhas) são detectadas por cor e classificadas pela forma.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
from rapidocr import RapidOCR

# (nome exibido, unidade, casas decimais, faixa plausível)
TEMPLATE = [
    ("GB", "x10³/uL", 1, (0, 300)),
    ("NºLinf", "x10³/uL", 1, (0, 300)),
    ("Nº méd", "x10³/uL", 1, (0, 300)),
    ("NºGran", "x10³/uL", 1, (0, 300)),
    ("%Linf", "%", 1, (0, 100)),
    ("% méd", "%", 1, (0, 100)),
    ("%Gran", "%", 1, (0, 100)),
    ("Hb", "g/dL", 1, (0, 30)),
    ("GV", "x10⁶/uL", 2, (0, 12)),
    ("Ht", "%", 1, (0, 90)),
    ("VCM", "fL", 1, (30, 160)),
    ("HCM", "pg", 1, (5, 60)),
    ("CHCM", "g/dL", 1, (15, 50)),
    ("ADVGVC", "%", 1, (3, 50)),
    ("ADVGSD", "fL", 1, (10, 150)),
    ("PLT", "x10³/uL", 0, (0, 3000)),
    ("VPM", "fL", 1, (2, 30)),
    ("ADP", "", 1, (2, 40)),
    ("PCT", "%", 3, (0, 3)),
]

HEADER_KEYS = [
    "ID", "Hora", "Local/Total", "Modo", "Nome", "Sexo", "Idade", "Testador",
    "Remetente", "Graf. nº", "Depto.", "Verificador", "Nº. leito",
]

NUM_RE = re.compile(r"^\s*(\d+)(?:\s*[.,]\s*(\d+)|\s+(\d)(?!\d))?")


@dataclass
class Box:
    text: str
    score: float
    pts: np.ndarray  # 4x2: tl, tr, br, bl

    @property
    def x0(self): return float(self.pts[:, 0].min())
    @property
    def x1(self): return float(self.pts[:, 0].max())
    @property
    def cx(self): return float(self.pts[:, 0].mean())
    @property
    def cy(self): return float(self.pts[:, 1].mean())
    @property
    def h(self): return float(self.pts[:, 1].max() - self.pts[:, 1].min())


def _norm(s: str) -> str:
    s = s.replace("º", "o").replace("°", "o")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9%]", "", s)


_LABELS = [_norm(t[0]) for t in TEMPLATE]


class Reader:
    def __init__(self):
        self.engine = RapidOCR(params={"Global.log_level": "error"})

    # ---------- imagem ----------

    @staticmethod
    def rectify(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (95, 120, 60), (130, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return img, (0, w)
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.25 * h * w:
            return img, (0, w)
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in np.linspace(0.01, 0.1, 20):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            return img, (0, w)
        s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
        q = np.array([quad[s.argmin()], quad[d.argmin()], quad[s.argmax()], quad[d.argmax()]], np.float32)
        W = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
        H = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
        # margem: a foto pode cortar a tela, mantém o que ficou fora do quadrilátero
        mx, my = int(0.06 * W), int(0.08 * H)
        dst = np.float32([[mx, my], [mx + W, my], [mx + W, my + H], [mx, my + H]])
        M = cv2.getPerspectiveTransform(q, dst)
        out = cv2.warpPerspective(img, M, (W + 2 * mx, H + 2 * my), borderValue=(0, 0, 0))
        return out, (mx, W)

    @staticmethod
    def enhance(img: np.ndarray) -> np.ndarray:
        r = img[:, :, 2]
        r = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(r)
        return cv2.cvtColor(255 - r, cv2.COLOR_GRAY2BGR)

    # ---------- OCR ----------

    def detect(self, img: np.ndarray) -> list[Box]:
        res = self.engine(img, use_det=True, use_cls=True, use_rec=True, text_score=0.3, box_thresh=0.4)
        if res.boxes is None:
            return []
        return [Box(t, float(s), np.asarray(b, np.float32)) for b, t, s in zip(res.boxes, res.txts, res.scores)]

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        res = self.engine(crop, use_det=False, use_cls=False, use_rec=True)
        if not res.txts:
            return "", 0.0
        return res.txts[0], float(res.scores[0])

    # ---------- parsing ----------

    @staticmethod
    def parse_number(text: str, decimals: int) -> str | None:
        m = NUM_RE.match(text.replace("O", "0").replace("o", "0"))
        if not m:
            return None
        whole, frac, spaced = m.group(1), m.group(2), m.group(3)
        if frac is None and spaced is not None and decimals > 0:
            frac = spaced  # "10 0" -> 10.0 (ponto perdido)
        return f"{whole}.{frac}" if frac else whole

    @staticmethod
    def fix_decimal(cand, decimals, lo, hi):
        """Ajusta casas decimais ao modelo da tela; ponto perdido ("84" no VPM) só é inserido se cabe na faixa."""
        v, sc, src = cand
        if "." in v and len(v.split(".")[1]) > decimals:
            # dígito extra colado (ex.: "7.7 fL" lido como "7.71")
            whole, frac = v.split(".")
            return (f"{whole}.{frac[:decimals]}" if decimals else whole), sc, "fix"
        if "." in v or decimals == 0 or len(v) <= decimals or lo <= float(v) <= hi:
            return cand
        fixed = f"{v[:-decimals]}.{v[-decimals:]}"
        return (fixed, sc, "fix") if lo <= float(fixed) <= hi else cand

    @staticmethod
    def label_index(text: str) -> int | None:
        n = _norm(text)
        if not n:
            return None
        scores = [SequenceMatcher(None, n, lab).ratio() for lab in _LABELS]
        best = int(np.argmax(scores))
        ranked = sorted(scores, reverse=True)
        if ranked[0] >= 0.75 and ranked[0] - ranked[1] > 0.1:
            return best
        return None

    def read(self, raw: bytes) -> dict:
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagem inválida")
        if max(img.shape[:2]) > 2000:
            f = 2000 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

        img, (ox, sw) = self.rectify(img)
        enh = self.enhance(img)
        H, W = img.shape[:2]
        boxes = self.detect(enh)

        def X(r):  # fração da largura da tela -> pixel
            return ox + r * sw

        # inclinação local das linhas de texto
        slopes = [
            (b.pts[1, 1] - b.pts[0, 1]) / (b.pts[1, 0] - b.pts[0, 0])
            for b in boxes
            if b.x1 - b.x0 > 60 and b.cx < X(0.6)
        ]
        slope = float(np.median(slopes)) if slopes else 0.0

        def yp(x, y):  # coordenada de linha, sem a inclinação
            return y - slope * x

        # rótulos -> âncoras (índice da linha, y')
        label_col = [b for b in boxes if b.cx < X(0.15)]
        anchors = []
        for b in label_col:
            i = self.label_index(b.text)
            if i is not None:
                anchors.append((i, yp(b.cx, b.cy)))

        # valores numéricos na coluna de valores
        numeric = [b for b in boxes if X(0.12) < b.x0 < X(0.45) and re.match(r"^\s*\d", b.text)]
        if numeric:
            col_left = float(np.median([b.x0 for b in numeric]))
            numeric = [b for b in numeric if b.x0 < col_left + 0.04 * sw]

        model = self._row_model(anchors, numeric, yp)
        if model is None:
            raise ValueError("Não foi possível localizar a tabela de resultados na imagem")
        row_y, spacing = model

        # coluna de valores
        vx0 = min(b.x0 for b in numeric) - 0.01 * sw if numeric else X(0.18)
        vx1 = vx0 + 0.18 * sw

        # apaga do mapa de flags o que o OCR reconheceu como número/unidade
        text_mask = np.zeros((H, W), np.uint8)
        for b in boxes:
            t = b.text.strip()
            if re.search(r"[\d/%×xXuUdDfFpPgG]", t):  # parece número ou unidade
                cv2.fillPoly(text_mask, [b.pts.astype(np.int32)], 1)
        text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8))
        flags = self._flags(img, text_mask, int(vx0 + 0.12 * sw), int(X(0.47)), yp, row_y, spacing)

        resultados = []
        for i, (nome, unidade, dec, (lo, hi)) in enumerate(TEMPLATE):
            yrow = row_y(i)
            cands = []
            row_boxes = sorted(
                (b for b in boxes if vx0 <= b.cx <= vx1 and abs(yp(b.cx, b.cy) - yrow) < 0.5 * spacing),
                key=lambda b: b.x0,
            )
            if row_boxes:
                v = self.parse_number(" ".join(b.text for b in row_boxes).replace(" .", ".").replace(". ", "."), dec)
                if v:
                    cands.append((v, min(b.score for b in row_boxes), "det"))
            # releitura da célula isolada
            ymid = yrow + slope * (vx0 + vx1) / 2
            y0, y1 = int(ymid - 0.55 * spacing), int(ymid + 0.55 * spacing)
            if y1 > 0 and y0 < H:
                crop = enh[max(0, y0):min(H, y1), max(0, int(vx0)):min(W, int(vx1))]
                if crop.size and crop.shape[0] > 8:
                    txt, sc = self.recognize(crop)
                    v = self.parse_number(txt, dec)
                    if v:
                        cands.append((v, sc, "rec"))

            valor, conf = "", "baixa"
            cands = [self.fix_decimal(c, dec, lo, hi) for c in cands]
            if cands:
                # concordância entre leituras aumenta a confiança
                def rank(c):
                    nd = len(c[0].split(".")[1]) if "." in c[0] else 0
                    return (c[1] + 0.2 * sum(o[0] == c[0] for o in cands)
                            + 0.3 * (nd == dec) + 0.3 * (lo <= float(c[0]) <= hi))
                best = max(cands, key=rank)
                valor, sc = best[0], best[1]
                agree = sum(o[0] == valor for o in cands) > 1
                ndec = len(valor.split(".")[1]) if "." in valor else 0
                ok = lo <= float(valor) <= hi and ndec == dec
                fixed = any(c[2] == "fix" for c in cands if c[0] == valor)
                if ok and not fixed and sc > 0.9 and (agree or len(cands) == 1):
                    conf = "alta"
                elif ok and sc > 0.7:
                    conf = "media"

            resultados.append({
                "parametro": nome,
                "valor": valor,
                "unidade": unidade,
                "flag": flags.get(i, ""),
                "confianca": conf,
            })

        return {
            "cabecalho": self._header(boxes),
            "resultados": resultados,
            "texto": [b.text for b in sorted(boxes, key=lambda b: (round(b.cy / 15), b.cx))],
        }

    # ---------- modelos auxiliares ----------

    @staticmethod
    def _row_model(anchors, numeric, yp):
        """Retorna (f(i) -> y', espaçamento).

        A coluna de valores define as linhas; o índice da primeira linha visível é escolhido
        pela plausibilidade dos números (casas decimais, faixa) e pelos rótulos lidos.
        """
        if len(numeric) < 3:
            return None
        # agrupa caixas da mesma linha (valor fragmentado: "16." + "2")
        h = float(np.median([b.h for b in numeric]))
        numeric = [b for b in numeric if b.h < 1.6 * h]  # caixa que engoliu duas linhas
        items = sorted(((yp(b.cx, b.cy), b) for b in numeric), key=lambda t: t[0])
        rows = [[items[0]]]
        for y, b in items[1:]:
            if y - rows[-1][-1][0] < 0.5 * h:
                rows[-1].append((y, b))
            else:
                rows.append([(y, b)])
        ys = [float(np.mean([y for y, _ in r])) for r in rows]
        texts = [" ".join(b.text for _, b in sorted(r, key=lambda t: t[1].x0)) for r in rows]
        if len(ys) < 3:
            return None

        # índices relativos: espaçamento varia pouco com a perspectiva
        diffs = np.diff(ys)
        sp = float(np.median(diffs))
        if len(diffs) >= 4:
            sp = float(np.median(diffs[diffs < 1.5 * np.min(diffs[diffs > 0.5 * sp])]))
        rel = [0]
        for d in diffs:
            n = max(1, int(round(d / sp)))
            rel.append(rel[-1] + n)
            sp = 0.7 * sp + 0.3 * (d / n)
        n_rows = len(TEMPLATE)
        if rel[-1] >= n_rows:
            # mais linhas que o modelo: mantém as primeiras
            keep = [k for k, r in enumerate(rel) if r < n_rows]
            ys, texts, rel = [ys[k] for k in keep], [texts[k] for k in keep], [rel[k] for k in keep]

        def score(off):
            sc = 0.0
            for r, t in zip(rel, texts):
                _, _, dec, (lo, hi) = TEMPLATE[r + off]
                v = Reader.parse_number(t, dec)
                if v is None:
                    continue
                nd = len(v.split(".")[1]) if "." in v else 0
                sc += (nd == dec) + (lo <= float(v) <= hi)
            for i, ya in anchors:
                r = i - off
                if r in rel and abs(ys[rel.index(r)] - ya) < 0.6 * sp:
                    sc += 1.0
            return sc - 0.01 * off  # empate: prefere começar no topo

        off = max(range(n_rows - rel[-1]), key=score)
        idx = [r + off for r in rel]

        # rótulos entram no ajuste, corrigidos pelo desnível rótulo -> valor
        pairs = [ys[idx.index(i)] - ya for i, ya in anchors if i in idx]
        shift = float(np.median(pairs)) if pairs else 0.0
        pts = list(zip(idx, ys)) + [(i, ya + shift) for i, ya in anchors]
        pts = np.array(pts, float)
        for _ in range(3):
            coef = np.polyfit(pts[:, 0], pts[:, 1], 2 if len(pts) >= 8 else 1)
            res = np.abs(np.polyval(coef, pts[:, 0]) - pts[:, 1])
            keep = res < 0.35 * sp
            if keep.all() or keep.sum() < 3:
                break
            pts = pts[keep]
        poly = np.poly1d(coef)
        spacing = float(abs(poly.deriv()(float(np.mean(pts[:, 0])))))
        return (lambda i: float(poly(i))), spacing

    @staticmethod
    def _flags(img, text_mask, x_lo, x_hi, yp, row_y, spacing) -> dict[int, str]:
        """Flags são letras vermelhas; com moiré/reflexo sobram os traços verticais: 2 traços = H, 1 = L."""
        b, g, r = (img[:, :, k].astype(np.int16) for k in range(3))
        # flag é rosa/lilás (R >= G); texto branco das unidades puxa para verde (G > R); fundo tem R baixo
        red = ((r - g >= 0) & (r > 70)).astype(np.uint8)
        red[text_mask > 0] = 0
        red[:, :x_lo] = 0
        red[:, x_hi:] = 0
        ys, xs = np.nonzero(red)
        if not len(xs):
            return {}
        ry = yp(xs.astype(float), ys.astype(float))
        centers = np.array([row_y(j) for j in range(len(TEMPLATE))])
        out = {}
        for i, c in enumerate(centers):
            sel = np.abs(ry - c) < 0.4 * spacing
            if sel.sum() < 6:
                continue
            cols = np.bincount(xs[sel] - x_lo, minlength=x_hi - x_lo)
            # coluna conta como traço se tiver altura mínima
            strokes, run = [], None
            for x, v in enumerate(cols >= max(3, 0.15 * spacing)):
                if v and run is None:
                    run = [x, x]
                elif v:
                    run[1] = x
                elif run is not None and x - run[1] > 2:
                    strokes.append(run)
                    run = None
            if run is not None:
                strokes.append(run)
            if not strokes:
                continue
            width = strokes[-1][1] - strokes[0][0]
            out[i] = "H" if len(strokes) >= 2 and width < 0.8 * spacing else "L"
        return out

    @staticmethod
    def _header(boxes: list[Box]) -> list[dict]:
        keys = {_norm(k): k for k in HEADER_KEYS}
        found = {}
        for b in boxes:
            # uma caixa pode juntar dois campos: "Hora : 11-10-2026 07:26Local/Total: 8/35000"
            text = re.sub(r"(?<=\d)\s*(?=[A-Za-z]{3,}[^:]*:)", " | ", b.text)
            for part in text.split(" | "):
                if ":" not in part:
                    continue
                k, _, v = part.partition(":")
                nk = _norm(k)
                if not nk:
                    continue
                best = max(keys, key=lambda kk: SequenceMatcher(None, nk, kk).ratio())
                if SequenceMatcher(None, nk, best).ratio() >= 0.7 and keys[best] not in found:
                    found[keys[best]] = v.strip().lstrip(":.").strip()
        return [{"campo": k, "valor": found.get(k, "")} for k in HEADER_KEYS]

#!/usr/bin/env python3
"""Генератор рукописных подписей по ФИО (stdlib-only).

Идея:
- Скелет подписи строится по символам фамилии/инициалов.
- Стиль (наклон, амплитуда, характер штрихов) зависит от хэша ФИО,
  поэтому для разных людей подписи заметно разные, но стабильные при одном seed.
- Есть опциональная оценка похожести на датасет `popisi`.
"""

from __future__ import annotations

import glob
import hashlib
import math
import os
import random
import struct
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class SignatureConfig:
    first_name: str = "Иван"
    last_name: str = "Петров"
    middle_name: str = "Сергеевич"
    use_full_last_name: bool = True
    seed: int = 7

    width: int = 768
    height: int = 768
    background_gray: int = 224
    stroke_gray: int = 8
    stroke_width: int = 6

    output_path: str = "generated/signature.png"
    examples_dir: str = "popisi"
    run_comparison: bool = False
    comparison_sample_size: int = 6


# -------- PNG I/O (stdlib) --------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc & 0xFFFFFFFF)


def save_grayscale_png(path: str, pixels: List[List[int]]) -> None:
    h = len(pixels)
    w = len(pixels[0]) if h else 0

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        raw.extend(max(0, min(255, int(v))) for v in row)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 0, 0, 0, 0)
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += _png_chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def load_grayscale_png(path: str) -> List[List[int]]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not PNG: {path}")

    i = 8
    w = h = None
    bit_depth = color_type = interlace = None
    idat = bytearray()

    while i < len(data):
        ln = struct.unpack("!I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        cdata = data[i + 8 : i + 8 + ln]
        i += ln + 12

        if ctype == b"IHDR":
            w, h, bit_depth, color_type, _, _, interlace = struct.unpack("!IIBBBBB", cdata)
        elif ctype == b"IDAT":
            idat.extend(cdata)
        elif ctype == b"IEND":
            break

    if w is None or h is None:
        raise ValueError("Invalid PNG")
    if interlace != 0:
        raise ValueError("Interlaced PNG unsupported")

    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported color type: {color_type}")
    if bit_depth not in (8, 16):
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    channels = channels_by_type[color_type]
    bps = bit_depth // 8
    bpp = channels * bps

    raw = zlib.decompress(bytes(idat))
    stride = 1 + w * bpp

    rows_raw: List[List[int]] = []
    prev = [0] * (w * bpp)

    for y in range(h):
        row = list(raw[y * stride : (y + 1) * stride])
        filt = row[0]
        cur = row[1:]

        if filt == 1:  # sub
            for x in range(w * bpp):
                cur[x] = (cur[x] + (cur[x - bpp] if x >= bpp else 0)) & 0xFF
        elif filt == 2:  # up
            for x in range(w * bpp):
                cur[x] = (cur[x] + prev[x]) & 0xFF
        elif filt == 3:  # average
            for x in range(w * bpp):
                left = cur[x - bpp] if x >= bpp else 0
                cur[x] = (cur[x] + ((left + prev[x]) // 2)) & 0xFF
        elif filt == 4:  # paeth
            def paeth(a: int, b: int, c: int) -> int:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    return a
                if pb <= pc:
                    return b
                return c

            for x in range(w * bpp):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                cur[x] = (cur[x] + paeth(a, b, c)) & 0xFF
        elif filt != 0:
            raise ValueError(f"Unsupported PNG filter: {filt}")

        rows_raw.append(cur)
        prev = cur

    out: List[List[int]] = []
    for rb in rows_raw:
        row_gray: List[int] = []
        for x in range(w):
            i0 = x * bpp
            if color_type in (0, 4):
                g = rb[i0]
            else:
                r = rb[i0]
                g = rb[i0 + bps]
                b = rb[i0 + 2 * bps]
                g = int(0.299 * r + 0.587 * g + 0.114 * b)
            row_gray.append(g)
        out.append(row_gray)
    return out


# -------- Geometry --------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").strip().lower())
    return "".join(ch for ch in text if ch.isalpha())


def sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> List[Point]:
    pts: List[Point] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _name_fingerprint(cfg: SignatureConfig) -> int:
    raw = f"{cfg.last_name}|{cfg.first_name}|{cfg.middle_name}|{cfg.seed}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _char_params(ch: str, pos: int, rnd: random.Random) -> Tuple[float, float, float, int]:
    code = ord(ch)
    family = (code + pos * 7) % 4  # 4 архетипа символов
    horiz = 18 + ((code * 19 + pos * 23) % 35)
    amp = 8 + ((code * 11 + pos * 17) % 28)
    sharp = 0.3 + (((code * 7 + pos * 13) % 100) / 100.0)
    amp *= rnd.uniform(0.9, 1.15)
    return horiz, amp, sharp, family


def build_signature_path(cfg: SignatureConfig) -> List[Point]:
    first = normalize_text(cfg.first_name)
    last = normalize_text(cfg.last_name)
    middle = normalize_text(cfg.middle_name)

    key = last if (cfg.use_full_last_name and last) else (last[:1] + first[:1] + middle[:1])
    if not key:
        key = "x"

    fp = _name_fingerprint(cfg)
    rnd = random.Random(fp)

    # Глобальный стиль зависит от ФИО
    slant = rnd.uniform(-0.22, 0.28)
    baseline = cfg.height * rnd.uniform(0.52, 0.66)
    start_x = cfg.width * rnd.uniform(0.12, 0.26)
    capital_size = 1.0 + (len(last) % 5) * 0.11 + rnd.uniform(-0.08, 0.1)

    path: List[Point] = []

    # Стартовая «буква»: 3 семейства, чтобы начало не было однотипным.
    fst = ord(key[0])
    family0 = fst % 3
    loop_w = (34 + (fst % 28)) * capital_size
    loop_h = (85 + (fst % 62)) * capital_size
    p0 = (start_x, baseline)
    if family0 == 0:
        # Левый заход и возврат как у «Л/П»-подобного старта.
        p1 = (start_x - loop_w * rnd.uniform(0.35, 0.9), baseline - loop_h * rnd.uniform(0.45, 0.85))
        p2 = (start_x - loop_w * rnd.uniform(0.1, 0.35), baseline + loop_h * rnd.uniform(0.4, 0.95))
        p3 = (start_x + loop_w * rnd.uniform(0.55, 1.0), baseline + loop_h * rnd.uniform(-0.05, 0.3))
    elif family0 == 1:
        # Более вертикальный старт как у «И/Н/М».
        p1 = (start_x + loop_w * rnd.uniform(-0.1, 0.18), baseline - loop_h * rnd.uniform(0.75, 1.25))
        p2 = (start_x + loop_w * rnd.uniform(0.05, 0.35), baseline + loop_h * rnd.uniform(0.2, 0.55))
        p3 = (start_x + loop_w * rnd.uniform(0.65, 1.1), baseline + loop_h * rnd.uniform(-0.2, 0.15))
    else:
        # Узкая петля и плавный выход как у «С/О/Э»-подобного начала.
        p1 = (start_x - loop_w * rnd.uniform(0.2, 0.45), baseline - loop_h * rnd.uniform(0.8, 1.2))
        p2 = (start_x + loop_w * rnd.uniform(-0.15, 0.2), baseline + loop_h * rnd.uniform(0.6, 1.05))
        p3 = (start_x + loop_w * rnd.uniform(0.45, 0.9), baseline + loop_h * rnd.uniform(-0.15, 0.2))
    path.extend(sample_cubic(p0, p1, p2, p3, 44))

    x, y = p3
    density = 1 + (len(key) > 6)

    for i, ch in enumerate(key):
        horiz, amp, sharp, family = _char_params(ch, i, rnd)
        dx = horiz * rnd.uniform(0.95, 1.35)
        dy = rnd.uniform(-10, 10) * (0.7 + sharp)

        if family == 0:  # дуга вверх-вниз
            c1 = (x + dx * 0.25, y - amp * 1.5)
            c2 = (x + dx * 0.7, y + amp * 0.9)
        elif family == 1:  # более острая "галка"
            c1 = (x + dx * 0.3, y - amp * (1.9 + sharp * 0.6))
            c2 = (x + dx * 0.55, y + amp * 1.2)
        elif family == 2:  # плоская волна
            c1 = (x + dx * 0.2, y - amp * 0.8)
            c2 = (x + dx * 0.75, y + amp * 0.45)
        else:  # длинная тянущаяся связка
            c1 = (x + dx * 0.35, y - amp * 1.1)
            c2 = (x + dx * 0.9, y + amp * 0.2)

        # Держим более «человечную» скорость движения: меньше резких волн.
        end = (x + dx, y + dy * 0.7)
        path.extend(sample_cubic((x, y), c1, c2, end, 14 + density * 4))

        # Центральные микро-штрихи (индивидуальная "дребезжащая" середина)
        if i < len(key) - 1 and rnd.random() < 0.75:
            spikes = 1 + (1 if rnd.random() < 0.45 else 0)
            for _ in range(spikes):
                h = rnd.uniform(25, 95) * (0.7 + sharp)
                up = (end[0] + rnd.uniform(2, 10), end[1] - h)
                back = (up[0] + rnd.uniform(3, 12), end[1] + rnd.uniform(-6, 6))
                path.extend(sample_cubic(end, (end[0] + 2, end[1] - h * 0.28), (up[0] - 2, up[1] + h * 0.2), up, 8))
                path.extend(sample_cubic(up, (up[0] + 1, up[1] + h * 0.22), (back[0] - 2, back[1] - h * 0.25), back, 8))
                end = back

        # Иногда добавляем вертикальный акцент в середине/конце
        if i in {len(key) // 2, max(1, len(key) - 2)} and rnd.random() < 0.55:
            top = (end[0] + rnd.uniform(-6, 6), end[1] - rnd.uniform(120, 240))
            ret = (top[0] + rnd.uniform(4, 12), end[1] + rnd.uniform(10, 70))
            path.extend(sample_cubic(end, (end[0], end[1] - 45), (top[0], top[1] + 25), top, 16))
            path.extend(sample_cubic(top, (top[0] + 2, top[1] + 38), (ret[0] - 3, ret[1] - 28), ret, 14))
            end = ret

        x, y = end

    # Финальный хвост
    tail_len = rnd.uniform(35, 90)
    tail_rise = rnd.uniform(4, 38)
    tail_end = (x + tail_len, y - tail_rise)
    path.extend(sample_cubic((x, y), (x + tail_len * 0.2, y + rnd.uniform(-20, 30)), (x + tail_len * 0.6, y - tail_rise * 1.05), tail_end, 25))

    # Небольшой шум и наклон
    cx = sum(px for px, _ in path) / len(path)
    cy = sum(py for _, py in path) / len(path)
    warped: List[Point] = []
    for px, py in path:
        px += rnd.uniform(-1.5, 1.5)
        py += rnd.uniform(-1.5, 1.5)
        dx, dy = px - cx, py - cy
        warped.append((cx + dx + dy * slant, cy + dy))

    # Нормализация размера/позиции под типичный вид датасета
    min_x = min(p[0] for p in warped)
    max_x = max(p[0] for p in warped)
    min_y = min(p[1] for p in warped)
    max_y = max(p[1] for p in warped)
    bw, bh = max_x - min_x, max_y - min_y

    target_w = cfg.width * rnd.uniform(0.30, 0.40)
    target_h = cfg.height * rnd.uniform(0.72, 0.94)
    scale = min(target_w / max(1.0, bw), target_h / max(1.0, bh))

    shifted = [((px - min_x) * scale, (py - min_y) * scale) for px, py in warped]

    # Корректируем вытянутость: в примерах подписи обычно выше и компактнее по ширине.
    sx_min = min(px for px, _ in shifted)
    sx_max = max(px for px, _ in shifted)
    sy_min = min(py for _, py in shifted)
    sy_max = max(py for _, py in shifted)
    cur_aspect = (sx_max - sx_min) / max(1.0, (sy_max - sy_min))
    if cur_aspect > 1.05:
        y_boost = min(2.2, cur_aspect / 0.95)
        shifted = [(px, py * y_boost) for px, py in shifted]
    elif cur_aspect < 0.55:
        x_boost = min(1.8, 0.7 / max(0.01, cur_aspect))
        shifted = [(px * x_boost, py) for px, py in shifted]
    max_sx = max(px for px, _ in shifted)
    max_sy = max(py for _, py in shifted)
    free_x = max(1.0, cfg.width - max_sx - 24.0)
    free_y = max(1.0, cfg.height - max_sy - 24.0)
    sx = 12.0 + rnd.uniform(0.32, 0.56) * free_x
    sy = 12.0 + rnd.uniform(0.35, 0.62) * free_y

    return [(px + sx, py + sy) for px, py in shifted]


def _draw_disk(canvas: List[List[int]], x: int, y: int, r: int, color: int) -> None:
    h = len(canvas)
    w = len(canvas[0])
    rr = r * r
    for yy in range(y - r, y + r + 1):
        if not (0 <= yy < h):
            continue
        row = canvas[yy]
        for xx in range(x - r, x + r + 1):
            if 0 <= xx < w and (xx - x) ** 2 + (yy - y) ** 2 <= rr:
                if color < row[xx]:
                    row[xx] = color


def draw_polyline(canvas: List[List[int]], points: Sequence[Point], stroke_width: int, color: int) -> None:
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for s in range(n + 1):
            t = s / max(1, n)
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            _draw_disk(canvas, x, y, max(1, stroke_width // 2), color)


def generate_signature_image(cfg: SignatureConfig) -> List[List[int]]:
    img = [[cfg.background_gray for _ in range(cfg.width)] for _ in range(cfg.height)]
    points = build_signature_path(cfg)
    draw_polyline(img, points, cfg.stroke_width, cfg.stroke_gray)
    return img


# -------- Comparison with dataset --------

def extract_features(img: List[List[int]], thr: int = 180, step: int = 2) -> dict:
    h = len(img)
    w = len(img[0])
    pts: List[Tuple[int, int]] = []

    for y in range(0, h, step):
        row = img[y]
        for x in range(0, w, step):
            if row[x] < thr:
                pts.append((x, y))

    if not pts:
        return {
            "ink_ratio": 0.0,
            "bbox_aspect": 1.0,
            "bbox_area_ratio": 0.0,
            "center_x": 0.5,
            "center_y": 0.5,
        }

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    sw = max(1, w // step)
    sh = max(1, h // step)

    return {
        "ink_ratio": len(pts) / (sw * sh),
        "bbox_aspect": bbox_w / max(1, bbox_h),
        "bbox_area_ratio": (bbox_w * bbox_h) / (sw * sh),
        "center_x": (sum(xs) / len(xs)) / w,
        "center_y": (sum(ys) / len(ys)) / h,
    }


def estimate_human_likeness(img: List[List[int]]) -> Tuple[float, dict]:
    """Быстрая оценка человекоподобности без чтения всего датасета."""
    f = extract_features(img, step=8)
    # Профиль приближен по наблюдаемым статистикам датасета popisi.
    target = {
        "ink_ratio": (0.0247, 0.0070, 0.010),
        "bbox_aspect": (0.8425, 0.1882, 0.35),
        "bbox_area_ratio": (13.6414, 5.0582, 6.0),
        "center_x": (0.4017, 0.0614, 0.09),
        "center_y": (0.5262, 0.0288, 0.08),
    }

    z_sum = 0.0
    details = {}
    for k, (mu, sd, floor) in target.items():
        z = abs((f[k] - mu) / max(sd, floor))
        details[k] = {"generated": f[k], "mean": mu, "std": sd, "z": z}
        z_sum += z

    score = max(0.0, 100.0 - (z_sum / len(target)) * 24.0)
    return score, details


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = list(values)
    if not arr:
        return 0.0, 1.0
    mu = sum(arr) / len(arr)
    var = sum((v - mu) ** 2 for v in arr) / len(arr)
    return mu, max(1e-6, math.sqrt(var))


def compare_with_examples(generated: List[List[int]], examples_dir: str, sample_size: int) -> Tuple[float, dict]:
    paths = sorted(glob.glob(os.path.join(examples_dir, "*.png")))
    random.Random(123).shuffle(paths)
    paths = paths[:sample_size]

    dataset = []
    for p in paths:
        try:
            dataset.append(extract_features(load_grayscale_png(p), step=8))
        except Exception:
            continue

    # Важно: одинаковый шаг для датасета и генерации, иначе метрики несопоставимы.
    gen = extract_features(generated, step=8)
    keys = ["ink_ratio", "bbox_aspect", "bbox_area_ratio", "center_x", "center_y"]

    tolerance_floor = {
        "ink_ratio": 0.007,
        "bbox_aspect": 0.25,
        "bbox_area_ratio": 4.0,
        "center_x": 0.06,
        "center_y": 0.06,
    }

    z_sum = 0.0
    details = {}
    for k in keys:
        mu, sd = mean_std(d[k] for d in dataset)
        denom = max(sd, tolerance_floor[k])
        z = abs((gen[k] - mu) / denom)
        details[k] = {"generated": gen[k], "mean": mu, "std": sd, "z": z}
        z_sum += z

    # мягче штрафуем выбросы, чтобы score не падал в 0 от одной метрики
    score = max(0.0, 100.0 - (z_sum / len(keys)) * 22.0)
    return score, details


def main() -> None:
    # ----- CONFIG: меняйте здесь -----
    cfg = SignatureConfig(
        first_name="Алексей",
        last_name="Ковалев",
        middle_name="Игоревич",
        use_full_last_name=True,
        seed=73,
        stroke_width=3,
        output_path="generated/signature_kovalev.png",
        run_comparison=True,
    )
    # -------------------------------

    generated = generate_signature_image(cfg)
    save_grayscale_png(cfg.output_path, generated)
    print(f"Сохранено: {cfg.output_path}")

    if cfg.run_comparison:
        score, details = estimate_human_likeness(generated)
        print(f"Оценка человекоподобности и сходства: {score:.1f}/100")
        for k, v in details.items():
            print(f"  {k}: gen={v['generated']:.4f}, mean={v['mean']:.4f}, std={v['std']:.4f}, z={v['z']:.2f}")


if __name__ == "__main__":
    main()

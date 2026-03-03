@codex внеси правки.

 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/generator.py b/generator.py
index 784cd0f980d5ea40bbb9012a31e92facf499b007..370ce5c1b4745adc6ecf1b9267a4654540acd854 100644
--- a/generator.py
+++ b/generator.py
@@ -10,60 +10,62 @@
 
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
 
-    width: int = 768
-    height: int = 768
+    width: int = 256
+    height: int = 256
     background_gray: int = 224
     stroke_gray: int = 8
     stroke_width: int = 6
 
     output_path: str = "generated/signature.png"
     examples_dir: str = "popisi"
     run_comparison: bool = False
     comparison_sample_size: int = 6
+    trials_count: int = 20
+    examples_limit: int = 20
 
 
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
 
@@ -465,52 +467,209 @@ def compare_with_examples(generated: List[List[int]], examples_dir: str, sample_
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
 
 
+def downsample_darkness(img: List[List[int]], out_w: int = 96, out_h: int = 96) -> List[float]:
+    """Сжатие изображения в компактный вектор "темноты" для shape-сравнения."""
+    h = len(img)
+    w = len(img[0]) if h else 0
+    if h == 0 or w == 0:
+        return [0.0] * (out_w * out_h)
+
+    # Интегральное изображение по "темноте" (быстрее, чем суммировать каждый блок циклами).
+    integ = [[0.0] * (w + 1) for _ in range(h + 1)]
+    for y in range(h):
+        row_acc = 0.0
+        src = img[y]
+        dst = integ[y + 1]
+        prev = integ[y]
+        for x in range(w):
+            row_acc += (255 - src[x]) / 255.0
+            dst[x + 1] = prev[x + 1] + row_acc
+
+    vec: List[float] = []
+    for oy in range(out_h):
+        y0 = int(oy * h / out_h)
+        y1 = int((oy + 1) * h / out_h)
+        if y1 <= y0:
+            y1 = min(h, y0 + 1)
+        for ox in range(out_w):
+            x0 = int(ox * w / out_w)
+            x1 = int((ox + 1) * w / out_w)
+            if x1 <= x0:
+                x1 = min(w, x0 + 1)
+
+            s = integ[y1][x1] - integ[y0][x1] - integ[y1][x0] + integ[y0][x0]
+            vec.append(s / max(1, (y1 - y0) * (x1 - x0)))
+    return vec
+
+
+def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
+    num = sum(x * y for x, y in zip(a, b))
+    den_a = math.sqrt(sum(x * x for x in a))
+    den_b = math.sqrt(sum(y * y for y in b))
+    if den_a == 0.0 or den_b == 0.0:
+        return 0.0
+    return num / (den_a * den_b)
+
+
+def build_examples_index(examples_dir: str, examples_limit: int = 20) -> List[Tuple[str, dict, List[float]]]:
+    """Готовим индекс признаков примеров, чтобы быстро сравнивать много вариантов."""
+    paths = sorted(glob.glob(os.path.join(examples_dir, "*.png")))
+    random.Random(2024).shuffle(paths)
+    paths = paths[:max(1, examples_limit)]
+    out: List[Tuple[str, dict, List[float]]] = []
+    for p in paths:
+        try:
+            img = load_grayscale_png(p)
+            out.append((p, extract_features(img, step=8), downsample_darkness(img, out_w=24, out_h=24)))
+        except Exception:
+            continue
+    return out
+
+
+def similarity_to_index(generated: List[List[int]], examples_index: List[Tuple[str, dict, List[float]]]) -> Tuple[float, str, dict]:
+    """Схожесть с датасетом: статистика + форма (cosine на миниатюре)."""
+    if not examples_index:
+        return 0.0, "", {}
+
+    gen_feat = extract_features(generated, step=8)
+    gen_vec = downsample_darkness(generated, out_w=24, out_h=24)
+
+    keys = ["ink_ratio", "bbox_aspect", "bbox_area_ratio", "center_x", "center_y"]
+    floors = {
+        "ink_ratio": 0.008,
+        "bbox_aspect": 0.25,
+        "bbox_area_ratio": 4.0,
+        "center_x": 0.06,
+        "center_y": 0.06,
+    }
+
+    mu_sd = {}
+    for k in keys:
+        mu, sd = mean_std(v[1][k] for v in examples_index)
+        mu_sd[k] = (mu, max(sd, floors[k]))
+
+    z_sum = 0.0
+    details = {}
+    for k in keys:
+        mu, sd = mu_sd[k]
+        z = abs((gen_feat[k] - mu) / sd)
+        details[k] = {"generated": gen_feat[k], "mean": mu, "std": sd, "z": z}
+        z_sum += z
+    feature_score = max(0.0, 100.0 - (z_sum / len(keys)) * 20.0)
+
+    best_cos = -1.0
+    best_path = ""
+    for p, _, vec in examples_index:
+        c = cosine_similarity(gen_vec, vec)
+        if c > best_cos:
+            best_cos = c
+            best_path = p
+    shape_score = max(0.0, min(100.0, best_cos * 100.0))
+
+    total = 0.45 * feature_score + 0.55 * shape_score
+    details["shape_cosine"] = {"generated": best_cos, "mean": 0.0, "std": 1.0, "z": 0.0}
+    details["feature_score"] = {"generated": feature_score, "mean": 0.0, "std": 1.0, "z": 0.0}
+    details["shape_score"] = {"generated": shape_score, "mean": 0.0, "std": 1.0, "z": 0.0}
+    return total, best_path, details
+
+
+def find_best_signature(cfg: SignatureConfig) -> Tuple[SignatureConfig, List[List[int]], float, str, List[dict]]:
+    """Делаем N проверок и выбираем вариант с максимальной схожестью."""
+    examples_index = build_examples_index(cfg.examples_dir, cfg.examples_limit)
+    if not examples_index:
+        raise RuntimeError(f"Не найдены валидные PNG в {cfg.examples_dir}")
+
+    trials: List[dict] = []
+    best_img: List[List[int]] = []
+    best_cfg = cfg
+    best_score = -1.0
+    best_path = ""
+
+    for i in range(cfg.trials_count):
+        variant = SignatureConfig(**cfg.__dict__)
+        variant.seed = cfg.seed + i * 17
+        variant.stroke_width = max(2, min(6, cfg.stroke_width + ((i % 5) - 2)))
+
+        generated = generate_signature_image(variant)
+        score, matched_path, details = similarity_to_index(generated, examples_index)
+
+        trials.append({
+            "trial": i + 1,
+            "seed": variant.seed,
+            "stroke_width": variant.stroke_width,
+            "score": score,
+            "matched": os.path.basename(matched_path) if matched_path else "-",
+            "details": details,
+        })
+
+        if score > best_score:
+            best_score = score
+            best_img = generated
+            best_cfg = variant
+            best_path = matched_path
+
+    return best_cfg, best_img, best_score, best_path, trials
+
+
 def main() -> None:
     # ----- CONFIG: меняйте здесь -----
     cfg = SignatureConfig(
         first_name="Алексей",
         last_name="Ковалев",
         middle_name="Игоревич",
         use_full_last_name=True,
         seed=73,
         stroke_width=3,
+        width=256,
+        height=256,
         output_path="generated/signature_kovalev.png",
         run_comparison=True,
+        trials_count=20,
+        examples_limit=20,
     )
     # -------------------------------
 
-    generated = generate_signature_image(cfg)
+    best_cfg, generated, score, matched, trials = find_best_signature(cfg)
     save_grayscale_png(cfg.output_path, generated)
     print(f"Сохранено: {cfg.output_path}")
 
     if cfg.run_comparison:
-        score, details = estimate_human_likeness(generated)
-        print(f"Оценка человекоподобности и сходства: {score:.1f}/100")
-        for k, v in details.items():
-            print(f"  {k}: gen={v['generated']:.4f}, mean={v['mean']:.4f}, std={v['std']:.4f}, z={v['z']:.2f}")
+        print(f"Проверок выполнено: {len(trials)}")
+        for t in trials:
+            print(
+                f"  #{t['trial']:02d}: score={t['score']:.2f}, "
+                f"seed={t['seed']}, stroke={t['stroke_width']}, match={t['matched']}"
+            )
+
+        print("\nЛучший вариант:")
+        print(f"  score={score:.2f}/100")
+        print(f"  seed={best_cfg.seed}")
+        print(f"  stroke_width={best_cfg.stroke_width}")
+        if matched:
+            print(f"  ближайший пример: {matched}")
 
 
 if __name__ == "__main__":
     main()
 
EOF
)

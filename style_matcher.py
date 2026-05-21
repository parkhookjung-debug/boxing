"""
Match a pose CSV or existing profile against boxing style archetypes.

Run:
    python style_matcher.py --target LIM_full_data8.csv
    python style_matcher.py --target-pattern "LIM_full_data*.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

from style_profile import FEATURE_ORDER, extract_features, read_csv_rows


def load_profiles(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vector(features: dict[str, float], order: list[str]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in order]


def profile_stats(profiles: dict, order: list[str]) -> tuple[list[float], list[float]]:
    rows = [vector(profile["features"], order) for profile in profiles.values()]
    means, stds = [], []
    for col in zip(*rows):
        m = sum(col) / len(col)
        var = sum((x - m) ** 2 for x in col) / max(1, len(col))
        means.append(m)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def z_vector(values: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(values, means, stds)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(y * y for y in b)) + 1e-9
    return dot / (na * nb)


def score_to_percent(score: float) -> float:
    return max(0.0, min(100.0, (score + 1.0) * 50.0))


def load_target_features(base_dir: str, target: str | None, target_pattern: str | None) -> dict[str, float]:
    paths: list[str] = []
    if target:
        path = target if os.path.isabs(target) else os.path.join(base_dir, target)
        paths = [path]
    elif target_pattern:
        paths = sorted(glob.glob(os.path.join(base_dir, target_pattern)))
    else:
        raise SystemExit("Provide --target or --target-pattern.")

    frames = []
    for path in paths:
        frames.extend(read_csv_rows(path))
    if not frames:
        raise SystemExit("No target frames found.")
    return extract_features(frames)


def match_styles(profile_data: dict, target_features: dict[str, float]) -> list[dict]:
    order = profile_data["feature_order"]
    profiles = profile_data["profiles"]
    means, stds = profile_stats(profiles, order)
    target_z = z_vector(vector(target_features, order), means, stds)

    results = []
    for style_id, profile in profiles.items():
        style_z = z_vector(vector(profile["features"], order), means, stds)
        raw = cosine(target_z, style_z)
        results.append(
            {
                "style_id": style_id,
                "display_name": profile["display_name"],
                "source_fighter": profile["source_fighter"],
                "similarity": round(score_to_percent(raw), 2),
                "description": profile["description"],
                "training_focus": profile["training_focus"],
            }
        )
    return sorted(results, key=lambda item: item["similarity"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--profiles", default="style_profiles.json")
    parser.add_argument("--target")
    parser.add_argument("--target-pattern")
    parser.add_argument("--out")
    args = parser.parse_args()

    profiles_path = args.profiles if os.path.isabs(args.profiles) else os.path.join(args.base_dir, args.profiles)
    profile_data = load_profiles(profiles_path)
    target_features = load_target_features(args.base_dir, args.target, args.target_pattern)
    results = match_styles(profile_data, target_features)

    output = {
        "target": args.target or args.target_pattern,
        "matches": results,
        "target_features": target_features,
    }

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(args.base_dir, args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")

    print("Style match:")
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item['display_name']} ({item['similarity']:.2f}%) - source: {item['source_fighter']}")
    print("\nPrimary recommendation:")
    top = results[0]
    print(f"{top['display_name']}: {top['description']}")
    print("Training focus:")
    for focus in top["training_focus"]:
        print(f"- {focus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

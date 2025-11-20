from __future__ import annotations
import argparse
from use_cases.generate_village import GenerateVillage, GenerateVillageParams
from infrastructure.exporter_svg import village_to_svg
from infrastructure.exporter_json import village_to_json

def main():
    ap = argparse.ArgumentParser(description="Generate a rural village map (Watabou-inspired).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--tags", type=str, default="")
    ap.add_argument("--out", type=str, default="village.svg")
    ap.add_argument("--format", type=str, choices=["svg","json"], default="svg")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    params = GenerateVillageParams(seed=args.seed, width=args.width, height=args.height, tags=tags)
    v = GenerateVillage(params).execute()

    if args.format == "svg":
        out = village_to_svg(v)
    else:
        out = village_to_json(v)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds an rt_netdb_expanded.json from:
  - ModuleManager.ConfigCache  (for CelestialBody radii / gravParameter if needed)
  - kerbin_distance_ranges...csv (for "4-Sat Adjacent Distance (m)")

Rules:
  NET3  = radius + (adjacent_distance_m)
  NET4..NET9 = NET3 * multipliers[net]  (defaults below; editable)
  NET1..NET2: left untouched (emit only if you want; defaults to omit unless a body already has them)
  All equatorial & circular by default (INC=0, ECC=0, LAN=0, LPE=0).

Output schema:
{
  "bodies": {
    "<BodyName>": {
      "net_presets": {
        "3": {"sma_m": <float>, "inc_deg": 0.0, "ecc": 0.0, "raan_deg_base": 0.0, "argp_deg": 0.0},
        "4": {...}, ... "9": {...}
      }
    }, ...
  }
}
"""

import os, re, json, argparse
import pandas as pd
from collections import defaultdict

def read_text(p):
    return open(p, "r", errors="ignore").read()

def find_blocks(txt, token):
    spans=[]
    for m in re.finditer(rf'(^|\n)([ \t]*){re.escape(token)}\s*\{{', txt):
        start=m.start(0)+(0 if m.group(1)=="" else 1)
        i=txt.find("{", m.end(0)-1)+1
        d=1; n=len(txt)
        while i<n and d>0:
            if txt[i]=="{": d+=1
            elif txt[i]=="}": d-=1
            i+=1
        spans.append((start,i,txt[start:i], m.group(2)))
    return spans

def grab(txt, key):
    m=re.search(rf'(^|\n)\s*{re.escape(key)}\s*=\s*([^\r\n#]+)', txt)
    return m.group(2).strip() if m else None

def norm_body(s):
    s = str(s).strip()
    # Normalize Sun to JNSQ display label if you use that naming; else leave as-is
    if s.lower() in ("sun","sol","kerbol"): return "Kerbol (the Sun)"
    return s

def main(args):
    mm_txt = read_text(args.mm)
    csv_path = args.csv

    # 1) Celestial radii from MM cache
    body_radius = {}
    for (s,e,blk,_) in find_blocks(mm_txt, "Body"):
        nm = grab(blk, "name") or grab(blk,"Name")
        if not nm: continue
        props = find_blocks(blk, "Properties")
        radius = None
        if props:
            r = grab(props[0][2], "radius")
            if r:
                try: radius = float(r)
                except: pass
        if radius:
            body_radius[nm] = radius

    # 2) Distances CSV: "Name of body" + "4-Sat Adjacent Distance (m)"
    df = pd.read_csv(csv_path)
    body_col = next(c for c in df.columns if c.strip().lower()=="name of body")
    adj_col  = next(c for c in df.columns if "4-Sat Adjacent Distance" in c)
    adj = {}
    for _, row in df.iterrows():
        b = norm_body(row[body_col])
        try:
            adj[b] = float(str(row[adj_col]).replace(",",""))
        except:
            pass

    # 3) Build netdb dict
    multipliers = {
        4: 1.06,
        5: 1.12,
        6: 1.18,
        7: 1.24,
        8: 1.30,
        9: 1.36,
    }
    def preset_entry(sma):
        return {
            "sma_m": float(sma),
            "inc_deg": 0.0,
            "ecc": 0.0,
            "raan_deg_base": 0.0,
            "argp_deg": 0.0
        }

    out = {"bodies": {}}
    # Combine keys from radii & adjacency so we don't miss mod-added bodies
    all_bodies = sorted(set(body_radius.keys()) | set(adj.keys()), key=lambda x: x.lower())

    for b in all_bodies:
        R = body_radius.get(b)
        A = adj.get(b)
        # Only produce a lane if we have both pieces (radius for SMA, and adjacency for NET3 altitude)
        if R is None or A is None:
            continue
        net_presets = {}
        sma3 = R + A
        net_presets["3"] = preset_entry(sma3)
        # Expand outward lanes
        for n,mul in multipliers.items():
            net_presets[str(n)] = preset_entry(sma3 * mul)
        out["bodies"][b] = {"net_presets": net_presets}

    # 4) Optionally: carry through existing body entries from an older netdb (merge mode)
    if args.merge and os.path.exists(args.merge):
        try:
            prev = json.load(open(args.merge, "r"))
            pbodies = prev.get("bodies", {})
            for b, node in pbodies.items():
                if b not in out["bodies"]:
                    out["bodies"][b] = node
                else:
                    # merge: keep new 3..9, preserve previous 1..2 if they existed
                    prev_np = node.get("net_presets", {})
                    cur_np  = out["bodies"][b].setdefault("net_presets", {})
                    for k,v in prev_np.items():
                        if k not in cur_np:
                            cur_np[k] = v
        except Exception as e:
            print("Merge warning:", e)

    # 5) Write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", args.out)
    print("Bodies with NET3..9:", len(out["bodies"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm",   required=True, help="ModuleManager.ConfigCache")
    ap.add_argument("--csv",  required=True, help="kerbin_distance_ranges*csv with the '4-Sat Adjacent Distance (m)' column")
    ap.add_argument("--out",  required=True, help="Path to write rt_netdb_expanded.json")
    ap.add_argument("--merge", required=False, help="(Optional) existing rt_netdb.json to merge (preserves NET1..2 if present)")
    args = ap.parse_args()
    main(args)

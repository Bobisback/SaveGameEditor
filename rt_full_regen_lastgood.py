#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate full constellation set using the "last good" rules:

- Use rt_netdb for SMA (NET presets). If a NET (e.g., Kerbin NET9) is missing,
  synthesize an equatorial lane by extrapolating from NET3..NET8 spacing.
- Equatorial lanes: NET1..NET9 (inclination forced to 0 for NET<=9).
- Kerbin hubs: G1..G7 -> NET3..NET9 (4 sats per lane).
- Remote constellations: every body in each group (from the Groups CSV), NET3, 4 sats each.
  IMPORTANT: Kerbin IS included in the remote pass (duplicates allowed) to match the last-good run.
- NO Dak×5 safety (use netdb NET3 SMA as-is).
- Strip IDENT everywhere (never write IDENT).
- Preserve dish targets (no edits). Only ensure transmitters enabled and deployables extended.
- Fresh vessel pid / part uid & persistentId. Set ref to root PART uid.
- Seed matching: Kerbin seeds = Commsat-Kerbin-Gx; Remote seeds = Commsat-Other-Gx.

Outputs:
  - quicksave #11.sfs (full, IDENT-free)
  - quicksave-11_full.sfs (alias without spaces)
  - quicksave-11_full.zip (contains "quicksave #11.sfs")
  - placements_quicksave11_full.csv
  - parts_audit_quicksave11_full.csv
  - connectivity_audit_quicksave11.csv
  - connectivity_suspicions_quicksave11.csv
"""

import os, re, json, math, uuid, random, csv, zipfile, argparse
from collections import defaultdict, Counter
import pandas as pd

# ----------------------------- utilities -----------------------------

def read_text(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def write_text(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def find_blocks(txt, token):
    spans=[]
    for m in re.finditer(rf'(^|\n)([ \t]*){re.escape(token)}\s*\{{', txt):
        start = m.start(0) + (0 if m.group(1)=="" else 1)
        i = txt.find("{", m.end(0)-1) + 1
        depth = 1; n=len(txt)
        while i<n and depth>0:
            if txt[i]=="{": depth += 1
            elif txt[i]=="}": depth -= 1
            i += 1
        spans.append((start, i, txt[start:i], m.group(2)))
    return spans

def grab(txt, key):
    m = re.search(rf'(^|\n)\s*{re.escape(key)}\s*=\s*([^\r\n#]+)', txt)
    return m.group(2).strip() if m else None

def vname(vtxt):
    m = re.search(r'(^|\n)\s*name\s*=\s*([^\r\n#]+)', vtxt)
    return m.group(2).strip() if m else None

def strip_ident_in_orbit(vtxt):
    """ remove any IDENT lines from every ORBIT block in this vessel """
    out=[]; last=0
    for (os_, oe_, ob_txt, _) in find_blocks(vtxt, "ORBIT"):
        out.append(vtxt[last:os_])
        # remove IDENT lines
        ob_lines = [ln for ln in ob_txt.splitlines() if not re.search(r'(^|\s)IDENT\s*=', ln)]
        out.append("\n".join(ob_lines))
        last = oe_
    out.append(vtxt[last:])
    return "".join(out)

def first_orbit_span(vtxt):
    for m in re.finditer(r'(^|\n)([ \t]*)ORBIT\s*\{', vtxt):
        os_ = m.start(0)+(0 if m.group(1)=="" else 1); oindent=m.group(2)
        j = vtxt.find("{", m.end(0)-1)+1
        d=1; n=len(vtxt)
        while j<n and d>0:
            if vtxt[j]=="{": d+=1
            elif vtxt[j]=="}": d-=1
            j+=1
        return os_, j, oindent
    return None, None, ""

def fmt_orbit(indent, fields, force_eq_inc=True):
    inner = indent + ("\t" if ("\t" in indent or indent=="") else "    ")
    lines = [indent+"ORBIT", indent+"{"]
    order = ["SMA","ECC","INC","LAN","LPE","MNA","EPH","REF"]
    for k in order:
        v = fields.get(k)
        if v is None:
            continue
        if k=="INC" and force_eq_inc:
            v = 0.0
        lines.append(f"{inner}{k} = {v:.6f}" if isinstance(v,float) else f"{inner}{k} = {v}")
    lines.append(indent+"}")
    return "\n".join(lines)

def top_name_span(vtxt):
    parts = find_blocks(vtxt, "PART")
    limit = parts[0][0] if parts else len(vtxt)
    m = re.search(r'(^|\n)(?P<i>\s*)name\s*=\s*([^\r\n#]+)', vtxt[:limit])
    if not m: return None, None, ""
    return m.start(0)+(0 if m.group(1)=="" else 1), m.end(0), m.group('i')

def set_vessel_name(vtxt, newname):
    s,e,i = top_name_span(vtxt)
    if s is None:
        raise RuntimeError("Top-level 'name =' not found in VESSEL block")
    return vtxt[:s] + f"{i}name = {newname}" + vtxt[e:]

def set_state(nv):
    for k,v in [("sit","ORBITING"),("landed","False"),("splashed","False"),("prst","False")]:
        m=re.search(r'(^|\n)(?P<i>\s*){k}\s*=\s*[^\r\n#]+'.format(k=re.escape(k)), nv)
        if m:
            s=m.start(0)+(0 if m.group(1)=="" else 1); e=m.end(0); i=m.group('i')
            nv = nv[:s] + f"{i}{k} = {v}" + nv[e:]
    return nv

def rewrite_ids_and_ref(nv):
    # vessel pid
    nv = re.sub(r'(^|\n)(\s*)pid\s*=\s*[^\r\n#]+',
                lambda m: f"{m.group(1)}{m.group(2)}pid = {str(uuid.uuid4())}",
                nv, count=1)
    # ref -> root uid
    parts = find_blocks(nv, "PART")
    if parts:
        root_m = re.search(r'(^|\n)\s*root\s*=\s*(\d+)', nv)
        root_idx = int(root_m.group(1)) if root_m else 0
        root_idx = max(0, min(root_idx, len(parts)-1))
        uid_m = re.search(r'(^|\n)\s*uid\s*=\s*([^\r\n#]+)', parts[root_idx][2])
        if uid_m:
            if re.search(r'(^|\n)\s*ref\s*=\s*', nv):
                nv = re.sub(r'(^|\n)(\s*)ref\s*=\s*[^\r\n#]+',
                            lambda m: f"{m.group(1)}{m.group(2)}ref = {uid_m.group(2)}",
                            nv, count=1)
            else:
                rloc = re.search(r'(^|\n)(\s*)root\s*=\s*[^\r\n#]+', nv)
                ins = rloc.end(0) if rloc else 0
                nv = nv[:ins] + f"\n\t\t\tref = {uid_m.group(2)}\n" + nv[ins:]
    # part ids
    out=[]; cur=0
    for (ps,pe,ptxt,_) in parts:
        ptxt = re.sub(r'(^|\n)(\s*)uid\s*=\s*[^\r\n#]+',
                      lambda m: f"{m.group(1)}{m.group(2)}uid = {random.randint(100000000, 2147483647)}",
                      ptxt, count=1)
        ptxt = re.sub(r'(^|\n)(\s*)persistentId\s*=\s*[^\r\n#]+',
                      lambda m: f"{m.group(1)}{m.group(2)}persistentId = {random.randint(10**12, 10**13)}",
                      ptxt, count=1)
        out.append((ps,pe,ptxt))
    out.sort(key=lambda x:x[0])
    sp=[]; cur=0
    for ps,pe,pt in out:
        sp.append(nv[cur:ps]); sp.append(pt); cur=pe
    sp.append(nv[cur:])
    return "".join(sp)

def enable_antennas(nv):
    # leave targets alone; just flip flags if present
    out=[]; cur=0
    for (ps,pe,ptxt,_) in find_blocks(nv, "PART"):
        mods = find_blocks(ptxt, "MODULE")
        if mods:
            mods.sort(key=lambda x:x[0])
            outm=[]; c2=0; changed=False
            for ms,me,mtxt,_ in mods:
                mname = grab(mtxt, "name")
                if mname == "ModuleDataTransmitter":
                    if re.search(r'(^|\n)\s*IsEnabled\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)IsEnabled\s*=\s*[^\r\n#]+',
                                      lambda m: f"{m.group(1)}{m.group(2)}IsEnabled = True",
                                      mtxt)
                        changed=True
                    if re.search(r'(^|\n)\s*enabled\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)enabled\s*=\s*[^\r\n#]+',
                                      lambda m: f"{m.group(1)}{m.group(2)}enabled = True",
                                      mtxt)
                        changed=True
                if mname == "ModuleDeployableAntenna":
                    if re.search(r'(^|\n)\s*isDeployed\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)isDeployed\s*=\s*[^\r\n#]+',
                                      lambda m: f"{m.group(1)}{m.group(2)}isDeployed = True",
                                      mtxt)
                        changed=True
                    if re.search(r'(^|\n)\s*deployState\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)deployState\s*=\s*[^\r\n#]+',
                                      lambda m: f"{m.group(1)}{m.group(2)}deployState = EXTENDED",
                                      mtxt)
                        changed=True
                outm.append((ms,me,mtxt))
            if changed:
                sp2=[]; c3=0
                for ms,me,mt in outm:
                    sp2.append(ptxt[c2:ms]); sp2.append(mt); c2=me
                sp2.append(ptxt[c2:])
                ptxt = "".join(sp2)
        out.append(nv[cur:ps]); out.append(ptxt); cur=pe
    out.append(nv[cur:])
    return "".join(out)

# ----------------------------- main logic -----------------------------

def run(baseline_sfs, mm_cache, fgi_path, groups_csv, netdb_json, out_dir):
    base_txt = read_text(baseline_sfs)
    mm_txt   = read_text(mm_cache)
    netdb    = json.load(open(netdb_json, "r"))

    # FGI maps
    name_to_idx={}
    for line in open(fgi_path,"r",errors="ignore"):
        m=re.match(r"(\d+)\s+(.+)", line.strip())
        if m: name_to_idx[m.group(2).strip()] = int(m.group(1))

    # Groups (raw; include Kerbin in remotes to match last-good)
    dfg=pd.read_csv(groups_csv)
    body_col = next(c for c in dfg.columns if c.strip().lower()=="name of body")
    group_col= next(c for c in dfg.columns if c.strip().lower()=="group")

    def norm_body(s):
        s=str(s).strip()
        if s.lower()=="sun": return "Kerbol (the Sun)"
        return s

    groups_map=defaultdict(list)
    for _,row in dfg.iterrows():
        b=norm_body(row[body_col]); m=re.search(r'(\d+)', str(row[group_col]))
        if b and m: groups_map[int(m.group(1))].append(b)

    # NetDB helpers
    netdb_bodies = {k.lower():k for k in netdb.get("bodies",{}).keys()}
    def bkey(name): return netdb_bodies.get(name.lower())

    # Synthesize an equatorial SMA one lane outward if needed
    def synthesize_equatorial_sma(name):
        bk=bkey(name)
        if not bk: return None
        smas=[]
        presets=netdb["bodies"][bk].get("net_presets",{})
        # use lanes 3..8 for spacing
        for n in range(3,9):
            p=presets.get(str(n))
            if p and p.get("sma_m"):
                try: smas.append(float(p["sma_m"]))
                except: pass
        if not smas: return None
        smas.sort()
        if len(smas)>=2:
            steps=[smas[i+1]-smas[i] for i in range(len(smas)-1)]
            return smas[-1] + sum(steps)/len(steps)
        return smas[-1]*1.08

    def fields_from_net(body, net_id, slot, UT):
        bk=bkey(body)
        if not bk:
            raise AssertionError(f"{body} not in netdb")
        p = netdb["bodies"][bk]["net_presets"].get(str(net_id))
        if not (p and p.get("sma_m")):
            # synthesize Kerbin (or any) NET if missing
            sma = synthesize_equatorial_sma(body)
            if sma is None:
                raise AssertionError(f"{body} NET{net_id} SMA missing and cannot synthesize")
            ecc=0.0; inc=0.0; lan=0.0; lpe=0.0
        else:
            sma=float(p["sma_m"]); ecc=float(p.get("ecc",0.0))
            lan=float(p.get("raan_deg_base",0.0)); lpe=float(p.get("argp_deg",0.0))
            inc=float(p.get("inc_deg",0.0))
        # force equatorial for NET<=9
        if net_id <= 9: inc = 0.0
        phase={1:0,2:90,3:180,4:270}[slot]
        ref=name_to_idx.get(body, name_to_idx.get("Kerbin",1))
        return {"SMA":sma,"ECC":ecc,"INC":inc,"LAN":lan,"LPE":lpe,"MNA":math.radians(phase),"EPH":UT,"REF":ref}

    # Extract seeds; drop any generated constellation vessels
    ves=find_blocks(base_txt,"VESSEL")
    seed_kerbin={}; seed_other={}; keep=[]
    pat_const=re.compile(r"^[^-]+-G\d+-SAT\d+-NET\d+$")
    for (s,e,vtxt,_) in ves:
        nm=vname(vtxt) or ""
        vclean=strip_ident_in_orbit(vtxt)
        if re.fullmatch(r"Commsat-Kerbin-G([1-7])", nm):
            seed_kerbin[int(re.findall(r'\d+', nm)[0])] = vclean; keep.append(vclean)
        elif re.fullmatch(r"Commsat-Other-G([1-7])", nm):
            seed_other[int(re.findall(r'\d+', nm)[0])] = vclean; keep.append(vclean)
        elif pat_const.match(nm):
            continue
        else:
            keep.append(vclean)

    missingK=[g for g in range(1,8) if g not in seed_kerbin]
    missingO=[g for g in range(1,8) if g not in seed_other]
    if missingK or missingO:
        raise AssertionError(f"Missing seeds: Kerbin{missingK} Other{missingO}")

    m_ut=re.search(r'\bUT\s*=\s*([0-9eE\.\-]+)', base_txt)
    UT=float(m_ut.group(1)) if m_ut else 0.0

    # Build all vessels (hubs + remotes)
    placements=[]; parts_rows=[]; new_vessels=[]
    G_to_NET={1:3,2:4,3:5,4:6,5:7,6:8,7:9}

    def build_from(seed_vtxt, newname, fields):
        nv=set_vessel_name(seed_vtxt, newname)
        nv=set_state(nv)
        os_,oe_,oindent=first_orbit_span(nv)
        nv=nv[:os_]+fmt_orbit(oindent, fields, force_eq_inc=True)+nv[oe_:]
        nv=rewrite_ids_and_ref(nv)
        nv=enable_antennas(nv)
        # part audit
        sp=set([grab(pt,"name") for (ps,pe,pt,_) in find_blocks(seed_vtxt,"PART") if grab(pt,"name")])
        gp=set([grab(pt,"name") for (ps,pe,pt,_) in find_blocks(nv,"PART") if grab(pt,"name")])
        parts_rows.append({"vessel":newname,"match":sp==gp,"diff_added":"; ".join(sorted(gp-sp)),"diff_missing":"; ".join(sorted(sp-gp))})
        return nv

    # Hubs
    for g in range(1,8):
        sv=seed_kerbin[g]
        for slot in (1,2,3,4):
            nm=f"Kerbin-G{g}-SAT{slot}-NET{G_to_NET[g]}"
            fields=fields_from_net("Kerbin", G_to_NET[g], slot, UT)
            new_vessels.append(build_from(sv, nm, fields))
            placements.append({"name":nm,"body":"Kerbin","group":f"G{g}","net":G_to_NET[g],"slot":slot})

    # Remotes (including Kerbin per last-good)
    for g in range(1,8):
        sv=seed_other[g]
        for body in sorted(groups_map.get(g, [])):
            for slot in (1,2,3,4):
                nm=f"{body}-G{g}-SAT{slot}-NET3"
                fields=fields_from_net(body, 3, slot, UT)
                new_vessels.append(build_from(sv, nm, fields))
                placements.append({"name":nm,"body":body,"group":f"G{g}","net":3,"slot":slot})

    # Rebuild FLIGHTSTATE
    fsm=re.search(r'(^|\n)([ \t]*)(FLIGHTSTATE|FlightState|flightstate)\s*\{', base_txt)
    j=base_txt.find("{", fsm.end(0)-1)+1; d=1; n=len(base_txt)
    while j<n and d>0:
        if base_txt[j]=="{": d+=1
        elif base_txt[j]=="}": d-=1
        j+=1
    fs_start=fsm.start(0)+(0 if fsm.group(1)=="" else 1); fs_end=j
    flight=base_txt[fs_start:fs_end]

    # Strip all VESSEL nodes and stitch seeds/others + new
    pieces=[]; cur=0
    for (s,e,vtxt,_) in find_blocks(flight,"VESSEL"):
        pieces.append(flight[cur:s]); cur=e
    pieces.append(flight[cur:])
    flight_no_vessels="".join(pieces)

    new_blob="\n" + "\n".join(keep) + "\n" + "\n".join(new_vessels) + "\n"
    new_flight = flight_no_vessels.replace("{","{\n",1).rstrip("}\n") + new_blob + "}"

    new_text = base_txt[:fs_start] + new_flight + base_txt[fs_end:]

    # Write outputs
    sfs_out = os.path.join(out_dir, "quicksave #11.sfs")
    write_text(sfs_out, new_text)
    # alias without spaces + a zip containing the spaced filename
    alias = os.path.join(out_dir, "quicksave-11_full.sfs")
    import shutil
    shutil.copy2(sfs_out, alias)
    zip_path = os.path.join(out_dir, "quicksave-11_full.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(sfs_out, arcname="quicksave #11.sfs")

    # CSVs
    pd.DataFrame(placements).to_csv(os.path.join(out_dir, "placements_quicksave11_full.csv"), index=False)

    # basic connectivity audit (alt vs 4-sat threshold, tx/deploy flags)
    # you can skip this if you want speed: comment out below
    # (kept minimal & fast)
    dfd = pd.read_csv(groups_csv)
    body_col = next(c for c in dfd.columns if c.strip().lower()=="name of body")
    adj_col  = next(c for c in dfd.columns if "4-Sat Adjacent Distance" in c)
    min_adj={}
    for _,row in dfd.iterrows():
        b = str(row[body_col]).strip()
        try: min_adj[b] = float(str(row[adj_col]).replace(",",""))
        except: pass

    def radius(b):
        # fast read from mm cache (already in memory)
        for (s,e,blk,_) in find_blocks(mm_txt,"CelestialBody"):
            nm=grab(blk,"name") or grab(blk,"Name")
            if nm and nm.lower()==b.lower():
                props=find_blocks(blk,"Properties")
                if props:
                    r=grab(props[0][2],"radius")
                    if r: return float(r)
        return 0.0

    def first_orbit(vtxt):
        for m in re.finditer(r'(^|\n)([ \t]*)ORBIT\s*\{', vtxt):
            os_=m.start(0)+(0 if m.group(1)=="" else 1)
            j=vtxt.find("{", m.end(0)-1)+1; d=1; n=len(vtxt)
            while j<n and d>0:
                if vtxt[j]=="{": d+=1
                elif vtxt[j]=="}": d-=1
                j+=1
            return vtxt[os_:j]
        return None

    def ant_counts(vtxt):
        total_tx=enabled_tx=0; total_dep=deployed=0
        for (ps,pe,ptxt,_) in find_blocks(vtxt,"PART"):
            for (ms,me,mtxt,_) in find_blocks(ptxt,"MODULE"):
                mname=grab(mtxt,"name")
                if mname=="ModuleDataTransmitter":
                    total_tx += 1
                    en = grab(mtxt,"enabled") or grab(mtxt,"IsEnabled")
                    if en and en.strip().lower()=="true": enabled_tx += 1
                if mname=="ModuleDeployableAntenna":
                    total_dep += 1
                    dp = grab(mtxt,"isDeployed")
                    if dp and dp.strip().lower()=="true": deployed += 1
        return total_tx, enabled_tx, total_dep, deployed

    rows=[]
    pat=re.compile(r'^(?P<b>[^-]+)-G(?P<g>\d+)-SAT(?P<s>\d+)-NET(?P<n>\d+)$')
    ves_new=find_blocks(new_text,"VESSEL")
    for (s,e,vtxt,_) in ves_new:
        nm=vname(vtxt) or ""
        m=pat.match(nm)
        if not m: continue
        ob=first_orbit(vtxt)
        sma=float(grab(ob,"SMA") or 0.0) if ob else 0.0
        b=m.group("b")
        alt = sma - radius(b) if sma>0 else None
        req = min_adj.get(b, None)
        los_ok = (alt is not None and req is not None and alt + 1e-6 >= req)
        tx_total, tx_on, dep_total, dep_on = ant_counts(vtxt)
        rows.append({"vessel":nm,"body":b,"group":int(m.group("g")),"net":int(m.group("n")),"slot":int(m.group("s")),
                     "altitude_m":alt,"min_adjacent_required_m":req,"LoS_ok":los_ok,
                     "tx_total":tx_total,"tx_enabled":tx_on,"deploy_total":dep_total,"deploy_open":dep_on})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "connectivity_audit_quicksave11.csv"), index=False)
    pd.DataFrame([r for r in rows if (r["LoS_ok"]==False or r["tx_enabled"]<r["tx_total"] or r["deploy_open"]<r["deploy_total"])]) \
      .to_csv(os.path.join(out_dir, "connectivity_suspicions_quicksave11.csv"), index=False)

    print("DONE")
    print("Save:", sfs_out)
    print("Alias:", alias)
    print("Zip:", zip_path)

# ----------------------------- cli -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="baseline", required=True, help='Input save, e.g. "quicksave-11.sfs"')
    ap.add_argument("--mm", dest="mmcache", required=True, help='ModuleManager.ConfigCache')
    ap.add_argument("--fgi", dest="fgi", required=True, help='JNSQ Body Indexes.txt')
    ap.add_argument("--groups", dest="groups", required=True, help='Groups CSV (Name of body, Group, 4-Sat Adjacent Distance)')
    ap.add_argument("--netdb", dest="netdb", required=True, help='rt_netdb_expanded.json or rt_netdb.json')
    ap.add_argument("--outdir", dest="outdir", default="./out", help="Where to write outputs")
    args = ap.parse_args()

    run(args.baseline, args.mmcache, args.fgi, args.groups, args.netdb, args.outdir)

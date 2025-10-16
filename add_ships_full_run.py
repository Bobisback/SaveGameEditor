# Update group mapping from the new CSV, then regenerate constellations from quicksave #11.sfs
# with NET1–NET9 equatorial lanes (G1..G7 -> NET3..NET9) and REM on NET3.
# Dak safety x5. No dish retargets. Strict renamer & validations.

import os, re, json, math, uuid, random, csv, zipfile, time
import pandas as pd
from collections import defaultdict


# --- Settings loader (TOML) -----------------------------------------------
import argparse
from dataclasses import dataclass, field
from typing import List, Union, Dict

# Python 3.11+ has tomllib; fallback to tomli if needed
try:
    import tomllib  # py311+
except Exception:
    tomllib = None
try:
    import tomli  # fallback
except Exception:
    tomli = None

def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        if tomllib:
            return tomllib.load(f)
        if tomli:
            return tomli.load(f)
        raise RuntimeError("No TOML loader available (need Python 3.11+ or install tomli)")

@dataclass
class ConfigDeploy:
    groups: Union[str, List[Union[str,int]]] = "all"   # "all" or e.g. ["Group 1","G12",15]

@dataclass
class Config:
    sfs_in: str
    mmcache: str
    fgi_path: str
    groups_csv: str
    netdb_expanded: str
    out_dir: str
    groups_index_json: str
    groups_diff_csv: str
    groups_curr_csv: str
    placements_csv: str
    zip_name: str
    deploy: ConfigDeploy = field(default_factory=ConfigDeploy)

    @staticmethod
    def with_defaults(d: dict) -> "Config":
        inp = d.get("inputs", {})
        out = d.get("outputs", {})
        data = d.get("data", {})
        out_files = out.get("files", {})
        deploy_cfg = d.get("deploy", {})
        return Config(
            sfs_in          = inp.get("sfs_in",   "./inputs/quicksave #11.sfs"),
            mmcache         = inp.get("mmcache",  "./inputs/ModuleManager.ConfigCache"),
            fgi_path        = inp.get("fgi_path", "./inputs/JNSQ Body Indexes.txt"),
            groups_csv      = inp.get("groups_csv", "./inputs/kerbin_distance_ranges.csv"),
            netdb_expanded  = inp.get("netdb_expanded", "./outputs/rt_netdb_expanded.json"),
            out_dir         = out.get("out_dir", "./outputs"),
            groups_index_json = data.get("groups_index_json","./data/rt_groups_index.json"),
            groups_diff_csv = out_files.get("groups_diff_csv","group_map_diff.csv"),
            groups_curr_csv = out_files.get("groups_curr_csv","group_map_current.csv"),
            placements_csv  = out_files.get("placements_csv","placements_quicksave11.csv"),
            zip_name        = out_files.get("zip_name","quicksave #11.zip"),
            deploy = ConfigDeploy(
                groups = deploy_cfg.get("groups", "all"),
            ),
        )

def load_config() -> Config:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", help="Path to TOML settings file")
    args, _ = p.parse_known_args()
    cfg_path = args.config or os.environ.get("ADDSHIPS_CONFIG") or "config/add_ships.toml"
    if os.path.exists(cfg_path):
        d = _load_toml(cfg_path)
        return Config.with_defaults(d)
    return Config.with_defaults({})
# --------------------------------------------------------------------------
# ----------------------- Paths (from config) -----------------------
cfg = load_config()
SFS_IN   = cfg.sfs_in
MMCACHE  = cfg.mmcache
FGI_PATH = cfg.fgi_path
GROUPS_CSV = cfg.groups_csv
NETDB_EXP = cfg.netdb_expanded

OUT_DIR = cfg.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS_JSON     = cfg.groups_index_json
GROUPS_DIFF_CSV = os.path.join(OUT_DIR, cfg.groups_diff_csv)
GROUPS_CURR_CSV = os.path.join(OUT_DIR, cfg.groups_curr_csv)

# ----------------------- Guards -----------------------
for p in [SFS_IN, MMCACHE, FGI_PATH, GROUPS_CSV]:
    assert os.path.exists(p), f"Missing required file: {p}"

netdb_path = NETDB_EXP
assert os.path.exists(netdb_path), "Missing rt_netdb*.json"
netdb = json.load(open(netdb_path, "r"))

# ----------------------- Utilities -----------------------
def find_blocks(txt, token):
    spans=[]
    for m in re.finditer(rf'(^|\n)([ \t]*){re.escape(token)}\s*\{{', txt):
        start=m.start(0)+(0 if m.group(1)=="" else 1)
        i=txt.find("{", m.end(0)-1)+1
        depth=1; n=len(txt)
        while i<n and depth>0:
            if txt[i]=="{": depth+=1
            elif txt[i]=="}": depth-=1
            i+=1
        spans.append((start,i,txt[start:i], m.group(2)))
    return spans

def grab(txt, key):
    m=re.search(rf'(^|\n)\s*{re.escape(key)}\s*=\s*([^\r\n#]+)', txt)
    return m.group(2).strip() if m else None

# --- Group selection helper (defined early so it's available when called) ---
_group_label = re.compile(r'^\s*(?:Group\s+|G)?(\d+)\s*$', re.IGNORECASE)

def parse_selected_groups(sel, *, infer_all="from_seeds"):
    """
    Return a sorted list of positive group indices.
    Accepts 'all', 'Group N', 'G<N>', or explicit integers.
    infer_all: 'from_seeds' (default) or 'from_csv' to decide what 'all' means.
    """
    if isinstance(sel, str) and sel.lower() == "all":
        # default: infer from seeds present in the save (names defined later in the file)
        return sorted(set(seed_kerbin.keys()) | set(seed_other.keys())) if "seed_kerbin" in globals() else []
    out = []
    for item in (sel or []):
        if isinstance(item, int):
            gi = item
        else:
            m = _group_label.match(str(item))
            if not m:
                print(f"[warn] Unrecognized group label in config: {item!r} (expected 'Group N' or 'G<N>')")
                continue
            gi = int(m.group(1))
        if gi <= 0:
            print(f"[warn] Group index must be positive: {gi}")
            continue
        out.append(gi)
    return sorted(set(out))


# ----------------------- Load CSV groups & update DB -----------------------
dfg = pd.read_csv(GROUPS_CSV)
body_col = next((c for c in dfg.columns if c.strip().lower()=="name of body"), None)
group_col = next((c for c in dfg.columns if c.strip().lower()=="group"), None)
assert body_col and group_col, "Could not detect 'Name of body' and 'Group' columns in CSV"

def norm_name(s):
    s = str(s).strip()
    if s.lower() in ("sun","sol","kerbol"): return "Kerbol"
    return s

# Build new mapping dict (by name only; Kerbol retained but excluded from REM generation)
new_map = {}
for _, row in dfg.iterrows():
    b = norm_name(row[body_col])
    graw = str(row[group_col]).strip()
    # tolerant parse; store normalized "G#"
    m = re.search(r'(\d+)', graw)
    if b and m:
        new_map[b] = f"G{int(m.group(1))}"

# Persist to rt_groups_index.json
prev_map = {}
if os.path.exists(GROUPS_JSON):
    try:
        prev_map = json.load(open(GROUPS_JSON,"r")).get("index", {})
    except Exception:
        prev_map = {}

json.dump({
    "index": new_map,
    "source": "csv",
    "csv_path": GROUPS_CSV,
    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "lanes": {"equatorial_NETs": list(range(1,10))}  # NET1..NET9 reserved EQ
}, open(GROUPS_JSON,"w"))

# Emit current CSV & diff
pd.DataFrame([{"Body": b, "Group": g} for b,g in sorted(new_map.items())]).to_csv(GROUPS_CURR_CSV, index=False)
diff_rows=[]
for b in sorted(set(prev_map.keys()).union(new_map.keys())):
    old = prev_map.get(b)
    new = new_map.get(b)
    if old != new:
        diff_rows.append({"Body": b, "Old": old, "New": new})
pd.DataFrame(diff_rows).to_csv(GROUPS_DIFF_CSV, index=False)

# ----------------------- Load save & references -----------------------
text = open(SFS_IN,"r",errors="ignore").read()
mm = open(MMCACHE,"r",errors="ignore").read()

# FGI map
FGI = {}
with open(FGI_PATH,"r",errors="ignore") as f:
    for line in f:
        m=re.match(r"(\d+)\s+(.+)", line.strip())
        if m: FGI[m.group(2).strip()] = int(m.group(1))

# Group -> bodies 1..7 (exclude Kerbol and kerbin)
from collections import defaultdict
g_bodies = defaultdict(list)
for b, g in new_map.items():
    if b.lower() in ("kerbol","sun","kerbin"): 
        continue
    m = re.search(r'(\d+)', str(g))
    if not m: 
        continue
    gi = int(m.group(1))
    if 1 <= gi <= 7:
        g_bodies[gi].append(b)

# ----------------------- Orbit helpers -----------------------
def body_radius(body):
    for (s,e,blk,_) in find_blocks(mm,"Body"):
        nm = grab(blk,"name") or grab(blk,"Name")
        if nm and nm.lower()==body.lower():
            props = find_blocks(blk,"Properties")
            if props:
                r = grab(props[0][2],"radius")
                if r: return float(r)
    return None

def get_body_key(bname):
    for k in netdb.get("bodies",{}).keys():
        if k.lower()==bname.lower(): return k
    return None

def synthesize_equatorial_sma(body):
    """Derive an outer EQ lane from NET3..NET8 spacing if NET9 SMA missing."""
    bkey = get_body_key(body)
    if not bkey: return None
    smas=[]
    for n in range(3,9):
        preset = netdb["bodies"][bkey]["net_presets"].get(str(n)) if netdb["bodies"][bkey].get("net_presets") else None
        if preset and preset.get("sma_m"):
            try: smas.append(float(preset["sma_m"]))
            except: pass
    if not smas: return None
    smas = sorted(smas)
    if len(smas)>=2:
        steps=[smas[i+1]-smas[i] for i in range(len(smas)-1)]
        step = sum(steps)/len(steps)
        return smas[-1] + step
    return smas[-1]*1.08

def orbit_from_preset(body, net_id, slot, UT, force_eq=True, scale_alt=1.0):
    bkey = get_body_key(body)
    if not bkey: raise KeyError(f"{body} not in netdb")
    preset = netdb["bodies"][bkey].get("net_presets",{}).get(str(int(net_id)))
    if preset and preset.get("sma_m"):
        sma = float(preset["sma_m"])
        inc = float(preset.get("inc_deg",0.0)); ecc=float(preset.get("ecc",0.0))
        lan = float(preset.get("raan_deg_base",0.0)); lpe=float(preset.get("argp_deg",0.0))
    else:
        if int(net_id)==3:
            base = netdb["bodies"][bkey].get("net_presets",{}).get("1")
            if base and base.get("sma_m"):
                sma = float(base["sma_m"])*1.5
                inc = float(base.get("inc_deg",0.0)); ecc=float(base.get("ecc",0.0))
                lan = float(base.get("raan_deg_base",0.0)); lpe=float(base.get("argp_deg",0.0))
            else:
                raise KeyError(f"{body} missing SMA for NET{net_id}")
        elif 4 <= int(net_id) <= 9:
            sma = synthesize_equatorial_sma(body)
            if sma is None:
                raise KeyError(f"{body} cannot synthesize SMA for NET{net_id}")
            inc=ecc=lan=lpe=0.0
        else:
            raise KeyError(f"{body} missing SMA for NET{net_id}")
    if scale_alt != 1.0:
        R = body_radius(body) or 0.0
        alt = max(0.0, sma - R)
        sma = R + scale_alt*alt
    if force_eq and net_id<=9:
        inc = 0.0
    phase_deg = {1:0,2:90,3:180,4:270}[slot]
    mna = math.radians(phase_deg)
    return {"SMA":sma,"ECC":ecc,"INC":inc,"LAN":lan,"LPE":lpe,"MNA":mna,"EPH":UT,"REF":FGI[body],"IDENT":body}

def fmt_orbit(indent, fields):
    inner = indent + ("\t" if ("\t" in indent or indent=="") else "    ")
    order = ["SMA","ECC","INC","LAN","LPE","MNA","EPH","REF","IDENT"]
    lines = [indent+"ORBIT", indent+"{"]
    for k in order:
        v = fields.get(k)
        if v is not None:
            if isinstance(v, float): lines.append(f"{inner}{k} = {v:.6f}")
            else: lines.append(f"{inner}{k} = {v}")
    lines.append(indent+"}")
    return "\n".join(lines)

# ----------------------- Vessel helpers -----------------------
def get_top_level_name_span(vtxt):
    parts = find_blocks(vtxt,"PART")
    limit = parts[0][0] if parts else len(vtxt)
    m = re.search(r'(^|\n)(?P<i>\s*)name\s*=\s*([^\r\n#]+)', vtxt[:limit])
    if not m: return None,None,None
    start = m.start(0) + (0 if m.group(1)=="" else 1); end = m.end(0); indent = m.group('i')
    return start,end,indent

def set_vessel_name(vtxt, new_name):
    s,e,i = get_top_level_name_span(vtxt)
    assert s is not None, "Top-level vessel name not found"
    return vtxt[:s] + f"{i}name = {new_name}" + vtxt[e:]

def find_first_orbit_block(vtxt):
    for m in re.finditer(r'(^|\n)([ \t]*)ORBIT\s*\{', vtxt):
        os_ = m.start(0)+(0 if m.group(1)=="" else 1)
        oindent = m.group(2)
        j = vtxt.find("{", m.end(0)-1)+1
        depth=1; n=len(vtxt)
        while j<n and depth>0:
            if vtxt[j]=="{": depth+=1
            elif vtxt[j]=="}": depth-=1
            j+=1
        return os_, j, oindent, vtxt[os_:j]
    return None

def set_or_insert_field(vtxt, key, value, preferred_after=("stg","ctrl","PQSMin","root","name")):
    m = re.search(r'(^|\n)(?P<i>\s*){key}\s*=\s*[^\r\n#]+'.format(key=re.escape(key)), vtxt)
    if m:
        start = m.start(0) + (0 if m.group(1)=="" else 1); end = m.end(0); indent = m.group("i")
        return vtxt[:start] + f"{indent}{key} = {value}" + vtxt[end:]
    ins_pos = None; ins_indent = "\t\t\t"
    for anch in preferred_after:
        mv = re.search(r'(^|\n)(?P<i>\s*){anch}\s*=\s*[^\r\n#]+'.format(anch=re.escape(anch)), vtxt)
        if mv:
            ins_pos = mv.end(0); ins_indent = mv.group("i"); break
    if ins_pos is None: ins_pos = 0
    return vtxt[:ins_pos] + f"\n{ins_indent}{key} = {value}\n" + vtxt[ins_pos:]

def fix_ref_to_root(nv):
    parts = find_blocks(nv,"PART")
    if not parts: return nv
    mroot = re.search(r'(^|\n)\s*root\s*=\s*(\d+)', nv)
    root_idx = int(mroot.group(2)) if mroot else 0
    root_idx = max(0, min(root_idx, len(parts)-1))
    mu = re.search(r'(^|\n)\s*uid\s*=\s*([^\r\n#]+)', parts[root_idx][2])
    if mu:
        nv = set_or_insert_field(nv,"ref", mu.group(2))
    return nv

def rewrite_ids(vtxt):
    # vessel pid
    vtxt = re.sub(r'(^|\n)(\s*)pid\s*=\s*[^\r\n#]+', lambda m: f"{m.group(1)}{m.group(2)}pid = {str(uuid.uuid4())}", vtxt, count=1)
    # parts ids
    parts = find_blocks(vtxt,"PART")
    parts.sort(key=lambda x:x[0])
    out=[]; cur=0
    for ps,pe,ptxt,pindent in parts:
        ptxt = re.sub(r'(^|\n)(\s*)uid\s*=\s*[^\r\n#]+', lambda m: f"{m.group(1)}{m.group(2)}uid = {random.randint(100000000,2147483647)}", ptxt, count=1)
        ptxt = re.sub(r'(^|\n)(\s*)persistentId\s*=\s*[^\r\n#]+', lambda m: f"{m.group(1)}{m.group(2)}persistentId = {random.randint(10**12,10**13)}", ptxt, count=1)
        out.append((ps,pe,ptxt))
    sp=[]; cur=0
    for ps,pe,pt in out:
        sp.append(vtxt[cur:ps]); sp.append(pt); cur=pe
    sp.append(vtxt[cur:])
    return "".join(sp)

def set_state(nv):
    nv = set_or_insert_field(nv,"sit","ORBITING")
    nv = set_or_insert_field(nv,"landed","False")
    nv = set_or_insert_field(nv,"splashed","False")
    nv = set_or_insert_field(nv,"prst","False")
    return nv

def enable_antennas(nv):
    parts = find_blocks(nv,"PART")
    parts.sort(key=lambda x:x[0])
    sp=[]; cur=0
    for ps,pe,ptxt,pindent in parts:
        mods = find_blocks(ptxt,"MODULE")
        if mods:
            mods.sort(key=lambda x:x[0])
            out=[]; c2=0
            changed=False
            for ms,me,mtxt,mind in mods:
                mname = grab(mtxt,"name")
                if mname=="ModuleDataTransmitter":
                    # Ensure enabled flags
                    if re.search(r'(^|\n)\s*IsEnabled\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)IsEnabled\s*=\s*[^\r\n#]+', lambda m: f"{m.group(1)}{m.group(2)}IsEnabled = True", mtxt)
                    else:
                        mtxt += "\n\t\t\t\tIsEnabled = True"
                    if re.search(r'(^|\n)\s*enabled\s*=\s*', mtxt):
                        mtxt = re.sub(r'(^|\n)(\s*)enabled\s*=\s*[^\r\n#]+', lambda m: f"{m.group(1)}{m.group(2)}enabled = True", mtxt)
                    else:
                        mtxt += "\n\t\t\t\tenabled = True"
                    changed=True
                if mname=="ModuleDeployableAntenna":
                    for k,v in [("deployState","EXTENDED"),("isDeployed","True"),("stateString","Extended")]:
                        if re.search(rf'(^|\n)\s*{k}\s*=', mtxt):
                            mtxt = re.sub(rf'(^|\n)(\s*){k}\s*=\s*[^\r\n#]+', lambda m,kk=k,vv=v: f"{m.group(1)}{m.group(2)}{kk} = {vv}", mtxt)
                        else:
                            mtxt += f"\n\t\t\t\t{k} = {v}"
                    changed=True
                out.append((ms,me,mtxt))
            if changed:
                sp2=[]; c3=0
                for ms,me,mt in out:
                    sp2.append(ptxt[c2:ms]); sp2.append(mt); c2=me
                sp2.append(ptxt[c2:])
                ptxt="".join(sp2)
        sp.append(nv[cur:ps]); sp.append(ptxt); cur=pe
    sp.append(nv[cur:])
    return "".join(sp)

# ----------------------- Parse seeds & existing -----------------------
vessels = find_blocks(text,"VESSEL")
seed_kerbin={}; seed_other={}; existing_names=set()
for (s,e,vtxt,_) in vessels:
    m = re.search(r'(^|\n)\s*name\s*=\s*([^\r\n#]+)', vtxt)
    if not m: continue
    nm = m.group(2).strip()
    pat_k = re.compile(r"^Commsat-Kerbin-G(\d+)$"); pat_o = re.compile(r"^Commsat-Other-G(\d+)$"); mk = pat_k.fullmatch(nm); mo = pat_o.fullmatch(nm)
    
    if mk: seed_kerbin[int(mk.group(1))]=(s,e,vtxt)
    if mo: seed_other[int(mo.group(1))]=(s,e,vtxt)
    existing_names.add(nm)


selected_groups = parse_selected_groups(cfg.deploy.groups)
missingK = [g for g in range(1,8) if g not in seed_kerbin]
missingO = [g for g in range(1,8) if g not in seed_other]
assert not missingK and not missingO, f"Missing seeds -> Kerbin:{missingK} Other:{missingO}"

# Current UT
m_ut = re.search(r'\bUT\s*=\s*([0-9eE\.\-]+)', text)
UT = float(m_ut.group(1)) if m_ut else None

# Kerbin G -> NET mapping (EQ lanes)
G_to_NET = {1:3,2:4,3:5,4:6,5:7,6:8,7:9}

# ----------------------- Build new vessels -----------------------
placements=[]; new_vessels_txt=[]

def build_from(templ_vtxt, new_name, body, net_id, slot, scale_alt=1.0, group=None):
    nv = templ_vtxt
    nv = set_vessel_name(nv, new_name)   # safe top-level rename only
    nv = set_state(nv)
    ob = find_first_orbit_block(nv); assert ob, "Template missing ORBIT"
    os_,oe_,oindent,_ = ob
    scale = scale_alt
    if body.lower()=="dak": scale = 5.0  # safety
    fields = orbit_from_preset(body, net_id, slot, UT, force_eq=True, scale_alt=scale)
    if net_id <= 9:
        fields["INC"] = 0.0
    nv = nv[:os_] + fmt_orbit(oindent, fields) + nv[oe_:]
    nv = rewrite_ids(nv)
    nv = fix_ref_to_root(nv)
    nv = enable_antennas(nv)
    placements.append({
        "name": new_name, "body": body, "group": group, "net": net_id, "slot": slot,
        "SMA": fields["SMA"], "INC": fields["INC"], "ECC": fields["ECC"],
        "LAN": fields["LAN"], "LPE": fields["LPE"], "MNA": fields["MNA"]
    })
    return nv

# Kerbin hubs
for g in selected_groups:
    templ = seed_kerbin[g][2]; net_id = G_to_NET[g]
    for slot in (1,2,3,4):
        nm = f"Kerbin-G{g}-SAT{slot}-NET{net_id}"
        if nm in existing_names: 
            continue
        new_vessels_txt.append(build_from(templ, nm, "Kerbin", net_id, slot, 1.0, group=f"G{g}"))

# Remote constellations
for g in selected_groups:
    templ = seed_other[g][2]
    for b in g_bodies[g]:
        for slot in (1,2,3,4):
            nm = f"{b}-G{g}-SAT{slot}-NET3"
            if nm in existing_names:
                continue
            new_vessels_txt.append(build_from(templ, nm, b, 3, slot, 1.0, group=f"G{g}"))

# If nothing to add, still publish the updated groups DB and a "no-op" marker
if not new_vessels_txt:
    # Just emit the group files for you to verify
    print("No new vessels were generated (everything already present).")
    print("Group map updated & written:", GROUPS_JSON)
    print("Diff CSV:", GROUPS_DIFF_CSV)
    print("Current map CSV:", GROUPS_CURR_CSV)
else:
    # Insert new vessels into FLIGHTSTATE
    fsm = re.search(r'(^|\n)([ \t]*)(FLIGHTSTATE|FlightState|flightstate)\s*\{', text)
    assert fsm, "FLIGHTSTATE block not found"
    j = text.find("{", fsm.end(0)-1)+1
    depth=1; n=len(text)
    while j<n and depth>0:
        if text[j]=="{": depth+=1
        elif text[j]=="}": depth-=1
        j+=1
    fs_start = fsm.start(0)+(0 if fsm.group(1)=="" else 1)
    fs_end = j
    flightstate = text[fs_start:fs_end]
    insert_at = fs_start + flightstate.rfind("}")
    insertion = "\n" + "\n".join(new_vessels_txt) + "\n"
    new_text = text[:insert_at] + insertion + text[insert_at:]

    # Validation: ensure no PART name == vessel name
    def part_name_equals_vessel(vtxt, vname):
        for (ps,pe,ptxt,pindent) in find_blocks(vtxt,"PART"):
            m = re.search(r'(^|\n)\s*name\s*=\s*([^\r\n#]+)', ptxt)
            if m and m.group(2).strip()==vname:
                return True
        return False

    bad_list=[]
    for (s,e,vtxt,_) in find_blocks(new_text,"VESSEL"):
        vm = re.search(r'(^|\n)\s*name\s*=\s*([^\r\n#]+)', vtxt)
        if vm and part_name_equals_vessel(vtxt, vm.group(2).strip()):
            bad_list.append(vm.group(2).strip())

    if bad_list:
        diag = os.path.join(OUT_DIR, "regen_failed_validation.txt")
        with open(diag,"w") as f:
            f.write("Detected PART 'name' overwritten for vessels:\n")
            for nbad in bad_list: f.write(nbad+"\n")
        print("VALIDATION FAILED. See:", diag)
    else:
        out_sfs = os.path.join(OUT_DIR, os.path.basename(SFS_IN))
        with open(out_sfs,"w") as f:
            f.write(new_text)
        rep_csv = os.path.join(OUT_DIR, "placements_quicksave11.csv")
        with open(rep_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(placements[0].keys()))
            w.writeheader()
            for r in placements: w.writerow(r)
        zip_path = os.path.join(OUT_DIR, cfg.zip_name)
        with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out_sfs, arcname=os.path.basename(out_sfs))
            z.write(rep_csv, arcname=os.path.basename(rep_csv))

        print("CREATED:", len(new_vessels_txt), "new vessels")
        print("SFS_OUT:", out_sfs)
        print("CSV_OUT:", rep_csv)
        print("ZIP_OUT:", zip_path)
        print("Group map updated & written:", GROUPS_JSON)
        print("Diff CSV:", GROUPS_DIFF_CSV)
        print("Current map CSV:", GROUPS_CURR_CSV)

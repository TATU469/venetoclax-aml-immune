"""
Download BEAT AML drug sensitivity from Synapse.

Synapse project: syn10808323 (Tyner et al. 2018, Nature)
Target file:     syn26427390  beataml_wv1to4_probit_curve_fits.txt
                 (public-tier; free account required)

Usage:
    python download_synapse_beat_aml.py <project_dir> <synapse_token>

Get your Personal Access Token at:
    https://www.synapse.org/#!PersonalAccessTokens:
    → Account Settings → Personal Access Tokens → Create New Token
    (scope: view + download)
"""

import sys, os, shutil

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "."
TOKEN   = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SYNAPSE_TOKEN", "")
OUT_DIR = os.path.join(PROJECT, "data/raw/BEAT_AML")
os.makedirs(OUT_DIR, exist_ok=True)

if not TOKEN:
    print("ERROR: Synapse token required. Pass as argument or set SYNAPSE_TOKEN env var.")
    print("  Get token: https://www.synapse.org/#!PersonalAccessTokens:")
    sys.exit(1)

try:
    import synapseclient
except ImportError:
    print("Installing synapseclient...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "synapseclient"])
    import synapseclient

syn = synapseclient.Synapse()
syn.login(authToken=TOKEN, silent=True)
print("Logged in to Synapse.")

# BEAT AML wave 1-4 probit curve fits (drug sensitivity, public tier)
TARGETS = {
    "syn26427390": "beataml_wv1to4_probit_curve_fits.txt",
    "syn26427391": "beataml_wv1to4_sample_info.txt",
}

for syn_id, fname in TARGETS.items():
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Already exists: {fname}")
        continue
    print(f"Downloading {syn_id} → {fname}...")
    try:
        entity = syn.get(syn_id, downloadLocation=OUT_DIR, ifcollision="overwrite.local")
        # Rename to expected filename if Synapse saved with different name
        dl_path = entity.path
        if dl_path and os.path.basename(dl_path) != fname:
            shutil.move(dl_path, out_path)
        print(f"  Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")
    except Exception as e:
        print(f"  Failed {syn_id}: {e}")

print("Done.")

# ── PATCH: apply these two changes to analyze_kovacs_v3.py ──────────────────
#
# CHANGE 1: In write_summary(), replace the open() call:
#
#   OLD:  with open(out_path, "w") as f:
#   NEW:  with open(out_path, "w", encoding="utf-8") as f:
#
# CHANGE 2: In write_summary(), replace the "SIMULATION" block line:
#
#   OLD:  "  N=10000, binary LJ, ρ=1.2, NVT",
#   NEW:  "  N=10000, binary LJ, rho=1.2, NVT",
#
# CHANGE 3: In main(), replace the checkmark loop:
#
#   OLD:  print(f"  {'checkmark' if os.path.exists(f) else 'x'}  {f}")
#   NEW:  print(f"  {'OK' if os.path.exists(f) else 'MISSING'}  {f}")
#
# OR: just run the standalone fixed version below.
# ─────────────────────────────────────────────────────────────────────────────

# Standalone fixed write_summary + footer — drop-in replacement for v3:

import os

def write_summary_fixed(peak_result, frames, key_frames, out_path,
                         extract_fn):
    lines = [
        "="*68, "KOVACS MEMORY EFFECT - SUMMARY", "="*68, "",
        "SIMULATION",
        "  N=10000, binary LJ, rho=1.2, NVT",
        "  Quench T=2.0->0.3 (50k steps)  |  Age T=0.3 (100k steps)",
        "  Kovacs hold T=0.4 (200k steps)",
        "", "PE RELAXATION",
        f"  Initial dPE : {peak_result['initial_delta']:+.6f} LJ",
        f"  Inflection  : step {peak_result['peak_step']:,}  "
        f"  dPE={peak_result['delta_at_peak']:+.6f}",
        f"  Final dPE   : {peak_result['final_delta']:+.6f} LJ",
    ]
    if peak_result['relaxation_tau']:
        tau = peak_result['relaxation_tau']
        lines.append(
            f"  KWW tau     : {tau:,.1f} steps ({tau*0.005:.2f} tau_LJ)")

    lines += ["", "KEY SNAPSHOT STRUCTURAL METRICS",
              f"  {'Frm':<5}{'Label':<36}{'std_r':>8}{'r_max':>8}{'skew':>8}",
              "  "+"-"*65]
    for label, fidx in key_frames:
        fidx = min(fidx, len(frames)-1) if frames else 0
        if frames and fidx < len(frames):
            s = extract_fn(frames[fidx]['positions'], frames[fidx]['box'])
            lines.append(f"  {fidx:<5}{label[:36]:<36}"
                         f"{s['std_r']:>8.5f}{s['r_max']:>8.5f}"
                         f"{s['skewness']:>8.4f}")
    lines += ["", "="*68]

    # utf-8 encoding avoids Windows cp1252 crash
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved -> {out_path}")


def print_output_summary(output_files):
    print("\n" + "="*68)
    for fname in output_files:
        status = "OK     " if os.path.exists(fname) else "MISSING"
        print(f"  [{status}]  {fname}")
    print("="*68)
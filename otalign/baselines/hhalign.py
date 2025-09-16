import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from otalign.io.parser import convert_a3m_text_to_a2m


def run_hhalign_hhm_pair(
    q_hhm: str | Path,
    t_hhm: str | Path,
    *,
    hhalign_bin: str = "hhalign",
    out_dir: Optional[str | Path] = None,
    mode: str = "local",  # "local" | "global" | "glocal"
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = 300,
) -> Tuple[str, str]:
    """
    Run hhalign on two .hhm files and return pairwise alignment as two gapped A3M strings.
    We parse -oa3m and strip lowercase (insertions) per A3M convention.
    Robust 'mode' handling: try candidate flag patterns until one succeeds.
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_a3m = out_dir / "pair.aln.a3m"

    base_cmd = [
        hhalign_bin,
        "-i",
        str(q_hhm),
        "-t",
        str(t_hhm),
        "-oa3m",
        str(out_a3m),
    ]

    # Known variants observed across HHsuite versions (not all builds support all flags).
    MODE_CANDIDATES = {
        "local": [
            [],  # default is typically local-ish
            ["-local", "1"],
            ["-global", "0"],
        ],
        "global": [
            ["-global", "1"],
            ["-glob", "1"],  # older alias in some builds
            ["-local", "0"],
        ],
        "glocal": [
            ["-glocal", "1"],  # present in some hhalign builds
            ["-local", "1", "-global", "1"],  # emulate if supported
        ],
    }

    if mode not in MODE_CANDIDATES:
        raise ValueError(f"Unknown mode={mode}. Choose from local|global|glocal")

    tried_errors: List[str] = []
    extra_args = list(extra_args or [])

    for cand in MODE_CANDIDATES[mode]:
        cmd = base_cmd + cand + extra_args
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=timeout)
            # success
            recs = convert_a3m_text_to_a2m(out_a3m.read_text())
            if len(recs) < 2:
                raise RuntimeError(f"hhalign produced no pairwise A3M: {out_a3m}")
            a1 = recs[0]["seq"].upper().replace(".", "-")
            a2 = recs[1]["seq"].upper().replace(".", "-")
            if len(strip_a3m_lowercase(a1)) != len(strip_a3m_lowercase(a2)):
                raise ValueError("Aligned strings have different lengths after stripping lowercase.")
            return a1, a2
        except subprocess.CalledProcessError:
            tried_errors.append(f"cmd failed: {' '.join(cmd)}")
        except Exception as e:
            tried_errors.append(str(e))

    raise RuntimeError(f"hhalign failed for mode={mode}. Tried {len(MODE_CANDIDATES[mode])} flag patterns.\n" + "\n".join(tried_errors[:4]) + ("\n..." if len(tried_errors) > 4 else ""))


def strip_a3m_lowercase(s: str) -> str:
    """
    In A3M, lowercase letters are insertions relative to the query/target.
    For a 2-sequence alignment, remove lowercase while keeping uppercase and '-'.
    """
    out = []
    for ch in s:
        if ch == "-" or ("A" <= ch <= "Z"):
            out.append(ch)
        # skip lowercase
    return "".join(out)

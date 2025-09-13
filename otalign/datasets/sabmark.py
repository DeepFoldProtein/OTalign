import os
import re
from itertools import combinations
from typing import Any

import torch
from torch.utils.data import Dataset


class SABmark(Dataset):
    """PyTorch Dataset for reference alignments in FASTA format.

    Each file listed in ids.txt contains multiple aligned sequences of equal length,
    with gaps represented by '.' characters. This Dataset returns all pairs of sequences
    within the same file along with the ground-truth coupling matrix T.
    """

    def __init__(self, data_dir, ids_file, regex=r"*"):
        """Args:
        data_dir (str): Directory containing the FASTA alignment files.
        ids_file (str): Path to ids.txt listing FASTA filenames (one per line).

        """
        self.data_dir = data_dir
        # Read list of FASTA files
        with open(ids_file) as f:
            self.files = [line.strip() for line in f if line.strip()]

        # Load alignments per file
        regex = re.compile(regex)
        self.alignments = {}  # filename -> list of (seq_id, align_str)
        for fn in self.files:
            if not regex.match(fn):
                continue
            path = os.path.join(data_dir, fn)
            self.alignments[fn] = self._parse_fasta(path)

        # Build list of (filename, idx_i, idx_j) for all sequence pairs in each file
        self.pairs = []
        for fn, seqs in self.alignments.items():
            for i, j in combinations(range(len(seqs)), 2):
                self.pairs.append((fn, i, j))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fn, i, j = self.pairs[idx]
        seq_id_a, aln_a = self.alignments[fn][i]
        seq_id_b, aln_b = self.alignments[fn][j]

        # Extract ungapped sequences and mapping from alignment positions to sequence indices
        seq_a, pos_map_a = self._ungap_and_map(aln_a)
        seq_b, pos_map_b = self._ungap_and_map(aln_b)

        len_a, len_b = len(seq_a), len(seq_b)
        # Build coupling matrix T
        T = torch.zeros((len_a, len_b), dtype=torch.uint8)
        # For each alignment column, if both have residues, mark the coupling
        for aln_idx, (pa, pb) in enumerate(zip(pos_map_a, pos_map_b, strict=False)):
            if pa is not None and pb is not None:
                T[pa, pb] = 1

        return {
            "fam": fn,
            "seq_id_a": seq_id_a,
            "seq_id_b": seq_id_b,
            "seq_a": seq_a,
            "seq_b": seq_b,
            "aln": T,
        }

    def _parse_fasta(self, filepath):
        """Simple FASTA parser that returns a list of (seq_id, sequence_str)."""
        entries = []
        with open(filepath) as f:
            seq_id = None
            seq_lines = []
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                if line.startswith(">"):
                    if seq_id is not None:
                        entries.append((seq_id, "".join(seq_lines)))
                    seq_id = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if seq_id is not None:
                entries.append((seq_id, "".join(seq_lines)))
        return entries

    def _ungap_and_map(self, aln_str):
        """Remove gaps ('.') from alignment string and return
        the ungapped sequence plus a map from each alignment
        position to the index in the ungapped sequence or None.

        Returns:
            seq (str): Ungapped sequence.
            pos_map (list[int|None]): List of length len(aln_str), mapping to ungapped indices.

        """
        seq = []
        pos_map = []
        ungapped_idx = 0
        for ch in aln_str:
            if ch in ".-":
                pos_map.append(None)
                pass
            else:
                seq.append(ch)
                pos_map.append(ungapped_idx)
                ungapped_idx += 1
        return "".join(seq).upper(), pos_map


def make_collate_fn(max_seqlen: int):
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = batch[0].keys()

        for key in keys:
            vals = [ex[key] for ex in batch]

            # Non-tensor
            if not torch.is_tensor(vals[0]):
                out[key] = vals
                continue

            padded = []
            masks = []

            if key == "aln":
                for v in vals:
                    v = v[:max_seqlen, :max_seqlen]
                    H, W = v.shape

                    buf = v.new_full((max_seqlen, max_seqlen), 0.0)
                    buf[:H, :W] = v

                    mask = buf.new_zeros((max_seqlen, max_seqlen), dtype=torch.bool)
                    mask[:H, :W] = True

                    padded.append(buf)
                    masks.append(mask)
            else:
                seqs = []
                orig_lens = []

                for v in vals:
                    v = v[:max_seqlen]
                    L = v.size(0)
                    seqs.append(v)
                    orig_lens.append(L)

                    feat_shape = seqs[0].shape[1:]
                    for v, L in zip(seqs, orig_lens, strict=False):
                        buf = v.new_full((max_seqlen, *feat_shape), 0.0)
                        buf[:L, ...] = v
                        padded.append(buf)

                        mask = v.new_zeros((max_seqlen,), dtype=torch.bool)
                        mask[:L] = True
                        masks.append(mask)

            out[key] = torch.stack(padded, dim=0)
            out[f"{key}_mask"] = torch.stack(masks, dim=0)

        return out

    return collate_fn

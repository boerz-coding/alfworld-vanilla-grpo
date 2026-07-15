#!/usr/bin/env python3
"""Paired exact McNemar between two 140-game evaluation files.

Usage:
    python3 tools/mcnemar.py A/games_round1.jsonl B/games_round1.jsonl

Pairs games by their ALFWorld task/trial path suffix (root-independent),
so files produced on different machines/data roots still pair correctly.
"""
import json
import sys
from math import comb


def wins(path):
    d = {}
    for line in open(path):
        g = json.loads(line)
        p = g["gamefile"]
        i = p.find("json_2.1.1")
        key = p[i:] if i >= 0 else p
        d[key] = bool(g.get("won"))
    return d


def mcnemar(a, b):
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, len(keys), 1.0
    p = sum(comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n * 2
    return n01, n10, len(keys), min(p, 1.0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = wins(sys.argv[1]), wins(sys.argv[2])
    a_only, b_only, paired, p = mcnemar(A, B)
    if paired < max(len(A), len(B)):
        print(f"WARNING: only {paired} games paired "
              f"(A has {len(A)}, B has {len(B)}) — check data roots/splits")
    print(f"A wins {sum(A.values())}/{len(A)}  B wins {sum(B.values())}/{len(B)}")
    print(f"discordant: A-only {a_only}, B-only {b_only}  |  exact McNemar p = {p:.4f}")

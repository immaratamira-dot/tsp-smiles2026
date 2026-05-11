# TSP Solver — Travelling Salesman Problem in C

A lightweight C implementation of the Travelling Salesman Problem using **Nearest Neighbour heuristic** combined with **2-opt local search** improvement.

Submitted as part of the **SMILES 2026** summer school application.

---

## Quick Start

```bash
gcc -O2 -o tsp tsp.c -lm
./tsp
```

## Files

| File | Description |
|---|---|
| `tsp.c` | Main C source code |
| `SOLUTION.md` | Full report: problem, approach, results, analysis |
| `README.md` | This file |

## Algorithm

1. **Nearest Neighbour** — greedy construction of an initial tour
2. **2-opt** — iterative improvement by swapping edge pairs
3. Both phases run from **every starting city**; the best tour is kept

## See Also

Full write-up with complexity analysis, results, and discussion in [`SOLUTION.md`](./SOLUTION.md).

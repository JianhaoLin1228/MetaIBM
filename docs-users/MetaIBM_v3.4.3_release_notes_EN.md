# MetaIBM v3.4.3 Release Notes

**Example / documentation release.** The `metaibm` package is unchanged from v3.4.2 — nothing in `metaibm/`, `extension/` or the CSV / schedule contracts is touched, so every existing model script keeps running as before.

---

## What is new

- **`examples/example2/` — a second tutorial: eco-evolutionary dynamics on alternative stable states (ATS) under rapid environmental change.**
- It asks whether two species with different thermal optima produce a **hysteresis loop** — a landscape whose composition depends on the direction the climate came from, not only on the current climate — and which parameters move its edges.
- Background: Scheffer et al. (2001) (fold bifurcation, tipping point, hysteresis) and Dakos et al. (2019) (standing variation can flatten the fold; adaptation can delay or advance a shift).

### Files

| file | role |
|---|---|
| `example2.ipynb` | annotated end-to-end notebook: background → model → design → results; runs the parameter grid with `multiprocessing` |
| `ats.py` | the same model with nothing that only exists to display results; `main()` is one parameter combination, same shape as `experiments/model.py` |
| `mpi_running.py` | MPI launcher: holds the parameter grid, allocates runs across ranks longest-processing-time-first |
| `tmp_nb_code2.py` | notebook helpers — loading the recorded tables, building the tables and figures |
| `bootstrap_metaibm.py` | puts the project root on `sys.path` |

### The model

- Two **mainland source pools**, one patch / one habitat each, whose environment equals one species' optimum: sp1 cold-adapted (0.2), sp2 warm-adapted (0.8). Each is burned in for 100 steps to carry standing variation, then frozen — the time loop only draws colonists from it.
- **100 patches on a 10 × 10 grid**, one habitat each, coupled by weak dispersal (`dispersal_amomg_rate=0.0001`), thinned by patch disturbance, and fed by `propagules_rain_num=10` from each mainland every step.
- **The climate:** one environmental axis offset by ±0.1 every 100 steps between step 99 and step 700 of 800 → 7 offsets, 0.2 → 0.9 (warming) or 0.8 → 0.1 (cooling). Only the mean moves; the variance never does.
- **One time-step:** selection (`niche_gaussian`) → reproduction + mutation into `offspring_pool` → the climate step → colonization from both mainlands → dispersal among patches → local germination → patch disturbance → clear the pools.
- Two tables are recorded every 4 steps: `species_distribution_over_time.csv.gz` and `phenotype_distribution_over_time.csv.gz`, plus `logger.log`, into a folder that encodes the run's own parameters.

### The design

- `3 × 3 × 2 = 18` runs at `rep = 0`: `(reproduce_mode, mutation_rate)` pairs `('asexual', 1e-5)`, `('asexual', 1e-4)`, `('sexual', 1e-4)` × `patch_dist_rate` `0.001`, `0.01`, `0.03` × climate direction (warming / cooling).
- The pairs are swept as pairs, not crossed: 1 vs 2 isolates mutation rate, 2 vs 3 isolates reproduction mode.

### The results in the notebook

- **§3.1** — does the trait keep up with the moving optimum? Trait lag per disturbance level, adaptation rates overlaid.
- **§3.2** — **alternative stable states**: warming and cooling compared at the same environment; the tipping point of each branch, and the horizontal distance between them as the width of the loop.
- **§3.3** — two ways to lose the loop: heavy propagule rain (`propagules_rain_num` 10 → 500) removes bistability; a patchy environment (an independent `Normal(0, 0.30)` offset per patch) buffers local extinction under rapid climate change.

---

## How to run

```bash
# notebook (also runnable in the browser via the Binder badge in README.md)
cd examples/example2 && jupyter lab example2.ipynb

# one parameter combination, no plotting
cd examples/example2 && python ats.py

# the whole 18-run grid, one run per rank
cd examples/example2 && mpiexec -np 18 python mpi_running.py
```

Run from `examples/example2/` — output paths are relative to the working directory, so the runs land next to the notebook that reads them.

---

## Compatibility

- No API change, no behaviour change in the core package. Results produced with v3.4.2 remain reproducible under v3.4.3.
- `example2` uses the hand-coded loop (like `experiments/`), not the `simulator` schedule DSL.
- Run outputs and figures (`*.csv.gz`, `*.log`, `*.jpg`, `*.png`, `*.gif`) are git-ignored, so a fresh clone contains the code and the notebook, and the runs have to be executed locally to regenerate them.
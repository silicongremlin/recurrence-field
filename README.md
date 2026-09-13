# Recurrence Field Harness

A small experimental harness for generating deterministic recurrence fields, sweeping controlled parameters, and measuring structure in rendered outputs.

The implementation uses a bounded iterative recurrence with configurable initialization, perturbation, and rendering parameters. This repository focuses on the experimental harness, visualization pipeline, and measurement tooling rather than the derivation of the underlying recurrence.

## Quick start

```bash
pip install -e .
recurrence-harness render --seed 404 --alpha 0.1745 --size 240 --steps 1600
recurrence-harness sweep --seed 404 --alpha 0.05 0.10 0.1745 0.25
recurrence-harness analyze outputs --pairwise
```

Generated files are written to ignored output directories. The analysis layer reports descriptive image statistics and structural distances; renders do not establish causality or physical dynamics.

## Status

Private research artifact pending publication review. Interfaces may evolve as the harness is refined.




## License

MIT. Fork it, modify it, and build your own version. This repository is provided as-is and carries no maintenance or support commitment.


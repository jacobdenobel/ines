# PBO evaluation accounting

The PBO table in the paper uses a specialized evaluation counter that differs
from IOH's ordinary batch counter:

- for dimensions below 10, offspring are sampled and evaluated sequentially;
  sampling stops immediately when an optimum is found, so the last generation
  need not contain all `lambda` offspring;
- when a raw DG step is the all-zero vector, its objective value is copied from
  the current parent; the unchanged candidate is not submitted to IOH and is
  not counted;
- for dimensions of 10 and above, the normal batch of `lambda=10` offspring is
  sampled before termination is checked;
- failed runs still stop after the configured number of sampled offspring, but
  their reported evaluation count excludes all-zero mutations.

This explains both the sub-`lambda` ERTs at small dimensions and most of the
difference with a straightforward IOH reproduction, which counts every member
of a submitted batch.

The reproduction driver now implements these rules directly with
`BarebonesINES`, the short original algorithm in `src/ines/barebones.py`. It
does not route paper runs through the extended recombination-capable optimizer.
The generated PBO CSV includes the manuscript ERT and the reproduced-to-paper
ratio as audit columns; modest differences are expected from 25 stochastic
runs and the selected RNG protocol.

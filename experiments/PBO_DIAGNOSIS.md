# Why the reproduced PBO ERTs are larger

The discrepancy is not caused by the packaged INES update. We traced the PBO
path against `integer-es` commit
`ae0fdb80cb693eddf4ecb959e3319fc6b279a058`, including:

- `lambda = 10`, `mu = 1` and first-best selection;
- `delta_i(0) = 1/n`, `c = 1 - 1.5/n`, and `eta = (2/n)^(1/3)`;
- cyclic binary mapping `(x + z) mod 2`;
- the selected raw DG mutation in the sufficient statistic;
- a single legacy NumPy random stream reused across repetitions;
- a `10,000 n` evaluation budget and the usual unsuccessful-run ERT formula.

With that source-compatible protocol, 25 local runs at `n=100` produced ERTs
of 1,274.4 on OneMax and 8,350.0 on LeadingOnes. The isolated-stream protocol
produced 1,358.4 and 8,359.6, respectively. Thus RNG modernization explains
only a small part of the OneMax difference and essentially none of the
LeadingOnes difference. The manuscript values are 759 and 7,138.

There is also a decisive accounting contradiction. The checked-in algorithm
always evaluates ten offspring before it can stop, so a run uses at least ten
objective evaluations. The manuscript table contains INES ERTs of 3 and 4.
Those values cannot be objective-evaluation ERTs from this implementation.
Nor are they generation ERTs obtained by dividing all reported values by ten.

The most defensible conclusion is that the PBO table was generated from an
uncommitted/intermediate experiment state or with a different, undocumented
counter. Changing the public algorithm until it matches that table would no
longer preserve `integer-es` as the source of truth.

The reproduction script therefore makes the discrepancy auditable instead:

- it runs all nine PBO dimensions from the table;
- it defaults to the source-compatible legacy RNG protocol;
- it reports evaluation ERT and generation ERT separately;
- it includes the manuscript ERT and ratio in adjacent CSV columns;
- it never silently rescales objective evaluations.

One independent runner defect was found and fixed: a budget exactly divisible
by `lambda` previously stopped one population early. Failed runs now consume
the complete allowed budget, matching `integer-es` and the ERT definition.


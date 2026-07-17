## Per (model, language)
| model | lang | n | parse | interp | under-triage | clar_recall | clar_spec | indirect_cult |
|---|---|---|---|---|---|---|---|---|
| M2ml | en | 540 | 1.000 | 0.617 | 0.463 | 0.208 | 0.878 | 0.611 |
| M2ml | fr | 540 | 0.998 | 0.614 | 0.440 | 0.333 | 0.862 | 0.600 |
| M2ml | ar | 540 | 1.000 | 0.617 | 0.420 | 0.375 | 0.849 | 0.611 |
| M3ml | fr | 540 | 0.998 | 0.605 | 1.000 | 0.000 | 1.000 | 0.618 |
| M3ml | ar | 540 | 0.998 | 0.618 | 0.998 | 0.000 | 0.998 | 0.578 |
| M3ml_v2 | en | 540 | 1.000 | 0.644 | 0.439 | 0.000 | 1.000 | 0.667 |
| M3ml_v2 | fr | 540 | 0.996 | 0.617 | 0.450 | 0.000 | 1.000 | 0.629 |
| M3ml_v2 | ar | 540 | 0.994 | 0.646 | 0.443 | 0.000 | 1.000 | 0.644 |

## Cross-language consistency (aligned triples)
| model | langs present | risk consistency | category consistency | n_groups |
|---|---|---|---|---|
| M2ml | en,fr,ar | 0.711 | 0.824 | 540 |
| M3ml | fr,ar | 0.994 | 0.880 | 540 |
| M3ml_v2 | en,fr,ar | 0.681 | 0.837 | 540 |

## McNemar (paired, per language) — p-values
| lang | pair | parse | category | risk | clarification |
|---|---|---|---|---|---|
| en | M2ml vs M3ml_v2 | 1.00e+00 | 2.28e-01 | 1.12e-02 | 4.77e-12 |
| fr | M2ml vs M3ml | 4.80e-01 | 7.45e-01 | 2.24e-64 | 5.62e-12 |
| fr | M2ml vs M3ml_v2 | 1.00e+00 | 1.00e+00 | 8.78e-01 | 1.62e-11 |
| fr | M3ml vs M3ml_v2 | 1.00e+00 | 4.17e-01 | 4.96e-65 | 1.00e+00 |
| ar | M2ml vs M3ml | 1.00e+00 | 9.34e-01 | 2.45e-66 | 1.48e-12 |
| ar | M2ml vs M3ml_v2 | 2.48e-01 | 2.68e-01 | 9.38e-01 | 7.30e-12 |
| ar | M3ml vs M3ml_v2 | 6.17e-01 | 5.53e-02 | 6.67e-66 | 1.00e+00 |

% LaTeX rows: model & lang & interp & under-triage & clar_recall
M2ml & en & 0.617 & 0.463 & 0.208 \\
M2ml & fr & 0.614 & 0.440 & 0.333 \\
M2ml & ar & 0.617 & 0.420 & 0.375 \\
M3ml & fr & 0.605 & 1.000 & 0.000 \\
M3ml & ar & 0.618 & 0.998 & 0.000 \\
M3ml_v2 & en & 0.644 & 0.439 & 0.000 \\
M3ml_v2 & fr & 0.617 & 0.450 & 0.000 \\
M3ml_v2 & ar & 0.646 & 0.443 & 0.000 \\

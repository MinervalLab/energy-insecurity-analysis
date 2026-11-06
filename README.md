# Household energy insecurity and residential energy intensity (RECS 2020/2015)

Analysis code for a study of household energy insecurity and residential
energy intensity using the U.S. Residential Energy Consumption Survey (RECS)
public-use microdata, with survey-design-based (jackknife / BRR) inference.

## Requirements

- Python ≥ 3.11 with `numpy`, `pandas`, `matplotlib`, `openpyxl`
  (exact versions in `requirements.txt`)

## Data (not included — public, free)

Download from the U.S. Energy Information Administration and place in a
`data/` folder next to the scripts (see `data/README.md` for links and
SHA-256 checksums):

- RECS 2020 public-use microdata v7 (`recs2020_public_v7.csv`) + codebook
- RECS 2015 public-use microdata v4 (`recs2015_public_v4.csv`) + codebook

By default the scripts read from `./data` and write to `./result`. Either
location can be changed with the environment variables `RECS_DATA` and
`RECS_RESULT`.

## Run order

```
python validate_codes.py           # optional input self-check (~1 min)
python s1_data_table1.py           # ~2 min
python s2_premium.py               # ~8 min
python s3_decomposition.py         # ~5 min
python s4_heterogeneity.py         # ~8 min
python s5_mechanism.py             # ~7 min
python s6_pandemic_2015.py         # ~4 min
python s7_counterfactual.py        # ~10-15 min (full-pipeline jackknife)
python s8_enduse.py                # ~8 min
python s9_revision_robustness.py   # ~6 min
python verify.py                   # all checks must PASS
python fig1_premium_specs.py       # figure scripts (seconds each)
python fig2_decomposition.py
python fig3_heterogeneity.py
python fig4_fuel_enduse_climate.py
python fig5_replication2015.py
python fig6_double_burden.py
python fig7_state_map.py
```

Each `s` script loads the data independently and can be run alone; `fig`
scripts only read files already written to the result folder.

## Scripts and outputs

Some intermediate data files carry historical internal names (e.g.,
`fig3_mech.json` feeds `fig4`, `fig2_data.json` feeds `fig6`); the table
below is authoritative.

| Script | Content | Key outputs (result folder) |
|---|---|---|
| s0_common.py | shared library: data loading, sample definition, design matrices, WLS, jackknife/BRR (imported, not run) | — |
| s1_data_table1.py | data checks; descriptive statistics by energy-security status | table1_descriptives.csv |
| s2_premium.py | raw gap; nested specifications M0–M4, state fixed effects, accounting and floor-area variants | table2_premium.csv, fig1a_specs.json, fig5_crossyear_2020.json |
| s3_decomposition.py | Oaxaca–Blinder decomposition and alternative reference coefficients | table3_oaxaca.csv, table3b_altrefs.csv, table3c_structure_split.csv |
| s4_heterogeneity.py | severity, dimensions, sample restrictions, interaction models | table4_heterogeneity.csv, table4b_subsample_appendix.csv |
| s5_mechanism.py | fuel-specific models, participation LPMs, climate interactions, portable heaters | table4b_mechanism.csv, fig3_mech.json |
| s6_pandemic_2015.py | replication on RECS 2015 | table5_2015.csv, fig3_2015.json, fig5_crossyear_2015.json |
| s7_counterfactual.py | peer-benchmark accounting scenario, placebo, double-burden group, state estimates | counterfactual_summary.csv, state_cf.json, state_table_appendix.csv, fig2_data.json |
| s8_enduse.py | end-use decomposition of the conditional difference | table6_enduse.csv, table6b_enduse_areabins.csv, table6c_refrig_memo.csv, fig_enduse.json |
| s9_revision_robustness.py | floor-area functional form and strata; benchmark sensitivity; peer counts; trimming; broader insecurity definition | table7_area_robustness.csv, table8_benchmark_sensitivity.csv, table8b_peer_counts.csv, table9_trim99.csv, table10_broad_definition.csv |
| fig1_premium_specs.py | premium across nested specifications | fig1_premium_specs.png/.pdf |
| fig2_decomposition.py | decomposition contributions | fig2_decomposition.png/.pdf |
| fig3_heterogeneity.py | robustness and heterogeneity forest plot | fig3_heterogeneity.png/.pdf |
| fig4_fuel_enduse_climate.py | fuel, end-use, climate, portable heaters (2×2) | fig4_fuel_enduse_climate.png/.pdf |
| fig5_replication2015.py | 2015 vs 2020 replication | fig5_replication2015.png/.pdf |
| fig6_double_burden.py | double-burden group profile | fig6_double_burden.png/.pdf |
| fig7_state_map.py | state-level geography of above-benchmark consumption | fig7_state_map.png/.pdf |

## Verification

Three layers:

1. `validate_codes.py` — value-domain and completeness checks on the raw
   data; aborts on failure.
2. In-script assertions — algebraic identities verified at run time
   (decomposition closure, end-use coefficient closure, the
   intensity/total-consumption coefficient identity, population
   reconciliations), with tolerances at machine-precision level.
3. `verify.py` against `expected_results.json` — numerical and structural
   checks of the core pipeline outputs; every check must report PASS.

## License and citation

Code is released under the MIT License (see `LICENSE`). Citation metadata
is provided in `CITATION.cff`.

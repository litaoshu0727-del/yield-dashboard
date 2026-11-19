# Yield Monitoring Dashboard (yield-dashboard)

[![tests](https://github.com/litaoshu0727-del/yield-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/litaoshu0727-del/yield-dashboard/actions/workflows/tests.yml)

**English** | [中文](README.zh-CN.md)

**In one sentence:** test data for 811k real wafers stored in SQLite, lot yield of each product monitored with SPC control charts, and every alarm lot traceable down to its individual wafer maps.

| What | Result |
|---|---|
| SQLite storage, SQL aggregation, Streamlit interface | 154 products, 811k wafers; the dashboard opens the database read-only |
| Three control charts: individuals (Western Electric rules), EWMA, CUSUM | Simulated sustained 1σ shift: the individuals chart alarms after 35 lots, EWMA and CUSUM after 9 |
| Monitor 1,647 lots of the product with the most lots | 14 low-yield alarms, all single-lot events; no sustained yield drop |
| Find out why lot yield alternated up and down on the chart | About 26% of lots turned out to be repeat or retest records of the same wafers; once separated, the chart behaves normally |
| Plug in the CNN trained in the wafer-map project | Wafers with defect pattern information on the main product grow from 287 to 1,198 |
| Fit four classic yield models | Defect density D0 of about 0.17–0.25 per cm² across products |

![Control chart page: lot yield, control limits and the lots that trigger run rules](docs/control_chart.png)

## What the dashboard has

| Page | Contents |
|---|---|
| Control chart | Switch between three charts: individuals (four Western Electric run rules), EWMA and CUSUM (with a list of detected shifts and their first and last lots); below, fail rates of edge dies against inner dies |
| Defect Pareto | Failing dies by defect pattern, from engineer labels or classifier predictions (with an adjustable confidence threshold) |
| Lot drill-down | Pick a lot and see the bin map of each wafer, with the classifier's pattern and confidence |
| Yield model | Yield of each product against die area, fitted with the Poisson, Murphy, Seeds and negative binomial yield models, giving the defect density D0 |

## Quick start

```bash
pip install -e ".[dev]"
pytest                 # 31 tests; one starts the dashboard from an empty data folder, as the hosted demo does
streamlit run app.py   # with no database, the app first builds a small synthetic demo database (a few seconds)
```

With the real data (WM-811K `LSWMD.pkl`, about 2 GB):

```bash
python -m yielddash.etl --source path/to/LSWMD.pkl --db data/yield.db
```

Building the database takes about 40 seconds and produces about 270 MB.

Optional: give every wafer a defect pattern with the CNN trained in the [wafer-map-yield](https://github.com/litaoshu0727-del/wafer-map-yield) project (needs PyTorch; about 7 minutes on a CPU for 811k wafers):

```bash
pip install -e ".[classify]"
# export the model in wafer-map-yield
python -m wafermap.export --run reports/wm811k_seeds/cnn_w0.5_s0 --out ../yield-dashboard/models/wafer_cnn
# back here, score the wafers into the database's predictions table
python -m yielddash.classify --db data/yield.db --model models/wafer_cnn.pt2
```

The dashboard itself does not need PyTorch; it only reads the `predictions` table, and the related columns and options do not appear when the table is missing.

## How the data is organised

Three tables: `products`, `wafers` (one row per wafer, with the compressed bin map stored as a BLOB) and `lot_yield` (lot summaries computed when the database is built). All aggregation in the dashboard is written in SQL, in `yielddash/queries.py`, including window functions (moving averages, cumulative shares). The dashboard opens the database read-only.

WM-811K has no product names and no timestamps, so two conventions are used:

- **Products** are told apart by wafer map grid size plus die count, for example `32x29-710`. Products have different yield levels and must be monitored separately.
- **Time order** is taken from the lot number.

## A data problem: the same wafers recorded several times

When all lots were first put on a control chart, yield jumped back and forth between neighbouring lots (alternating between about 83% and about 99.9%), and the control limits were too wide to mean anything. Comparing the wafer maps one by one showed that **consecutive lot numbers are often the same wafers recorded again at a later test step**: the same slots, exactly the same die positions, and fail maps that are related in a regular way. The dataset documentation does not mention this; what follows is inferred from the wafer maps.

When the database is built, each lot is compared with the previous lot number and put into one of three classes:

| Class | Rule | Lots | Wafers |
|---|---|---|---|
| `first` first test | Neither of the relations below to the previous lot | 34,344 | 623,945 |
| `repeat` repeat record | Fail map almost identical to the previous lot, usually with a few more fails, like a cumulative result for the same wafers | 3,997 | 65,159 |
| `retest` retest record | Fewer than 20% of the previous lot's fails, and either all of them off the dies that already failed or all of them on those dies | 7,952 | 122,345 |

The dashboard monitors first-test records by default; the other two classes can be selected in the sidebar.

There is also a group of lots with almost no fails that cannot be matched to a neighbouring lot, but whose yields form a cluster of their own (above 99.5%) with an empty band between them and normal lots. This happens in 6 of the 106 products with enough lots, 504 lots in all. A sidebar option, on by default, hides them and can be switched off. Products whose yield is genuinely high are not affected by this option.

## Results on the real data

The product with the most lots, `32x29-710` (first test, at least 5 wafers per lot, 1,647 lots left after hiding 205 near-zero-fail lots):

- With the first 100 lots as baseline: centre line 82.2%, σ of 3.9 percentage points, control limits 70.5%–93.8%.
- 14 lots (0.9%) trigger a low-yield alarm, all under rule 1 (beyond 3σ); the worst lot has a yield of only 17.5%.
- 444 lots trigger a high-side signal, 434 of them under rule 4 (8 points in a row on the same side of the centre line): after the baseline, yield is higher than the baseline level overall. These signals are not alarms and are drawn as hollow points to tell them apart.
- On this product the fail rate of edge dies (about 20%–30%) stays at about twice that of inner dies (about 10%–15%).

EWMA and CUSUM (same baseline, same σ):

- Without clipping, CUSUM reports 4 "yield drops" and EWMA has 24 lots below its lower limit. Checking each one shows that they are all triggered by single scrapped lots (z as low as −16): one point pushes the cumulative sum past the threshold, and it then takes dozens of lots to return to zero, while the median yield inside these stretches is 82%–85%, not below the centre line.
- With each lot's z clipped to ±3, neither chart has any low-side signal: this product has **no sustained yield drop**. The low-yield problems are all single-lot events, which rule 1 of the individuals chart handles.
- On the high side, CUSUM finds 4 sustained upward shifts, with yields of about 84%–86%, about 2 points above the baseline. The first starts at lot 96 and ends around lot 309. This is the same thing as the many rule 4 high-side signals on the individuals chart, but CUSUM gives the first and last lot.

Yield models (154 products with at least 50 first-test lots, assuming 300 mm wafers):

| Model | D0 (per cm²) | Clustering factor α | Fit error (yield points) |
|---|---|---|---|
| Poisson | 0.168 | – | 0.078 |
| Murphy | 0.171 | – | 0.078 |
| Seeds | 0.185 | – | 0.078 |
| Negative binomial | 0.250 | 0.23 | 0.077 |

The four models fit almost equally well; the data cannot tell which is better.

## Filling in missing defect labels with a classifier

Engineers labeled only a small share of wafers: the product with the most lots, `32x29-710`, has 38,345 first-test wafers, of which only 287 are labeled. So the original defect Pareto described less than 1% of wafers, and the lot drill-down mostly had no pattern information.

The classifier is the CNN from the [wafer-map-yield](https://github.com/litaoshu0727-del/wafer-map-yield) project (test macro-F1 0.854 ± 0.005 over 5 seeds), exported as a standalone `.pt2` file and run offline. Each wafer gets its most likely class and that class's probability. By default the dashboard only uses predictions with confidence ≥ 0.9; anything below is in effect left for manual review.

On `32x29-710`:

- 27,839 wafers (73%) have confidence ≥ 0.9, and 1,198 of them are predicted to have a defect pattern, 4 times the number of labeled wafers.
- The Pareto from predictions differs from the one from labels: Edge-Loc comes first in both (44% against 60%), but Random rises from 12% to 32%. The wafers engineers chose to label are not representative of all wafers.
- All 14 low-yield alarm lots get a main defect pattern; 10 of them are Random or Near-full, meaning large-area failure across the whole wafer rather than a local process signature.

Things to keep in mind:

- On the labeled test set, about 90% of wafers have confidence above 0.88; on all first-test wafers only 62% reach 0.9. Unlabeled wafers are distributed differently from labeled ones, so the model's accuracy on them is probably lower than on the test set.
- The model was trained on first-test records only. On retest records (where most dies were already removed at the previous step) only 13% reach confidence 0.9, so retest predictions almost always fall below the threshold and are not misused.
- Some labeled wafers were part of the model's training data, so the dashboard does not show "agreement between model and labels", which would be inflated. The model's accuracy is the one measured on the lot-grouped test set in `wafer-map-yield`.

## What each control chart is for

| Chart | Good at | Alarm delay after a 1σ drop (median) | Runs of 100 normal lots with a false alarm |
|---|---|---|---|
| Individuals + rule 1 | A single lot far off | 35 lots | 15% |
| EWMA (λ = 0.2, L = 3) | Small, sustained shifts | 9 lots | 10% |
| CUSUM (k = 0.5, h = 5) | Small, sustained shifts, with an estimate of when the shift began | 9 lots | 10% |

Simulation: normal noise, the first 100 lots as baseline, the mean drops by 1σ from lot 200 on, 200 repetitions, low-side alarms only.

Lot yield has a heavy lower tail (now and then a whole lot is scrapped), so EWMA and CUSUM clip each lot to the centre line ±3σ first: single-lot events are left to the individuals chart, and EWMA and CUSUM only answer "has the yield level moved?". The start of a CUSUM shift is the last lot at which the cumulative sum left zero, and its end is where the cumulative sum peaks: after a large shift it takes many lots for the sum to come back to zero, so using the return to zero would overstate how long the shift lasted.

## Why an individuals chart and not a p chart

Lot yield varies far more than a binomial distribution would predict (lots genuinely differ in defect level), so p-chart limits would be so narrow that almost every lot alarms. Lot yield is therefore treated as a single measurement, with σ estimated from the moving range of neighbouring lots. The moving range is used instead of the overall standard deviation because a shift or drift in yield inflates the overall standard deviation and hides the shift.

## Limitations

- Repeat and retest records are inferred from wafer maps, and the thresholds (80% shared slots, 90% containment, 20% of fails) are mine, with no official basis. The conclusions are not sensitive to them, though: in the `wafer-map-yield` project each of the four thresholds was swept, and the number of lots classed as repeat or retest changes by no more than 3% (see that project's [data quality analysis](https://github.com/litaoshu0727-del/wafer-map-yield/blob/main/docs/en/data-quality.md)).
- The wafer diameter is assumed; it only affects die area and the absolute value of D0.
- The yield models are fitted across products, which differ in process and maturity, so they are only a rough description. The product list and the product yields on that page do not exclude the near-zero-fail lots described above.
- Wafers in the same lot with different die counts end up under different products.
- Defect labels are very uneven: only 287 of the 38,345 wafers of `32x29-710` are labeled.

## Layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `yielddash/etl.py` | Build the database: detect repeat records, compress wafer maps, compute lot summaries |
| `yielddash/queries.py` | All SQL used by the dashboard |
| `yielddash/spc.py` | Control limits, Western Electric run rules, EWMA, CUSUM and shift episodes |
| `yielddash/yieldmodel.py` | Four yield models, fitting, implied D0 |
| `yielddash/classify.py` | Score every wafer with the exported CNN into the predictions table (the only part that needs PyTorch) |
| `yielddash/wm811k.py` | Read the legacy WM-811K file |
| `yielddash/synthetic.py` | Synthetic data with known events (yield step, slow drift, repeat records) |

## Data source

The project uses the public WM-811K dataset, which is not included in the repository (see [wafer-map-yield's data sources](https://github.com/litaoshu0727-del/wafer-map-yield#data-sources) for how to get it). Source:

M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets," *IEEE Transactions on Semiconductor Manufacturing*, vol. 28, no. 1, pp. 1–12, 2015. [doi:10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237)

## Related projects

- [wafer-map-yield](https://github.com/litaoshu0727-del/wafer-map-yield): wafer map yield analysis and defect pattern classification; the dashboard's CNN comes from here
- [secom-root-cause](https://github.com/litaoshu0727-del/secom-root-cause): root-cause screening on the SECOM fab data

## License

The code is under the MIT license, see [LICENSE](LICENSE). Follow the dataset's terms and cite the original paper.

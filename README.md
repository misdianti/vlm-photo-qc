# Fact-Layer Photo QC with Vision-Language Models

An experiment harness for automated quality control of field-installation photos in a telecom network rollout. A vision-language model (VLM) reports **facts** about each photo, one attribute at a time. **Plain code** applies the numeric thresholds. The model never issues a pass/fail verdict.

> This is a sanitized public copy of work done for a client. Client, department and partner names are pseudonymized, and the catalog included here is a small **synthetic sample**. No photos, ground truth files or experiment results are included.

## The problem

A network operator receives thousands of installation photos from field teams (poles, cable closures, cabinets, optical power readings, device labels). Each photo has to be checked against a written installation standard. Manual checking is slow and varies between reviewers, and the standards are mostly judgment calls ("the coil is tidy", "the cabinet door is fully closed") rather than anything a pixel rule can measure.

## Approach: facts first, verdict later

```
photo + reference photos + rules ──► VLM ──► per attribute: evidence → is_true → value
                                                      │
                                   thresholds (e.g. RSL ≥ −24 dBm) checked in code
                                                      │
                                   compared with the photo folders / ground truth
```

Each requirement (sub-category) is broken into **rules**, and each rule into **attributes**: small, observable statements such as `closure_underground` or `nilai_rsl`. For every attribute the model returns:

| Field | Meaning |
|---|---|
| `evidence` | what is visible, written **first** so the model looks before it answers |
| `is_true` | whether the attribute's description holds for this photo (true is not "good": a defect attribute like `retak` = cracked is true when the crack is visible) |
| `value` | text, number or list read from the photo, when the attribute asks for one; `null` when unsure |

Attribute types are `bool`, `text`, `number` and `json` (a list with one item per object seen, each with its own fields). The output schema is generated per sub-category from the rules and enforced through structured output.

Why split it this way:

- **Perception and policy stay separate.** The model describes what it sees. The standard (a number, a count, a combination of attributes) lives in code and in the catalog.
- **Threshold changes are free.** `reevaluate()` re-checks stored facts without a new model call. Rule or prompt changes need a new run, but never retraining.
- **Auditable.** Every answer has its evidence next to it, so wrong reads can be traced to a specific attribute description instead of an opaque verdict.

## What is in the notebook

`notebook/photo_qc_experiments.ipynb` is both the pipeline and the experiment log.

| Section | Content |
|---|---|
| 1 Setup | scope (department / category / sub-category / phase), models, storage |
| 2 Data | photo storage (object-storage bucket or a local folder with the same layout), catalog + photo manifest, photo upload helpers |
| 3 Rules and prompt | rule and attribute rendering, the fact prompt, the watermark prompt |
| 4 Output schema | Pydantic models built from attributes, JSON-schema flattening for Bedrock |
| 5 Threshold check | `>`, `>=`, `<`, `<=`, `=` checks in code, plus Pydantic validation of rule format before any paid call |
| 6 Models | Kimi K2.5 and Nova Pro on AWS Bedrock; Qwen3-VL 32B via vLLM (commented out); per-call cost estimate |
| 7 Evaluation | `run_exp`, `reevaluate`, `run_layer2`, summaries, per-attribute breakdown, ground-truth drafting and comparison, `dump_prompt` |
| 8 Change points | the ten things an experiment can change, and the rule format |
| 9 Experiments | worked examples on the sample catalog |
| Bounding box | a second call that draws evidence boxes per attribute |

### Design decisions worth knowing

- **Experiments are parameter sets, not global state.** `run_exp(prompt=..., rules=..., models=...)`: anything not passed falls back to `BASELINE`, and unknown setting names are rejected so a typo cannot silently run the wrong experiment.
- **Reference photos with leakage control.** Positive and negative examples are sent with each request; the photo being judged is excluded from its own references (`exclude_self=True`).
- **Rule format is validated before spending money.** `check_rules` catches misspelled keys, thresholds on non-number attributes and duplicate names.
- **Watermark extraction is a separate call.** Date, GPS coordinates and address printed on the photo are read in their own request, as literal text. Code converts DMS to signed decimals. In testing, bundling extraction into the facts call made one model invent coordinates and slightly disturbed the facts, so the calls were split.
- **Bounding boxes come after the facts.** The box call receives the fact answers. For one of the models, structured output returned empty boxes on most test photos, so that model is asked for plain JSON text instead.
- **Layer 2.** A second look at attributes that are often misread, sent only for photos whose first answer matches a chosen value, without showing the model its first answer.
- **Concurrency per model.** Separate semaphores, because rate limits differ widely between models.

### Evaluation

One photo is one test case, and its folder (`positive` or `negative`) is its label. With no verdict layer yet, an experiment is read through its facts:

- `attribute_breakdown`: share of `true` answers on positive vs negative photos and the gap between them. A gap near zero means the attribute does not separate the folders, or is being misread.
- Ground truth per photo × attribute × model, kept in Excel. `save_ground_truth_draft` seeds the file from one model's answers, a human corrects `actual`, and `compare_ground_truth` reports accuracy per rule × attribute × model, false positives and negatives, misread numbers, and a prediction × actual table.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example
├── notebook/
│   ├── photo_qc_experiments.ipynb
│   ├── catalog_fact_phase1.json      synthetic sample catalog
│   └── images/                       not versioned: <dept>/<main>/<sub>/{positive,negative}/<file>
└── (git-ignored) notebook/ground_truth/, notebook/hasil-experiment/
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add Bedrock credentials
jupyter lab notebook/photo_qc_experiments.ipynb
```

1. Put photos under `notebook/images/dept_b/<main-category>/<sub-category>/positive|negative/`, matching the ids in `catalog_fact_phase1.json`.
2. Run the notebook from the top. Sections 1 to 8 define the pipeline and run without any model call.
3. In section 9, set `RUN_PAID = True` to run the worked examples against Bedrock. `dump_prompt(...)` shows exactly what the model will receive before you spend anything.

To use another bucket layout, set `STORAGE = "cos"` in section 1 and provide the `COS_*` variables.

## The sample catalog

`catalog_fact_phase1.json` is invented for illustration. It shows each attribute type:

| Id | Requirement (English gloss) | Shows |
|---|---|---|
| `fr-b-01` | A closure is mounted on a pole (aerial) or in a handhole (underground) | `text` and `bool` attributes, a conditional rule |
| `fr-b-02` | An optical power meter reading is visible at 1490 nm and at least −24 dBm | `number` attributes with thresholds checked in code |
| `fr-b-03` | Every device carries a label whose code follows the numbering standard | a `json` attribute, one item per label |

Rules and prompts are in Bahasa Indonesia on purpose: the photos, labels and on-image watermarks are Indonesian, and the experiments were run in that language. In English, the fact prompt tells the model to act as a network-installation QC auditor, shows the requirement and the rules with their attributes, treats reference images as positive (meets the requirement) or negative (has a finding) examples, and asks it to report every attribute without judging pass or fail: evidence first, `is_true` for whether the description holds, `value` only for what is literally readable, `null` when unsure, and no invented findings when in doubt.

## Anonymization and what is left out

- Client, department, partner and operator names are replaced (`dept_a`, `dept_b`, `dept_c`, `PARTNER-X`), as are internal repository, cluster and path names.
- The original catalog, the photos, the ground truth workbooks, the experiment CSVs and the cell outputs are not included.
- Code comments and docstrings are still partly in Bahasa Indonesia.

## Known limitations

- **No verdict layer.** Combining attribute facts into a pass/fail per requirement is future work, so there is no end-to-end accuracy number here.
- **Ground truth drafts start from model output.** Reviewers correcting a model's draft can be anchored by it. A blind first pass on a subset would measure that.
- **Small, hand-picked samples** were used for several sub-categories, and each configuration was run once at temperature 0. There is no estimate of run-to-run variance.
- **Pricing constants** in section 6 are copied from public price lists at the time and must be re-checked before relying on cost figures.
- **Live model calls were not re-run on this public copy.** After anonymization the pipeline was exercised end to end with a stub provider on synthetic images, which covers the code paths but not model behavior.

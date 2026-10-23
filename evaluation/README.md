# Small labelled retrieval evaluation

This is a development benchmark, separate from software behaviour tests. Questions and labels were authored by the coding assistant from the source snapshot before running retrieval; they have **not** been independently reviewed by a chemist. No model training, parameter tuning, or generation API is involved.

## Run

From the repository root, with the app dependencies installed:

```bash
python evaluation/run.py
```

The first run may download the CPU embedding model. Subsequent runs can use `HF_HUB_OFFLINE=1 python evaluation/run.py` when the model is cached. The evaluation embeds the committed `corpus.json`; it does not fetch current records or replace the app's index. Results overwrite `results.json`.

## Questions and labels

`questions.json` contains 16 questions, their dataset, query SMILES, similarity threshold, expected source IDs, and supporting quotations. The runner checks every quotation against the frozen corpus before scoring. All four datasets are included: 30 androgen-receptor records and 100 records each for estrogen-receptor alpha, Tox21, and ESOL (330 total). Original metadata, source URLs, identifiers, units, and missing labels are preserved in `corpus.json`. Source datasets retain their original licenses and attribution requirements; see the main README's Sources section.

- **8 identity controls:** two exact-SMILES lookups per dataset. Both modes prioritize exact identities, so these measure identity retrieval rather than semantic quality.
- **8 text lookups:** two compound-identifier/name questions per dataset. An ethanol query SMILES, absent from these snapshots, and threshold zero make every record in the selected dataset eligible. Combined mode still mixes chemical and text ranks; Text only is the semantic baseline. These are entity lookups, not broad discovery questions.

The unmodified production `evidence.retrieve()` supplies the top five records. Each molecular record is one evidence passage. Queries use the same local BGE model as the app. Matching molecule IDs in repeated ChEMBL assays are labelled as alternative supporting records.

## Measured results

| Task | Mode | Support in top five | Hit@5 |
| --- | --- | --- | --- |
| Exact-identity controls | Combined | 8 / 8 | 100% |
| Exact-identity controls | Text only | 8 / 8 | 100% |
| Text lookups | Combined | 2 / 8 | 25% |
| Text lookups | Text only | 5 / 8 | 62.5% |

**Hit@5** is the fraction of questions for which at least one labelled supporting passage appears in the first five results. It is not precision@5 or recall over all relevant passages. Per-question retrieved IDs, first-support rank, misses, corpus/question hashes, model name, settings, and FastEmbed version are in `results.json`.

The low Combined score is a real limitation on this task: mixing an unrelated chemical query with compound-name text lookups can displace the requested evidence. These results do not demonstrate that Combined mode is worse for analogue discovery; that task has not been labelled here. No ranking changes were made to improve this score.

## Limits and next evaluation

The sample is small, source-derived, and not a held-out user benchmark. Questions contain compound names or identifiers and may be easier than natural research questions. A single question changes either subgroup score by 12.5 percentage points. Labels do not establish scientific truth, answer quality, safety, or exhaustive relevance. There are no unanswerable queries, full-paper questions, or expert-labelled analogue questions. Embedding weights are identified by model name rather than a pinned artifact hash, so runtime/model changes may change results.

Before tuning, have an independent reviewer check these labels and create a separate held-out set with paraphrases, endpoint-specific analogue questions, and absent evidence. Preserve this initial baseline and report results separately by task and dataset.

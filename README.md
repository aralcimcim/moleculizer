# Moleculizer

A molecular evidence RAG prototype connecting structure-based retrieval with semantic search and cited answers.

Choose a dataset from the selector at the top of the app:

| Dataset | Evidence |
| --- | --- |
| ChEMBL androgen receptor | Human receptor binding Ki measurements in nM |
| ChEMBL estrogen receptor alpha | Human receptor binding Ki measurements in nM |
| MoleculeNet Tox21 | Active/inactive labels for 12 screening endpoints; missing labels remain missing |
| MoleculeNet ESOL | Measured aqueous solubility in log10(mol/L), excluding the prediction column |

Each dataset has its own index and is searched separately. Binding affinity, screening labels, and solubility are different endpoints. This app retrieves existing evidence; it does not train a model or implement Chemprop predictions.

## Setup with Conda

```bash
git clone https://github.com/aralcimcim/moleculizer.git
cd moleculizer
conda env create -f environment.yml
conda activate moleculizer
python -m pip install -r requirements.txt
python build_index.py --dataset all --limit 100
python -m streamlit run app.py
```

The index builder downloads public source records and a small CPU embedding model on first use. No GPU or API key is needed for retrieval. Open the local URL printed by Streamlit; its port may differ if another app is running.

You can also build one dataset at a time:

```bash
python build_index.py --dataset tox21 --limit 1000
python build_index.py --dataset esol --limit 1000
python build_index.py --dataset estrogen-receptor --limit 100
python build_index.py --dataset androgen-receptor --limit 100
```

The limit applies separately to each dataset, from 1 to 1000 records. Rebuilding one dataset replaces only its index. Existing androgen-receptor indexes continue to work. Missing indexes show the corresponding build command in the interface. Imports use source order and are bounded snapshots, not representative samples of the complete datasets.

The SMILES field starts with a molecule actually present in your snapshot. Ask an English evidence question, select a retrieval mode, and click **Search evidence**.

After that first search, changing the similarity cutoff or **Records to display** updates results immediately, without re-embedding the question. The app shows how many evidence records pass the cutoff; select **All matches** to see them all. The cutoff removes lower-scoring analogues rather than changing the ranking. For example, if the five displayed results all score above 0.76, changing the cutoff from 0.3 to 0.7 can leave those five unchanged. Exact identities always remain. Optional Gemini summaries use at most the first five displayed records.

## Interactive workspace

The main panel is an interactive molecular landscape, not a static chart. Before searching, it displays molecular weight against calculated ring count for the dataset snapshot. After a search, it displays structural similarity against molecular weight. Select a point, or choose a source from **Source records**, to inspect its structure, measurement, descriptors, and exact source text.

- **Measurements** plots Ki on a log10(nM) axis, or the existing ESOL log10(mol/L) target without taking a second logarithm. Tox21 has no scalar measurement chart; its endpoint selector colors screening labels as active, inactive, or missing.
- **Pin to comparison** collects up to three source records in a side-by-side tray. When pins exist after a search, they define the optional Gemini summary's evidence scope.
- **Export comparison** downloads the pinned records as JSON. **Export matched evidence** downloads all matching records, not just the chart's first 120 ranked points.
- **Try a dataset molecule** fills the SMILES field with an actual source molecule. Cutoff and display-count changes continue to update existing searches without re-embedding.

Molecular weight, logP, ring count, hydrogen-bond donor count, and acceptor count are RDKit calculations, not new assay measurements or Chemprop predictions. Coordinates do not establish shared biological activity. Different assay conditions may limit direct comparisons.

The workspace is a local HTML/CSS/JavaScript/SVG Streamlit component, with no external chart library or CDN. The dark blue interface uses locally hosted STIX Two Text for academic headings; the font license is included in `fonts/OFL.txt`.

## Optional Gemini summaries

```bash
cp .env.example .env
```

Put your own key in `.env`, never in source code or Git. Change `GEMINI_MODEL` if necessary. No key is copied from your other projects.

Check the Google processing terms and billing requirements before using the API. Only clicking **Summarize evidence**, with consent selected, sends your question, SMILES, and the retrieved records to Gemini. Retrieval and molecule drawings remain local. This app is for local development; it does not include the access and rate-limit controls needed before public hosting.

## Retrieval design

1. Validate SMILES and create a canonical, stereochemistry-preserving representation using RDKit.
2. Compute radius-2, 2048-bit Morgan fingerprints and Tanimoto similarity.
3. Filter analogue candidates by a user-selected minimum similarity, retaining exact identities.
4. Embed the question and retrieve assay text by cosine similarity using `BAAI/bge-small-en-v1.5` through FastEmbed (ONNX, CPU).
5. In Combined mode, fuse structure and text ranks using reciprocal-rank fusion with constant 60. Exact canonical identities take priority in every mode.
6. Send up to five evidence records to Gemini for structured, cited synthesis if requested.

**Text only** and **Structure only** change the ranking, while retaining the same chemical filter and exact-identity priority. They are not unrestricted corpus-wide retrieval baselines. Multiple assays for the same molecule may appear; each is a separate evidence record.

## Vector store and provenance

`.local/evidence.sqlite3` holds JSON evidence records, normalized float32 text embeddings, and snapshot metadata. Retrieval computes exact cosine scores in memory. This is a small local vector store built on SQLite, **not an approximate-nearest-neighbor database**; it is suitable for the bounded 100–1000-record prototype.

Each record forms one evidence passage. ChEMBL records preserve measurement type, relation, value, units, target, assay description, molecule ID, and document ID/year. Benchmark records preserve the original data row number, compound identifier, source URL, and measured targets or endpoint labels. Tox21 uses 0 for inactive and 1 for active in its screening tasks; missing labels are not imputed. ESOL stores the measured target and its logarithmic units.

The original androgen index remains `.local/evidence.sqlite3`; other indexes are `.local/<dataset>.sqlite3`. Matching `.json` files contain human-readable snapshots. Metadata includes retrieval time and embedding model; ChEMBL records include API release information, and benchmark metadata includes a SHA-256 hash of the downloaded file. Dataset links for benchmarks point to the source CSV, with original data-row numbers (excluding the header) shown in the app.

For ChEMBL, only equality-relation Ki measurements in nM are included. Records flagged as potentially duplicated or with data-validity comments are skipped. Measurements from different assays are not pooled. Benchmark rows with unreadable SMILES or invalid targets are skipped; skipped-row counts are recorded.

## Evidence checks and limitations

Generated claims must cite retrieved activity IDs and exact quotations. The code rejects unknown source IDs and quotes absent from the corresponding source text (allowing whitespace differences). This checks citation existence, not whether a claim follows scientifically from it.

- Structural similarity does not prove shared biological activity.
- Exact identity does not normalize salts, tautomers, or protonation states.
- Binding data alone does not establish toxicity, agonism, antagonism, or a mechanism.
- Missing records in this selected snapshot do not imply lack of activity or lack of published evidence.
- The embedding model is a general English text model, not a chemistry-specialized model.
- Tox21 screening labels do not establish clinical toxicity or safety; ESOL does not provide assay conditions here.
- This version retrieves structured dataset records and source links, not full research papers.
- No clinical recommendations or calibrated predictions are provided.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use small controlled examples and mocked embeddings/model responses. They do not establish scientific accuracy or benchmark retrieval quality.

## Next steps

Create manually labeled evidence questions to compare retrieval methods. Later, integrate a compatible trained Chemprop checkpoint as a separate prediction panel. Keep predictions and experimental records distinct, and exclude held-out labels from retrieval if evaluating predictive performance.

## Container deployment

```bash
docker build -t moleculizer .
docker run --rm -p 8504:8501 moleculizer
```

Open `http://localhost:8504`. The image builds a starter index for all four datasets and includes the CPU embedding model. It runs as a non-root user with a health endpoint at `/_stcore/health`. Building requires internet access. Rebuild to refresh source snapshots. `.env`, existing local indexes, and secrets are excluded from the build context.

For optional Gemini summaries, supply credentials at runtime with `docker run --rm --env-file .env -p 8504:8501 moleculizer`. Configure HTTPS, hosting resource limits, and API usage controls on your hosting platform before unrestricted public access. This code does not provision cloud resources.

The interface uses an academic dark blue theme, an interactive chart, a source inspector, comparison trays, and responsive controls. Styles and fonts stay local.

## Sources

- [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/docs)
- [Human androgen receptor target](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL1871.json)
- [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
- [FastEmbed](https://qdrant.github.io/fastembed/)
- [MoleculeNet](https://moleculenet.org/)
- [Tox21 CSV distributed by DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz)
- [ESOL CSV distributed by DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)

Source datasets and third-party code/model weights retain their respective licenses and attribution requirements. Preserve source identifiers, URLs, and version information when sharing derived evidence.

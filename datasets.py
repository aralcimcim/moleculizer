"""Dataset presets and importers for public molecular benchmarks."""

import csv
from datetime import datetime, timezone
import gzip
import io
import math

import requests

from evidence import DB_PATH, EMBEDDING_MODEL, ROOT, canonical_smiles


DATASETS = {
    "androgen-receptor": {
        "label": "ChEMBL · Androgen receptor · Ki", "target": "CHEMBL1871",
        "target_name": "Androgen receptor",
        "scope": "Human androgen-receptor binding assays; Ki in nM with equality relation. Not Tox21 labels.",
        "question": "What binding evidence is available for this molecule at the human androgen receptor?",
    },
    "estrogen-receptor": {
        "label": "ChEMBL · Estrogen receptor alpha · Ki", "target": "CHEMBL206",
        "target_name": "Estrogen receptor",
        "scope": "Human estrogen-receptor alpha binding assays; Ki in nM with equality relation. Not Tox21 labels.",
        "question": "What binding evidence is available for this molecule at the human estrogen receptor alpha?",
    },
    "tox21": {
        "label": "MoleculeNet · Tox21 screening labels",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "scope": "MoleculeNet Tox21 benchmark labels for 12 in-vitro screening endpoints. Missing labels stay missing. Labels are not clinical toxicity or safety assessments.",
        "question": "Which Tox21 endpoint labels are reported for this molecule, and which are missing?",
    },
    "esol": {
        "label": "MoleculeNet · ESOL aqueous solubility",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "scope": "MoleculeNet ESOL benchmark measured aqueous solubility, log10(mol/L). These are dataset measurements, not predictions. Assay conditions are not supplied here.",
        "question": "What measured aqueous solubility is reported for this molecule?",
    },
}
ENDPOINTS = ("NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
             "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53")
SOLUBILITY_COLUMN = "measured log solubility in mols per litre"


def index_path(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset.")
    return DB_PATH if dataset == "androgen-receptor" else ROOT / ".local" / f"{dataset}.sqlite3"


def source_id(record):
    return str(record.get("source_id", record.get("activity_id", "")))


def make_benchmark_record(dataset, row, row_number):
    """Preserve source row numbers and labels; never impute missing targets."""
    smiles = canonical_smiles(row.get("smiles") or "")
    record = {"source_id": f"{dataset}:row:{row_number}", "dataset": dataset,
              "smiles": smiles, "source_url": DATASETS[dataset]["url"], "source_row": row_number}
    if dataset == "tox21":
        labels = {}
        for endpoint in ENDPOINTS:
            value = (row.get(endpoint) or "").strip()
            if not value or value.lower() == "nan":
                labels[endpoint] = None
            elif value in ("0", "0.0", "1", "1.0"):
                labels[endpoint] = int(float(value))
            else:
                raise ValueError(f"Invalid Tox21 label in row {row_number}.")
        if all(v is None for v in labels.values()):
            return None
        record["molecule_name"] = row.get("mol_id") or f"Tox21 row {row_number}"
        record["labels"] = labels
        details = "; ".join(f"{k}: {'Not reported' if v is None else str(v) + (' (active)' if v else ' (inactive)')}"
                            for k, v in labels.items())
        record["measurement_summary"] = f"Tox21: {sum(v is not None for v in labels.values())}/12 labels reported"
        record["text"] = (
            f"MoleculeNet Tox21 source data row {row_number} (header excluded). Molecule: {record['molecule_name']}. "
            f"SMILES: {smiles}. Endpoint labels: {details}. "
            "0 = inactive, 1 = active in the dataset screening task; missing is not inactive. "
            "Do not interpret these labels as clinical toxicity, safety, or a quantitative binding affinity."
        )
    elif dataset == "esol":
        value = float(row[SOLUBILITY_COLUMN])
        if not math.isfinite(value):
            raise ValueError(f"Invalid solubility in row {row_number}.")
        record["molecule_name"] = row.get("Compound ID") or f"ESOL row {row_number}"
        record.update(standard_type="Measured log10 aqueous solubility", standard_relation="=",
                      standard_value=str(value), standard_units="log10(mol/L)")
        record["measurement_summary"] = f"Measured aqueous solubility: {value} log10(mol/L)"
        record["text"] = (
            f"MoleculeNet ESOL source data row {row_number} (header excluded). Compound: {record['molecule_name']}. "
            f"SMILES: {smiles}. Measured aqueous solubility: {value} log10(mol/L). "
            "This is the measured dataset target, not an ESOL model prediction. Measurement conditions are not supplied here."
        )
    else:
        raise ValueError("Unknown benchmark dataset.")
    return record


def fetch_benchmark(dataset, limit):
    preset = DATASETS[dataset]
    response = requests.get(preset["url"], timeout=60)
    response.raise_for_status()
    content = gzip.decompress(response.content) if dataset == "tox21" else response.content
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    required = {"smiles", *ENDPOINTS} if dataset == "tox21" else {"smiles", SOLUBILITY_COLUMN}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("The public dataset has an unexpected column layout.")
    records, skipped = [], 0
    for row_number, row in enumerate(reader, 1):
        try:
            record = make_benchmark_record(dataset, row, row_number)
        except ValueError:
            skipped += 1
            continue
        if record is not None:
            records.append(record)
        else:
            skipped += 1
        if len(records) >= limit:
            break
    if not records:
        raise ValueError("No usable benchmark records were found.")
    import hashlib

    return records, {
        "dataset": dataset, "dataset_label": preset["label"], "source_url": preset["url"],
        "source_sha256": hashlib.sha256(response.content).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "embedding_model": EMBEDDING_MODEL,
        "record_count": len(records), "skipped_rows": skipped, "scope": preset["scope"],
    }

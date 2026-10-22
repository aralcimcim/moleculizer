"""Download a bounded ChEMBL or MoleculeNet snapshot and embed evidence locally."""

import argparse
from datetime import datetime, timezone
import json
import time

import requests

from evidence import EMBEDDING_MODEL, canonical_smiles, embed_documents, load_embedder, save_index
from datasets import DATASETS, fetch_benchmark, index_path


API = "https://www.ebi.ac.uk/chembl/api/data"


def get_json(session, resource, params=None):
    response = session.get(f"{API}/{resource}.json", params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def make_record(activity, target):
    if (not activity.get("canonical_smiles") or activity.get("standard_value") is None
            or activity.get("data_validity_comment") or activity.get("potential_duplicate")
            or activity.get("standard_units") != "nM" or activity.get("standard_relation") != "="):
        return None
    try:
        smiles = canonical_smiles(activity["canonical_smiles"])
    except ValueError:
        return None
    record = {key: activity.get(key) for key in [
        "activity_id", "molecule_chembl_id", "assay_chembl_id", "assay_description",
        "standard_type", "standard_relation", "standard_value", "standard_units",
        "document_chembl_id", "document_year", "assay_type", "target_chembl_id",
    ]}
    record.update(smiles=smiles, target_name=target["pref_name"], organism=target["organism"])
    record["source_url"] = f"{API}/activity/{record['activity_id']}.json"
    record["text"] = (
        f"ChEMBL activity {record['activity_id']}. Molecule: {record['molecule_chembl_id']}. "
        f"Target: {record['target_name']} ({record['target_chembl_id']}), {record['organism']}. "
        f"Assay: {record['assay_chembl_id']}. {record['assay_description']} "
        f"Measurement: {record['standard_type']} {record['standard_relation']} "
        f"{record['standard_value']} {record['standard_units']}. "
        f"Source document: {record['document_chembl_id']}, year {record['document_year']}."
    )
    return record


def fetch_records(limit=100, dataset="androgen-receptor"):
    preset = DATASETS[dataset]
    target_id = preset["target"]
    records = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "Moleculizer educational evidence explorer"
        target = get_json(session, f"target/{target_id}")
        if target["organism"] != "Homo sapiens" or target["pref_name"].lower() != preset["target_name"].lower():
            raise ValueError("The selected target does not match the expected human receptor.")
        status = get_json(session, "status")
        offset = 0
        while len(records) < limit:
            data = get_json(session, "activity", {
                "target_chembl_id": target_id, "assay_type": "B", "standard_type": "Ki",
                "standard_relation": "=", "standard_units": "nM", "order_by": "activity_id",
                "limit": min(100, limit), "offset": offset,
            })
            batch = data["activities"]
            if not batch:
                break
            for activity in batch:
                record = make_record(activity, target)
                if record:
                    record["dataset"] = dataset
                    records.append(record)
                    if len(records) == limit:
                        break
            offset += len(batch)
            if offset >= data["page_meta"]["total_count"]:
                break
            time.sleep(0.15)
    if not records:
        raise ValueError("No usable evidence records were found.")
    return records, {
        "dataset": dataset, "dataset_label": preset["label"],
        "target": target_id, "target_name": target["pref_name"], "organism": target["organism"],
        "chembl_status": status, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL, "record_count": len(records),
        "scope": preset["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="Number of records (1–1000)")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="androgen-receptor")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error("Choose a limit between 1 and 1000.")
    selected = list(DATASETS) if args.dataset == "all" else [args.dataset]
    embedder = None
    for dataset in selected:
        print(f"Downloading {DATASETS[dataset]['label']}...", flush=True)
        records, metadata = (fetch_records(args.limit, dataset) if "target" in DATASETS[dataset]
                             else fetch_benchmark(dataset, args.limit))
        print(f"Embedding {len(records)} records on CPU (first run downloads the embedding model)...", flush=True)
        if embedder is None:
            embedder = load_embedder()
        vectors = embed_documents(embedder, [r["text"] for r in records])
        path = index_path(dataset)
        save_index(records, vectors, metadata, path)
        snapshot = path.with_suffix(".json")
        snapshot.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2))
        print(f"Saved index: {path}", flush=True)
        print(f"Try this corpus molecule: {records[0]['smiles']}", flush=True)


if __name__ == "__main__":
    main()

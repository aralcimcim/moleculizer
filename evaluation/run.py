"""Evaluate top-five evidence retrieval on the committed, labelled snapshot.

No generation API is called. First run may download the local embedding model.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def load_cases(corpus, source_key):
    cases = json.loads((HERE / "questions.json").read_text())
    seen = set()
    for case in cases:
        if case["id"] in seen or not case["relevant_sources"]:
            raise ValueError("Duplicate case ID or missing labels")
        seen.add(case["id"])
        sources = {source_key(r): r["text"] for r in corpus(case)}
        for label in case["relevant_sources"]:
            if not label["quote"].strip() or label["quote"] not in sources.get(label["source_id"], ""):
                raise ValueError("Label does not match corpus: " + case["id"])
    return cases


def score(case, hits, mode, source_key):
    expected = {r["source_id"] for r in case["relevant_sources"]}
    ids = [source_key(r) for r in hits[:5]]
    rank = next((i for i, sid in enumerate(ids, 1) if sid in expected), None)
    return {"id": case["id"], "task": case.get("task", "trial_question"),
            "mode": mode, "question": case["question"],
            "support_in_top_5": rank is not None, "first_support_rank": rank,
            "expected_source_ids": sorted(expected), "retrieved_source_ids": ids}


def report(rows, model, corpus_counts, settings):
    groups = {}
    for row in rows:
        key = row["task"] + " / " + row["mode"]
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, values in groups.items():
        hits = sum(v["support_in_top_5"] for v in values)
        summary[key] = {"questions": len(values), "hits": hits, "hit_at_5": hits / len(values)}
        print(f"{key}: {hits}/{len(values)} supporting passage in top five ({hits/len(values):.1%})")
    result = {"metric": "Hit@5: fraction of questions with at least one labelled supporting passage in the first five results",
              "embedding_model": model, "fastembed_version": importlib.metadata.version("fastembed"),
              "corpus_sha256": hashlib.sha256((HERE / "corpus.json").read_bytes()).hexdigest(),
              "questions_sha256": hashlib.sha256((HERE / "questions.json").read_bytes()).hexdigest(),
              "corpus_counts": corpus_counts, "settings": settings,
              "summary": summary, "cases": rows}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Per-question results written to evaluation/results.json")


def main():
    from evidence import EMBEDDING_MODEL, embed_documents, load_embedder, retrieve
    from datasets import source_id

    data = json.loads((HERE / "corpus.json").read_text())
    cases = load_cases(lambda c: data[c["dataset"]]["records"], source_id)
    embedder = load_embedder()
    vectors = {ds: embed_documents(embedder, [r["text"] for r in corpus["records"]])
               for ds, corpus in data.items()}
    queries = list(embedder.query_embed([c["question"] for c in cases]))
    rows = []
    for case, query in zip(cases, queries):
        for mode in ["Combined", "Text only"]:
            hits = retrieve(data[case["dataset"]]["records"], vectors[case["dataset"]],
                            case["smiles"], query, mode=mode,
                            minimum_similarity=case["minimum_similarity"], top_k=5)
            rows.append(score(case, hits, mode, source_id))
    report(rows, EMBEDDING_MODEL, {ds: len(corpus["records"]) for ds, corpus in data.items()},
           {"top_k": 5, "scope": "Selected dataset", "identity_controls": "Exact SMILES, threshold 0.3",
            "text_lookups": "Ethanol query SMILES (absent from corpus), threshold 0.0; all records eligible",
            "ranking": "Unmodified production retrieve(); exact identities first in both modes"})


if __name__ == "__main__":
    main()

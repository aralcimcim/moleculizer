"""Molecular identity, local vector storage, and evidence retrieval."""

import json
import sqlite3
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / ".local" / "evidence.sqlite3"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonical_smiles(smiles: str) -> str:
    if not smiles.strip() or len(smiles) > 2000:
        raise ValueError("Enter a valid SMILES string of at most 2,000 characters.")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not read this SMILES string.")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def load_embedder():
    # Small ONNX model runs on CPU; no CUDA or PyTorch installation needed.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(ROOT / ".local" / "models"), threads=2)


def embed_documents(embedder, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(list(embedder.passage_embed(texts)), dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def save_index(records: list[dict], vectors: np.ndarray, metadata: dict, path: Path = DB_PATH):
    if len(records) == 0 or vectors.ndim != 2 or len(records) != len(vectors):
        raise ValueError("The index requires one embedding per evidence record.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, record TEXT, vector BLOB)")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("DELETE FROM records")
        db.execute("DELETE FROM metadata")
        db.executemany(
            "INSERT INTO records VALUES (?, ?, ?)",
            [(str(record.get("source_id", record.get("activity_id"))), json.dumps(record), np.asarray(vector, dtype=np.float32).tobytes())
             for record, vector in zip(records, vectors)],
        )
        db.executemany("INSERT INTO metadata VALUES (?, ?)",
                       [(key, json.dumps(value)) for key, value in metadata.items()])


def load_index(path: Path = DB_PATH):
    if not path.exists():
        raise ValueError("Build the evidence index first: python build_index.py")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT record, vector FROM records ORDER BY id").fetchall()
        metadata = {key: json.loads(value) for key, value in db.execute("SELECT key, value FROM metadata")}
    if not rows:
        raise ValueError("The evidence index is empty. Run python build_index.py.")
    return [json.loads(row[0]) for row in rows], np.stack([
        np.frombuffer(row[1], dtype=np.float32) for row in rows
    ]), metadata


def retrieve(records, vectors, smiles, query_vector, *, mode="Combined", minimum_similarity=0.3, top_k=5):
    """Exact canonical identities first; then eligible analogues ranked by the chosen method."""
    if mode not in ("Combined", "Structure only", "Text only"):
        raise ValueError("Unknown retrieval mode.")
    smiles = canonical_smiles(smiles)
    query_fp = FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(smiles))
    fps = [FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in records]
    structural = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, fps))
    query_vector = np.asarray(query_vector, dtype=np.float32)
    query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    semantic = vectors @ query_vector
    exact = {i for i, record in enumerate(records) if record["smiles"] == smiles}
    eligible = [i for i in range(len(records)) if i in exact or structural[i] >= minimum_similarity]
    def identity(i):
        return str(records[i].get("source_id", records[i].get("activity_id")))

    structure_order = sorted(eligible, key=lambda i: (-structural[i], identity(i)))
    text_order = sorted(eligible, key=lambda i: (-semantic[i], identity(i)))
    structure_rank = {i: rank for rank, i in enumerate(structure_order, 1)}
    text_rank = {i: rank for rank, i in enumerate(text_order, 1)}
    scores = {}
    for i in eligible:
        if mode == "Combined":
            scores[i] = 1 / (60 + structure_rank[i]) + 1 / (60 + text_rank[i])
        else:
            scores[i] = float(structural[i] if mode == "Structure only" else semantic[i])
    ranked = sorted(eligible, key=lambda i: (i not in exact, -scores[i], identity(i)))
    return [{**records[i], "match": "Exact identity" if i in exact else "Structural analogue",
             "tanimoto": float(structural[i]), "semantic_score": float(semantic[i])}
            for i in ranked[:top_k]]

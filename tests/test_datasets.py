from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from datasets import SOLUBILITY_COLUMN, fetch_benchmark, index_path, make_benchmark_record, source_id
from evidence import EMBEDDING_MODEL, load_index, retrieve, save_index
from rag import Answer, validate_citations


class DatasetTests(unittest.TestCase):
    def test_tox21_preserves_missing_and_zero_labels(self):
        record = make_benchmark_record("tox21", {"smiles": "CCO", "mol_id": "synthetic",
                                               "NR-AR": "0", "NR-ER": "1", "SR-p53": "nan"}, 7)
        self.assertEqual(record["labels"]["NR-AR"], 0)
        self.assertEqual(record["labels"]["NR-ER"], 1)
        self.assertIsNone(record["labels"]["SR-p53"])
        self.assertEqual(len(record["labels"]), 12)
        self.assertIn("NR-AR: 0 (inactive)", record["text"])
        self.assertEqual(source_id(record), "tox21:row:7")
        with self.assertRaises(ValueError):
            make_benchmark_record("tox21", {"smiles": "CCO", "NR-AR": "2"}, 1)
        self.assertIsNone(make_benchmark_record("tox21", {"smiles": "CCO"}, 1))

    def test_esol_uses_measured_target_and_log_units(self):
        record = make_benchmark_record("esol", {"smiles": "CCO", "Compound ID": "Synthetic ethanol",
                                               SOLUBILITY_COLUMN: "-1.5",
                                               "ESOL predicted log solubility in mols per litre": "-9"}, 2)
        self.assertEqual(record["standard_value"], "-1.5")
        self.assertEqual(record["standard_units"], "log10(mol/L)")
        self.assertIn("-1.5 log10(mol/L)", record["text"])
        self.assertNotIn("-9", record["text"])
        with self.assertRaises(ValueError):
            make_benchmark_record("esol", {"smiles": "CCO", SOLUBILITY_COLUMN: "nan"}, 3)

    def test_import_keeps_original_row_number_when_skipping(self):
        response = Mock()
        response.content = f"smiles,{SOLUBILITY_COLUMN}\ninvalid,-1\nCCO,-2\n".encode()
        with patch("datasets.requests.get", return_value=response):
            records, meta = fetch_benchmark("esol", 1)
        self.assertEqual(records[0]["source_row"], 2)
        self.assertEqual(meta["skipped_rows"], 1)
        self.assertEqual(meta["record_count"], 1)
        self.assertEqual(len(meta["source_sha256"]), 64)

    def test_csv_schema_is_validated(self):
        response = Mock(content=b"wrong,columns\na,b\n")
        with patch("datasets.requests.get", return_value=response), self.assertRaises(ValueError):
            fetch_benchmark("esol", 2)

    def test_generic_records_roundtrip_retrieve_and_cite(self):
        record = make_benchmark_record("esol", {"smiles": "CCO", SOLUBILITY_COLUMN: "-1.5"}, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "esol.sqlite3"
            save_index([record], np.array([[1., 0.]], dtype=np.float32), {}, path)
            records, vectors, _ = load_index(path)
            hits = retrieve(records, vectors, "CCO", [1., 0.])
        self.assertEqual(source_id(hits[0]), "esol:row:2")
        answer = Answer(claims=[{"text": "A measured solubility is present.", "citations": [
            {"source_id": "esol:row:2", "quote": "-1.5 log10(mol/L)"}]}], limitations="Synthetic data")
        validate_citations(answer, hits)

    def test_dataset_paths_are_separate(self):
        self.assertNotEqual(index_path("tox21"), index_path("esol"))
        with self.assertRaises(ValueError):
            index_path("../../etc")

    def test_dataset_switch_clears_search_and_summary(self):
        record = make_benchmark_record("tox21", {"smiles": "CCO", "NR-AR": "0"}, 1)
        metadata = {"embedding_model": EMBEDDING_MODEL, "scope": "Synthetic records", "retrieved_at": "2026-01-01"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.sqlite3"
            path.touch()
            with patch("evidence.DB_PATH", path), patch("datasets.index_path", return_value=path), patch(
                "evidence.load_index", return_value=([record], np.array([[1., 0.]]), metadata)
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
                app.selectbox(key="dataset").select("tox21").run()
                app.selectbox(key="retrieval_mode").select("Structure only").run()
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(source_id(app.session_state['hits'][0]), "tox21:row:1")
                self.assertTrue(app.dataframe)
                app.session_state['answer'] = "stale summary"
                app.selectbox(key="dataset").select("esol").run()
                self.assertFalse(app.exception)
                self.assertNotIn('hits', app.session_state)
                self.assertNotIn('answer', app.session_state)
                self.assertIn("solubility", app.text_input[1].value)


if __name__ == "__main__":
    unittest.main()

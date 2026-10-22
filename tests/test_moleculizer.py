import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from build_index import make_record
from evidence import EMBEDDING_MODEL, canonical_smiles, load_index, retrieve, save_index
from rag import Answer, summarize, validate_citations


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"activity_id": 1, "smiles": canonical_smiles("CCO"), "text": "Ethanol record."},
            {"activity_id": 2, "smiles": canonical_smiles("CCN"), "text": "Ethylamine record."},
            {"activity_id": 3, "smiles": canonical_smiles("c1ccccc1"), "text": "Benzene record."},
        ]
        self.vectors = np.eye(3, dtype=np.float32)

    def test_identity_precedes_text_similarity_in_every_mode(self):
        for mode in ("Combined", "Text only", "Structure only"):
            hits = retrieve(self.records, self.vectors, "OCC", [0, 1, 0], mode=mode,
                            minimum_similarity=0)
            self.assertEqual(hits[0]["activity_id"], 1)
            self.assertEqual(hits[0]["match"], "Exact identity")
            self.assertEqual(hits[0]["tanimoto"], 1)

    def test_no_match_is_empty(self):
        self.assertEqual(retrieve(self.records, self.vectors, "[SiH4]", [1, 0, 0],
                                  minimum_similarity=1), [])

    def test_structure_filter_removes_semantically_close_unrelated_molecule(self):
        hits = retrieve(self.records, self.vectors, "CCO", [0, 0, 1], minimum_similarity=1)
        self.assertEqual([hit["activity_id"] for hit in hits], [1])

    def test_invalid_smiles_rejected(self):
        for smiles in ("", "invalid molecule", "C" * 2001):
            with self.assertRaises(ValueError):
                canonical_smiles(smiles)

    def test_index_round_trip_and_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            save_index(self.records, self.vectors, {"release": "test"}, path)
            records, vectors, metadata = load_index(path)
            self.assertEqual(records, self.records)
            np.testing.assert_array_equal(vectors, self.vectors)
            self.assertEqual(metadata["release"], "test")
            save_index(self.records[:1], self.vectors[:1], {"release": "new"}, path)
            self.assertEqual(len(load_index(path)[0]), 1)


class EvidenceTests(unittest.TestCase):
    def test_activity_filter_preserves_units_and_source(self):
        activity = dict(activity_id=1, canonical_smiles="CCO", standard_value="42.0",
                        standard_units="nM", standard_relation="=", standard_type="Ki",
                        molecule_chembl_id="CHEMBL1", target_chembl_id="CHEMBL1871",
                        assay_chembl_id="CHEMBL2", assay_description="Binding experiment.",
                        document_chembl_id="CHEMBL3", document_year=2020, assay_type="B")
        target = {"pref_name": "Androgen receptor", "organism": "Homo sapiens"}
        record = make_record(activity, target)
        self.assertIn("Ki = 42.0 nM", record["text"])
        self.assertTrue(record["source_url"].endswith("activity/1.json"))
        for change in ({"data_validity_comment": "Invalid"}, {"potential_duplicate": True},
                       {"standard_units": "uM"}, {"standard_relation": ">"}):
            self.assertIsNone(make_record({**activity, **change}, target))

    def test_citations_reject_unknown_ids_and_invented_quotes(self):
        records = [{"activity_id": 1, "text": "Measurement: Ki = 42.0 nM."}]
        answer = Answer(claims=[{"text": "A binding measurement is reported.", "citations": [
            {"source_id": "1", "quote": "Ki = 42.0 nM"}]}], limitations="Small snapshot.")
        validate_citations(answer, records)
        answer.claims[0].citations[0].source_id = "999"
        with self.assertRaises(ValueError):
            validate_citations(answer, records)
        answer.claims[0].citations[0].source_id = "1"
        answer.claims[0].citations[0].quote = "This molecule is safe."
        with self.assertRaises(ValueError):
            validate_citations(answer, records)

    def test_summary_uses_only_supplied_sources_without_search(self):
        records = [{"activity_id": 1, "text": "Ki = 42 nM", "match": "Structural analogue", "tanimoto": 0.4}]
        answer = Answer(claims=[], limitations="Analogue evidence cannot establish query activity.")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), patch('google.genai.Client') as client:
            generate = client.return_value.__enter__.return_value.models.generate_content
            generate.return_value.text = answer.model_dump_json()
            result = summarize("What evidence exists?", "CCO", records)
            config = generate.call_args.kwargs["config"]
            self.assertFalse(config.tools)
            self.assertIn("Never transfer an analogue", config.system_instruction)
            self.assertEqual(result.claims, [])

    def test_empty_retrieval_does_not_call_model(self):
        with patch('google.genai.Client') as client:
            with self.assertRaises(ValueError):
                summarize("Any evidence?", "CCO", [])
            client.assert_not_called()


class AppTests(unittest.TestCase):
    def test_slider_updates_existing_search_without_another_click(self):
        records = [dict(activity_id=i, smiles=canonical_smiles(smiles), text="Synthetic assay record.",
                        molecule_chembl_id=f"CHEMBL{i}", standard_type="Ki", standard_relation="=",
                        standard_value="42", standard_units="nM", source_url="https://example.org",
                        assay_chembl_id="CHEMBL2", document_chembl_id="CHEMBL3")
                   for i, smiles in enumerate(("CCO", "CCN", "c1ccccc1"), 1)]
        metadata = dict(embedding_model=EMBEDDING_MODEL, target_name="Synthetic target",
                        retrieved_at="2026-01-01", scope="Synthetic test data")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            path.touch()
            with patch('evidence.DB_PATH', path), patch('evidence.load_index', return_value=(records, np.eye(3), metadata)):
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                app.selectbox(key="retrieval_mode").select("Structure only").run()
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state['hits']), 2)
                app.slider[0].set_value(0.0).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state['hits']), 3)
                app.session_state['answer'] = "stale summary"
                app.slider[0].set_value(1.0).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state['hits']), 1)
                self.assertNotIn('answer', app.session_state)
                self.assertEqual(app.session_state['hits'][0]['match'], "Exact identity")

    def test_missing_index_shows_setup(self):
        with patch('evidence.load_index', side_effect=ValueError("Build the evidence index first")):
            app = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)


if __name__ == "__main__":
    unittest.main()

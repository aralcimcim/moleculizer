"""Molescope: interactive, source-backed molecular evidence workspace."""

import hashlib
import json

import numpy as np
import streamlit as st

from datasets import DATASETS, index_path, source_id
from evidence import DB_PATH, EMBEDDING_MODEL, canonical_smiles, load_embedder, load_index, retrieve
from molecular_workbench import workbench
from rag import summarize
from ui import badges, section, setup_ui
from workbench import evidence_points, validate_pins


@st.cache_resource
def embedding_model():
    return load_embedder()


def main():
    st.set_page_config(page_title="Molescope · Evidence workspace", page_icon="🧪", layout="wide")
    setup_ui("#0080FE", "#edf2f8")
    st.markdown('''<header class="product-nav"><div class="product-brand"><span class="product-mark" aria-hidden="true">M</span>Molescope<span class="product-edition">Evidence workspace</span></div><span class="product-status">RESEARCH EDITION</span></header><div class="product-intro"><div class="eyebrow">MOLECULAR RESEARCH, WITH CONTEXT</div><h1>Evidence at molecular resolution.</h1><p>Explore structures, inspect measurements, and compare the sources behind them.</p></div>''', unsafe_allow_html=True)
    dataset = st.selectbox("Evidence dataset", list(DATASETS), format_func=lambda key: DATASETS[key]["label"], key="dataset")
    preset = DATASETS[dataset]
    path = DB_PATH if dataset == "androgen-receptor" else index_path(dataset)
    try:
        records, vectors, metadata = load_index(path)
    except ValueError as error:
        st.warning(str(error))
        st.code(f"python build_index.py --dataset {dataset} --limit 100")
        return
    except Exception:
        st.error("The local index could not be read. Rebuild it with python build_index.py.")
        return
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        st.error("The index uses a different embedding model. Rebuild it before searching.")
        return

    if st.session_state.get("active_dataset") != dataset:
        for key in ("query_vector", "hits", "answer", "summary_consent", "pinned_ids", "example_molecule", "last_workbench_event"):
            st.session_state.pop(key, None)
        st.session_state.active_dataset = dataset
        st.session_state.query_smiles = records[0]["smiles"]
        st.session_state.evidence_question = preset["question"]

    metrics = st.columns(3)
    metrics[0].metric("Source records", len(records))
    metrics[1].metric("Unique structures", len({r['smiles'] for r in records}))
    metrics[2].metric("Snapshot", metadata["retrieved_at"][:10])
    controls, explorer = st.columns([1, 2.65], gap="large")
    with controls:
        with st.container(border=True, key="search_controls"):
            st.markdown("### Structure search")
            st.caption("Search a molecule, then select points in the landscape to inspect and compare records.")
            smiles = st.text_input("Molecule SMILES", key="query_smiles", max_chars=2000)
            question = st.text_input("Evidence question (English)", key="evidence_question", max_chars=1500)
            examples = {source_id(r): r for r in records[:50]}
            def use_example():
                picked = st.session_state.example_molecule
                if picked in examples:
                    st.session_state.query_smiles = examples[picked]["smiles"]
            st.selectbox("Try a dataset molecule", ["custom", *examples], key="example_molecule",
                         format_func=lambda value: "Choose an example…" if value == "custom" else str(examples[value].get("molecule_name", examples[value].get("molecule_chembl_id", value))),
                         on_change=use_example)
            mode = st.selectbox("Retrieval mode", ["Combined", "Structure only", "Text only"], key="retrieval_mode")
            threshold = st.slider("Minimum analogue similarity", 0.0, 1.0, 0.3, 0.05,
                                  help="Tanimoto cutoff. Exact identities remain at every cutoff. After a search, this updates results immediately.")
            result_limit = st.selectbox("Records to display", [5, 10, 25, "All matches"])
            clicked = st.button("Search evidence", type="primary", width="stretch")
            st.caption("Search is local. AI summaries are optional and require separate consent.")

    search_key = hashlib.sha256(json.dumps([dataset, smiles, question, mode, path.stat().st_mtime_ns]).encode()).hexdigest()
    if st.session_state.get("search_key") != search_key:
        st.session_state.pop("query_vector", None)
        st.session_state.pop("hits", None)
    result_key = (search_key, threshold, result_limit)
    if st.session_state.get("result_key") != result_key:
        for key in ("answer", "summary_consent", "pinned_ids"):
            st.session_state.pop(key, None)
        st.session_state.result_key = result_key
    if clicked:
        for key in ("answer", "hits", "query_vector", "pinned_ids"):
            st.session_state.pop(key, None)
        try:
            normalized = canonical_smiles(smiles)
            if not question.strip():
                raise ValueError("Enter an evidence question.")
            with st.spinner("Searching the local evidence index…"):
                query_vector = np.zeros(vectors.shape[1], dtype=np.float32) if mode == "Structure only" else next(embedding_model().query_embed(question))
            st.session_state.update(query_vector=query_vector, search_key=search_key, normalized=normalized)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Search failed. Check the local embedding model and dependencies, then try again.")

    searched = "query_vector" in st.session_state
    if searched:
        matches = retrieve(records, vectors, st.session_state.normalized, st.session_state.query_vector,
                           mode=mode, minimum_similarity=threshold, top_k=len(records))
        hits = matches if result_limit == "All matches" else matches[:result_limit]
        st.session_state.hits = hits
    else:
        matches, hits = records, []

    with explorer:
        if not matches:
            st.info("No evidence records match this cutoff. Lower the similarity threshold or try a different structure.")
        else:
            points = evidence_points(matches)
            context = json.dumps([search_key, searched, threshold, len(matches)])
            payload = workbench(points, st.session_state.get("pinned_ids", []), context, preset["label"], len(matches), searched,
                                st.session_state.get("last_workbench_event"))
            if isinstance(payload, dict) and payload.get("context") == context and payload.get("event_id") != st.session_state.get("last_workbench_event"):
                event = payload.get("event_id")
                if isinstance(event, str) and len(event) <= 100:
                    try:
                        st.session_state.pinned_ids = validate_pins(payload.get("pins"), {p["id"] for p in points})
                        st.session_state.last_workbench_event = event
                        st.session_state.pop("answer", None)
                        st.session_state.pop("summary_consent", None)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        st.caption("The landscape shows up to 120 ranked source records. Select a point or use Source records to inspect it. Pin up to three records to compare.")

    with st.expander("Dataset scope, provenance, and privacy"):
        st.write(metadata["scope"])
        st.write("Only this downloaded subset is searched. Exact SMILES identity preserves stereochemistry but does not normalize salts, tautomers, or protonation. All retrieval modes apply the structural cutoff and prioritize exact identities. Combined mode uses reciprocal-rank fusion.")
        st.write("Structure descriptors and drawings are calculated locally with RDKit. Query text and SMILES are not saved as history. Only an explicit summary request sends the molecule, question, and selected public records to Gemini.")
        st.markdown("[ChEMBL](https://www.ebi.ac.uk/chembl/) · [MoleculeNet](https://moleculenet.org/) · [Google data-handling terms](https://ai.google.dev/gemini-api/terms)")

    if not searched or not hits:
        st.caption("For research exploration. Measurements, screening labels, and calculated descriptors are not predictions of toxicity or safety.")
        return

    exact = sum(h["match"] == "Exact identity" for h in matches)
    st.caption(f"{len(matches)} matching records: {exact} exact-identity records and {len(matches)-exact} analogues. Showing {len(hits)} detailed records below.")
    if not exact:
        st.warning("Only analogue evidence was retrieved. Those measurements are not measurements of your molecule.")
    with st.expander(f"Detailed evidence · {len(hits)} records"):
        for index, hit in enumerate(hits, 1):
            badges([f"SOURCE {index:02}", hit.get("molecule_name", hit.get("molecule_chembl_id", source_id(hit))), hit["match"], f"Tanimoto {hit['tanimoto']:.2f}"])
            st.text(hit["text"])
            if "labels" in hit:
                st.dataframe([{"Endpoint": k, "Dataset label": "Not reported" if v is None else f"{v} ({'active' if v else 'inactive'})"}
                              for k, v in hit["labels"].items()], hide_index=True, width="stretch")
            st.markdown(f"[Original source ↗]({hit['source_url']})")

    st.download_button("Export matched evidence (JSON)", json.dumps({"dataset": dataset, "query_smiles": st.session_state.normalized,
                       "question": question, "records": matches}, indent=2), file_name="molescope-evidence.json", mime="application/json")
    pinned = set(st.session_state.get("pinned_ids", []))
    summary_hits = [r for r in matches if source_id(r) in pinned] if pinned else hits[:5]
    section("SYNTHESIS", "Explain the selected evidence", "A source-checked summary that keeps exact matches and analogues distinct.")
    st.caption(f"Summary scope: {len(summary_hits)} {'pinned comparison' if pinned else 'top-ranked'} source records.")
    consent = st.checkbox("Send this molecule, question, and selected records to Google Gemini", key="summary_consent")
    if st.button("Summarize evidence", disabled=not consent):
        st.session_state.pop("answer", None)
        try:
            with st.spinner("Preparing a cited summary…"):
                st.session_state.answer = summarize(question, st.session_state.normalized, summary_hits)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Summary could not complete. Check model access and quota.")
    if "answer" in st.session_state:
        answer = st.session_state.answer
        lookup = {source_id(r): r for r in summary_hits}
        for claim in answer.claims:
            st.write(claim.text)
            for citation in claim.citations:
                st.text(citation.quote)
                st.markdown(f"[Source {citation.source_id}]({lookup[citation.source_id]['source_url']})")
        st.caption(answer.limitations)
        st.caption("Quotation checks verify source text, not the scientific correctness of an interpretation.")
    st.divider()
    st.caption("Molescope / Molecular evidence workspace · Research use · Review the original source and assay context.")


if __name__ == "__main__":
    main()

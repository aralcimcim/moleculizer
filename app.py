import hashlib
import json

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from evidence import DB_PATH, EMBEDDING_MODEL, canonical_smiles, load_embedder, load_index, retrieve
from rag import summarize
from datasets import DATASETS, index_path, source_id
from ui import badges, empty_state, hero, section, setup_ui


@st.cache_resource
def embedding_model():
    return load_embedder()


def main():
    st.set_page_config(page_title="Moleculizer", page_icon="🧪", layout="wide")
    setup_ui("#9a4931", "#f8ece5")
    hero("Moleculizer", "MOLECULAR RESEARCH / EVIDENCE EXPLORER",
         "From structure to source. Explore molecular measurements and screening labels, with the evidence in view.",
         ["4 public datasets", "Structure + text search", "Source-checked summaries"])
    st.caption("For research exploration. Retrieved records are evidence from a selected dataset, not predictions of toxicity or safety.")
    section("01", "Choose your evidence", "Each dataset is searched separately, with its original measurements and context.")
    dataset = st.selectbox("Dataset", list(DATASETS), format_func=lambda key: DATASETS[key]["label"], key="dataset")
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

    overview = st.columns(3)
    overview[0].metric("Evidence records", len(records))
    overview[1].metric("Unique structures", len({r['smiles'] for r in records}))
    overview[2].metric("Snapshot updated", metadata['retrieved_at'][:10])
    with st.expander("What is being searched?"):
        st.write(metadata["scope"])
        st.write(
            "Only this downloaded subset is searched. No match does not mean no evidence exists. "
            "SMILES identity preserves stereochemistry but does not normalize salts, tautomers, or protonation. "
            "All modes apply the same structural-similarity filter and prioritize exact identities. "
            "Combined mode uses reciprocal-rank fusion of chemical and text rankings."
        )
        st.write("Text embeddings run locally on CPU. Only an explicit summary request sends your SMILES, question, and retrieved records to Google Gemini. No query history is written to disk.")
        st.markdown("[ChEMBL data](https://www.ebi.ac.uk/chembl/) · [MoleculeNet](https://moleculenet.org/) · [Google data-handling terms](https://ai.google.dev/gemini-api/terms)")

    section("02", "Explore a molecule", "Start with the example molecule, or enter your own SMILES and evidence question.")
    with st.container(border=True, key="card_query"):
        smiles = st.text_input("Molecule SMILES", value=records[0]["smiles"], max_chars=2000)
        question = st.text_input("Evidence question (English)", value=preset["question"], max_chars=1500)
        left, right, last = st.columns(3)
        mode = left.selectbox("Retrieval mode", ["Combined", "Structure only", "Text only"], key="retrieval_mode")
        threshold = right.slider(
            "Minimum analogue similarity", 0.0, 1.0, 0.3, 0.05,
            help="Tanimoto cutoff: filters out analogues below this score. Exact identities remain. The highest-ranked results may stay the same until the cutoff exceeds their scores.",
        )
        result_limit = last.selectbox("Records to display", [5, 10, 25, "All matches"])
    search_key = hashlib.sha256(json.dumps([
        dataset, smiles, question, mode, path.stat().st_mtime_ns,
    ]).encode()).hexdigest()
    if st.session_state.get("search_key") != search_key:
        st.session_state.pop("query_vector", None)
        st.session_state.pop("hits", None)
    result_key = (search_key, threshold, result_limit)
    if st.session_state.get("result_key") != result_key:
        st.session_state.pop("answer", None)
        st.session_state.pop("summary_consent", None)
        st.session_state.result_key = result_key
    if st.button("Search evidence", type="primary"):
        st.session_state.pop("answer", None)
        st.session_state.pop("hits", None)
        st.session_state.pop("query_vector", None)
        try:
            normalized = canonical_smiles(smiles)
            if not question.strip():
                raise ValueError("Enter an evidence question.")
            with st.spinner("Searching the local evidence index..."):
                if mode == "Structure only":
                    query_vector = np.zeros(vectors.shape[1], dtype=np.float32)
                else:
                    query_vector = next(embedding_model().query_embed(question))
            st.session_state.update(query_vector=query_vector, search_key=search_key, normalized=normalized)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Search failed. Check dependencies and network access for the first embedding-model download.")

    if "query_vector" not in st.session_state:
        empty_state("A structure. A question. A starting point.",
                    "Select Search evidence to explore this dataset. Afterwards, the similarity cutoff and display count update your results immediately.")
        return
    matches = retrieve(records, vectors, st.session_state.normalized, st.session_state.query_vector,
                       mode=mode, minimum_similarity=threshold, top_k=len(records))
    st.session_state.hits = matches if result_limit == "All matches" else matches[:result_limit]
    hits = st.session_state["hits"]
    exact_count = sum(hit["match"] == "Exact identity" for hit in matches)
    section("03", "Inspect the evidence", f"Showing {len(hits)} of {len(matches)} matching records in this dataset.")
    with st.container(border=True, key="card_context"):
        drawing, context = st.columns([1, 1.5])
        drawing.image(Draw.MolToImage(Chem.MolFromSmiles(st.session_state["normalized"]), size=(420, 260)), caption="Query molecule")
        with context:
            st.markdown("### Your search, in context")
            badges([f"{exact_count} exact-identity records", f"{len(matches) - exact_count} analogues", f"Tanimoto ≥ {threshold:.2f}"])
            st.write(preset["label"])
            st.caption("The cutoff filters matches; it does not change their similarity scores. Exact identities remain at every cutoff. Similarity does not establish shared biological activity.")
    if not hits:
        st.warning("No records meet the similarity threshold in this snapshot. Try a corpus molecule or lower the threshold.")
        return
    if not any(hit["match"] == "Exact identity" for hit in hits):
        st.warning("Only structural analogues were retrieved. Their measurements are not measurements of your molecule.")

    for index, hit in enumerate(hits, 1):
        name = hit.get("molecule_name", hit.get("molecule_chembl_id", source_id(hit)))
        with st.container(border=True, key=f"card_source_{index}"):
            badges([f"SOURCE {index:02}", hit['match'], f"Tanimoto {hit['tanimoto']:.2f}"])
            st.markdown(f"### {name}")
            if "measurement_summary" in hit:
                st.write(hit["measurement_summary"])
            else:
                st.write(f"{hit['standard_type']} {hit['standard_relation']} {hit['standard_value']} {hit['standard_units']}")
            if "labels" in hit:
                with st.expander("Screening endpoints", expanded=True):
                    st.dataframe([{"Endpoint": endpoint, "Dataset label": "Not reported" if value is None else f"{value} ({'active' if value else 'inactive'})"}
                                  for endpoint, value in hit["labels"].items()], hide_index=True, row_height=42, width="stretch")
            with st.expander("Read source evidence"):
                st.text(hit["text"])
            st.caption(f"Evidence-text cosine similarity: {hit['semantic_score']:.3f}" if mode != "Structure only" else "Ranked by molecular structure only.")
            if "activity_id" in hit:
                st.markdown(f"[Activity source]({hit['source_url']}) · [Assay](https://www.ebi.ac.uk/chembl/api/data/assay/{hit['assay_chembl_id']}.json) · [Document](https://www.ebi.ac.uk/chembl/api/data/document/{hit['document_chembl_id']}.json)")
            else:
                st.caption(f"Source ID: {source_id(hit)} · data row {hit['source_row']} (header excluded)")
                st.markdown(f"[Source dataset download]({hit['source_url']})")

    section("04", "Connect the findings", "An optional summary that keeps exact matches and analogue evidence distinct.")
    summary_hits = hits[:5]
    st.caption(f"The optional summary uses the first {len(summary_hits)} displayed records.")
    consent = st.checkbox("Send this molecule, question, and retrieved evidence to Google Gemini.", key="summary_consent")
    if st.button("Summarize evidence", disabled=not consent):
        st.session_state.pop("answer", None)
        try:
            with st.spinner("Summarizing retrieved evidence..."):
                st.session_state["answer"] = summarize(question, st.session_state["normalized"], summary_hits)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Summary failed. Check Gemini configuration, model access, and quota.")
    if "answer" in st.session_state:
        answer = st.session_state["answer"]
        by_id = {source_id(hit): hit for hit in summary_hits}
        for claim in answer.claims:
            st.write(claim.text)
            for citation in claim.citations:
                st.text(citation.quote)
                st.markdown(f"[Source {citation.source_id}]({by_id[citation.source_id]['source_url']})")
        st.write("Limitations:", answer.limitations)
        st.caption("Citation matching checks text presence, not scientific correctness. Review the assay context and source records.")
    st.divider()
    st.caption("Moleculizer · Public molecular evidence · Measurements and screening labels retain their original context.")


if __name__ == "__main__":
    main()

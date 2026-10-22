"""Public evidence points and validated comparison selections."""

import base64
from functools import lru_cache
import math

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski

from datasets import source_id


@lru_cache(maxsize=2048)
def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecular structure.")
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(320, 200)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    image = base64.b64encode(drawer.GetDrawingText().encode()).decode()
    return {"weight": round(Descriptors.MolWt(mol), 2), "rings": Lipinski.RingCount(mol),
            "logp": round(Crippen.MolLogP(mol), 2), "donors": Lipinski.NumHDonors(mol),
            "acceptors": Lipinski.NumHAcceptors(mol), "image": "data:image/svg+xml;base64," + image}


def evidence_points(records, limit=120):
    points = []
    for record in records[:limit]:
        number = None
        try:
            value = float(record.get("standard_value"))
            if math.isfinite(value):
                number = value
        except (ValueError, TypeError):
            pass
        measurement = record.get("measurement_summary") or " ".join(str(record.get(k, "")) for k in (
            "standard_type", "standard_relation", "standard_value", "standard_units"))
        points.append({"id": source_id(record), "name": record.get("molecule_name", record.get("molecule_chembl_id", source_id(record))),
                       "smiles": record["smiles"], "measurement": measurement, "value": number,
                       "measurement_type": record.get("standard_type", "Screening labels"),
                       "units": record.get("standard_units", ""), "labels": record.get("labels", {}),
                       "match": record.get("match", "Dataset record"), "similarity": record.get("tanimoto"),
                       "text": record["text"], "url": record["source_url"], **describe(record["smiles"])})
    return points


def validate_pins(value, allowed_ids):
    if not isinstance(value, list) or len(value) > 3 or any(not isinstance(i, str) or i not in allowed_ids for i in value):
        raise ValueError("Choose up to three visible source records to compare.")
    if len(set(value)) != len(value):
        raise ValueError("A source may only be pinned once.")
    return value

from pathlib import Path

import streamlit.components.v1 as components


_workbench = components.declare_component("molecular_workbench", path=str(Path(__file__).with_name("frontend")))


def workbench(points, pinned, context, dataset, total, query_mode, ack=None):
    return _workbench(points=points, pinned=pinned, context=context, dataset=dataset, total=total,
                      query_mode=query_mode, ack=ack, default=None, key="molecular_workbench")

"""Display existing AcroForm appearances without changing or validating a signed PDF."""
from __future__ import annotations

from typing import Any


def initialize_static_forms(pdf: Any) -> dict[str, Any]:
    """Call before obtaining page handles/length. XFA and unknown types fail closed.

    No document JavaScript/open-action APIs or user-interaction callbacks are invoked.
    This remains a native PDF renderer, not an operating-system security sandbox.
    """
    from pypdfium2 import raw

    form_type = raw.FPDF_GetFormType(pdf.raw)
    if form_type not in (raw.FORMTYPE_NONE, raw.FORMTYPE_ACRO_FORM):
        raise ValueError("Unsupported XFA/unknown form type; reviewed conversion is required")
    if form_type == raw.FORMTYPE_ACRO_FORM:
        config = raw.FPDF_FORMFILLINFO(version=2, xfa_disabled=True)
        pdf.init_forms(config=config)
        if not pdf.formenv:
            raise ValueError("AcroForm environment could not be initialized")
    return {
        "type": "acroform" if form_type == raw.FORMTYPE_ACRO_FORM else "none",
        "render_existing_appearances": True,
        "draw_annotations": True,
        "document_actions_invoked": False,
        "source_rewritten": False,
        "digital_signature_validation": "not_performed",
        "requires_visual_review": True,
    }

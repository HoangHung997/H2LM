from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pypdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    RectangleObject,
    TextStringObject,
)
from pypdfium2 import raw

from h2lm.scans.config import ScanConfig
from h2lm.scans.form_policy import initialize_static_forms
from h2lm.scans.ingest import prepare_pdf


def form_fixture(path: Path) -> None:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(240, 320)
    appearance = DecodedStreamObject()
    appearance.set_data(b"1 0 0 rg 0 0 80 40 re f")
    appearance.update({NameObject("/Type"): NameObject("/XObject"),
                       NameObject("/Subtype"): NameObject("/Form"),
                       NameObject("/BBox"): RectangleObject((0, 0, 80, 40)),
                       NameObject("/Resources"): DictionaryObject()})
    field = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Widget"),
        NameObject("/FT"): NameObject("/Sig"),
        NameObject("/T"): TextStringObject("AppearanceFixtureNotARealSignature"),
        NameObject("/F"): NumberObject(4),
        NameObject("/Rect"): RectangleObject((20, 250, 100, 290)),
        NameObject("/AP"): DictionaryObject({NameObject("/N"): writer._add_object(appearance)}),
    })
    field_ref = writer._add_object(field)
    page[NameObject("/Annots")] = ArrayObject([field_ref])
    writer._root_object[NameObject("/AcroForm")] = DictionaryObject({
        NameObject("/Fields"): ArrayObject([field_ref]),
        NameObject("/NeedAppearances"): BooleanObject(False),
    })
    writer.write(path)
    writer.close()


def test_static_signature_appearance_is_rendered_without_rewriting(tmp_path):
    source = tmp_path / "signed-fixture.pdf"
    form_fixture(source)
    before = source.read_bytes()
    config = ScanConfig(dpi=72, tile_size=128, overlap=20, preview_size=128)
    out = tmp_path / "out"
    report = prepare_pdf(source, out, config)
    assert source.read_bytes() == before == (out / "source.pdf").read_bytes()
    assert report["form_policy"]["type"] == "acroform"
    assert report["form_policy"]["digital_signature_validation"] == "not_performed"
    assert report["form_policy"]["document_actions_invoked"] is False
    with Image.open(out / report["pages"][0]["page"]["path"]) as image:
        red, green, blue = image.getpixel((50, 50))
        assert red > 200 and green < 50 and blue < 50
        with (
            pdfium.PdfDocument(source) as pdf,
            closing(pdf[0]) as page,
            closing(page.render(scale=1, may_draw_forms=False,
                                draw_annots=False, rev_byteorder=True)) as bitmap,
            bitmap.to_pil().convert("RGB") as omitted,
        ):
            assert ImageChops.difference(image, omitted).getbbox() is not None


@pytest.mark.parametrize("form_type", [-1, 2, 3, 999])
def test_unknown_and_xfa_remain_rejected(monkeypatch, form_type):
    monkeypatch.setattr(raw, "FPDF_GetFormType", lambda handle: form_type)
    with pytest.raises(ValueError, match="Unsupported"):
        initialize_static_forms(SimpleNamespace(raw=None))


def test_none_does_not_initialize_form_environment(monkeypatch):
    monkeypatch.setattr(raw, "FPDF_GetFormType", lambda handle: raw.FORMTYPE_NONE)
    assert initialize_static_forms(SimpleNamespace(raw=None))["type"] == "none"


def test_init_failure_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(raw, "FPDF_GetFormType", lambda handle: raw.FORMTYPE_ACRO_FORM)
    pdf = SimpleNamespace(raw=None, formenv=None, init_forms=lambda config: None)
    with pytest.raises(ValueError, match="could not"):
        initialize_static_forms(pdf)


def test_no_javascript_platform_or_callbacks_registered(monkeypatch):
    captured = {}
    monkeypatch.setattr(raw, "FPDF_GetFormType", lambda handle: raw.FORMTYPE_ACRO_FORM)
    def init(config):
        captured["config"] = config
    pdf = SimpleNamespace(raw=None, formenv=True, init_forms=init)
    initialize_static_forms(pdf)
    config = captured["config"]
    assert not config.m_pJsPlatform
    assert config.xfa_disabled
    assert not config.FFI_DoURIAction
    assert not config.FFI_DoGoToAction

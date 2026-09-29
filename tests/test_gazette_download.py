import pytest

from h2lm.tokenization.public_seed import discover_pdf


def test_observed_gazette_stream_download_format():
    url = ("https://g7.cdnchinhphu.vn/api/download/stream?Url=public-token"
           "&file_name=2023_869+%2b+870_22-2023-QH15.pdf")
    html = ('<a href="' + url.replace("&", "&amp;") + '">file</a>').encode()
    assert discover_pdf(html, "https://congbao.chinhphu.vn/doc", "22/2023/QH15") == url
    with pytest.raises(ValueError):
        discover_pdf(html.replace(b".pdf", b".doc"), "https://congbao.chinhphu.vn/doc", "22/2023/QH15")
    with pytest.raises(ValueError):
        discover_pdf(html.replace(b"/api/download/stream", b"/another/endpoint"),
                     "https://congbao.chinhphu.vn/doc", "22/2023/QH15")

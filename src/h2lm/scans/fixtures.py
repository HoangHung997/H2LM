"""Synthetic rescan stress fixtures. They are not genuine scanner acceptance evidence."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .ingest import write_json

LINES = [
    "MẪU KIỂM THỬ H2LM — KHÔNG PHẢI VĂN BẢN PHÁP LUẬT",
    "Điều 1. Nội dung giả lập để kiểm tra ảnh scan",
    "Khoản 2, điểm đ: giữ nguyên dấu tiếng Việt và dấu chấm phẩy.",
    "Phân biệt: 0/O, 1/l, 5/S; một chữ mờ không được tự đoán.",
    "Số ví dụ: XX/20XX/NĐ-CP; 1.250,50; -2,5; 20/09/20XX.",
    "Bảng thử: Hạng mục | Đơn vị | Số lượng | Ghi chú",
    "Mục giả lập A       bộ          05         chưa nghiệm thu",
    "Mục giả lập B       m²         12,50       chỉ là dữ liệu thử",
    "Chú thích nhỏ: phải nhìn rõ đ, ă, â, ê, ô, ơ, ư và số điều.",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [Path("C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("Demo needs a local Vietnamese-capable Arial/DejaVu Sans font; no font bundled")


def make_scan_fixture(output: str | Path) -> Path:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    base = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(base)
    for n, line in enumerate(LINES):
        draw.text((70, 100 + n * 90), line, font=_font(20 if n == 8 else 27), fill="black")
    draw.rectangle((65, 500, 1170, 790), outline="gray", width=2)
    # Simulated stamp-like artifact, not a genuine seal/signature.
    draw.ellipse((800, 1100, 1030, 1300), outline=(120, 50, 50), width=6)
    draw.text((815, 1180), "MẪU THỬ", font=_font(27), fill=(120, 50, 50))
    clean = base.copy()
    low = base.resize((620, 877), Image.Resampling.BILINEAR)
    blur = low.resize(base.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(0.8))
    low.close()
    skew = base.rotate(2.2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    encoded = io.BytesIO()
    base.save(encoded, "JPEG", quality=28)
    with Image.open(io.BytesIO(encoded.getvalue())) as opened:
        jpeg = opened.convert("RGB")
    shade = Image.new("RGB", base.size)
    pixels = ImageDraw.Draw(shade)
    for x in range(base.width):
        value = int(180 + 75 * x / (base.width - 1))
        pixels.line((x, 0, x, base.height), fill=(value, value, value))
    shadow = ImageChops.multiply(jpeg, shade)
    jpeg.close()
    shade.close()
    faint = ImageEnhance.Contrast(base).enhance(0.06)
    images = [clean, blur, skew, shadow,
              base.transpose(Image.Transpose.ROTATE_90),
              base.transpose(Image.Transpose.ROTATE_180), faint,
              Image.new("RGB", base.size, "white")]
    profiles = ["clean", "low_resolution_blur", "skew_2.2_degrees", "jpeg_shadow",
                "quarter_turn", "upside_down", "faint", "blank"]
    try:
        pdf_path = out / "synthetic-rescans.pdf"
        images[0].save(pdf_path, "PDF", save_all=True, append_images=images[1:], resolution=150)
        write_json(out / "fixture_metadata.json", {
            "origin": "synthetic_fixture", "physical_scanner": False,
            "document_id": "synthetic-scan-v1", "family_id": "synthetic-scan-v1",
            "split": "unassigned", "profiles": profiles,
            "source_transcript_before_degradation": "\n".join(LINES),
            "labels_after_degradation": "unverified; do not assume all reference text remains readable",
            "eligible_for_supervised_training": False,
        })
        return pdf_path
    finally:
        for image in images:
            image.close()
        base.close()

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ScanConfig
from .ingest import prepare_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Local scan images/tiles, NOT OCR or neural inference")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/scan/scan_v1.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--page-count", type=int, default=5)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--document-id")
    parser.add_argument("--family-id")
    parser.add_argument("--split", choices=("unassigned", "train", "validation", "test"),
                        default="unassigned")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo and args.pdf:
        parser.error("Choose --demo OR --pdf")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output = args.output or Path("artifacts/scan-input") / stamp
    try:
        if args.demo:
            from .fixtures import make_scan_fixture
            source = make_scan_fixture(output.with_name(output.name + "-fixture"))
        elif args.pdf:
            source = args.pdf
        else:
            import tkinter as tk
            from tkinter.filedialog import askopenfilename
            root = tk.Tk()
            root.withdraw()
            try:
                selected = askopenfilename(title="Chọn PDF scan — xử lý local, không upload",
                                           filetypes=[("PDF", "*.pdf")])
            finally:
                root.destroy()
            if not selected:
                print("Cancelled. No document processed.")
                return 0
            source = Path(selected)
        report = prepare_pdf(source, output, ScanConfig.load(args.config),
                             start_page=args.start_page, page_count=args.page_count,
                             rotation=args.rotate, document_id=args.document_id,
                             family_id=args.family_id, split=args.split)
        print(f"Prepared {len(report['pages'])}/{report['total_document_pages']} pages; "
              f"{sum(len(p['tiles']) for p in report['pages'])} image tiles.")
        print("No OCR, no model inference, no document upload. Labels remain unverified.")
        print(f"Review local images: {(output / 'review.html').resolve()}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; partial output is not a completed dataset.", file=sys.stderr)
        return 130
    except Exception:
        logging.exception("Scan preparation failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

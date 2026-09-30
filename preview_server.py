#!/usr/bin/env python3
"""Local visual dashboard for the Netra-One collection and labeling pipeline."""
from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "clean"
REVIEW = ROOT / "dataset" / "review_smoke"
CLASSES = ("indian_ambulance", "indian_police_vehicle", "indian_fire_tender")


def image_for(class_name: str) -> Path | None:
    return next(iter(sorted((DATASET / class_name).glob("*.jpg"))), None)


def count_for(class_name: str) -> int:
    return len(list((DATASET / class_name).glob("*.jpg")))


def send_file(handler: BaseHTTPRequestHandler, file_path: Path) -> None:
    if not file_path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime_type)
    handler.send_header("Content-Length", str(file_path.stat().st_size))
    handler.end_headers()
    handler.wfile.write(file_path.read_bytes())


def dashboard_html() -> bytes:
    total = sum(count_for(name) for name in CLASSES)
    cards = "".join(
        f'<article><img src="/sample/{name}" alt="{name}"><h3>{name.replace("_", " ")}</h3>'
        f'<p>{count_for(name)} cleaned candidates</p></article>'
        for name in CLASSES
    )
    review = next(iter(sorted(REVIEW.glob("*.jpg"))), None)
    review_src = "/review" if review else "/sample/indian_ambulance"
    label_state = "Grounding DINO auto-label proposal" if review else "Sample image; auto-label preview not generated yet"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netra-One Preview</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#07121f;color:#e9f4ff;font-family:Arial,sans-serif}}
header{{padding:34px max(24px,calc((100% - 1120px)/2));background:linear-gradient(120deg,#0f2b45,#0c6035)}}
h1{{margin:0;font-size:clamp(28px,5vw,48px)}} header p{{margin:10px 0 0;color:#b8d7e9}}
main{{max-width:1120px;margin:28px auto;padding:0 24px}} .status{{display:inline-block;padding:8px 12px;border-radius:99px;background:#134f30;color:#aaf5bd;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin:22px 0 34px}} article,.panel{{background:#0d2133;border:1px solid #23445d;border-radius:15px;overflow:hidden}}
article img{{display:block;width:100%;height:155px;object-fit:cover}} article h3,article p{{margin:12px 14px}} article h3{{text-transform:capitalize}} article p{{color:#a7c3d7}}
.panel{{padding:18px}} .preview{{width:100%;max-height:570px;object-fit:contain;background:#02070c;border-radius:9px;margin-top:12px}}
.note{{color:#b8d7e9;line-height:1.55}} code{{color:#aaf5bd}}
</style></head><body><header><h1>Netra-One</h1><p>Indian Emergency Vehicle Detection · Local pipeline preview</p></header>
<main><span class="status">COLLECTION COMPLETE · {total} CANDIDATES</span><h2>Collected classes</h2><section class="grid">{cards}</section>
<section class="panel"><h2>Current detection preview</h2><p class="note">{label_state}. Green boxes indicate detected vehicle regions.</p>
<img class="preview" src="{review_src}" alt="Emergency vehicle detection preview"><p class="note">Train YOLOv8n on a T4 GPU to replace this proposal view with final <code>best.pt</code> inference.</p></section>
</main></body></html>"""
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            body = dashboard_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/review":
            review = next(iter(sorted(REVIEW.glob("*.jpg"))), None)
            if review is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                send_file(self, review)
        elif route.startswith("/sample/"):
            class_name = route.removeprefix("/sample/")
            if class_name not in CLASSES:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                image = image_for(class_name)
                if image is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    send_file(self, image)
        elif route == "/health":
            payload = {"classes": {name: count_for(name) for name in CLASSES}, "weights_available": (ROOT / "weights" / "best.pt").exists()}
            body = json.dumps(payload).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Netra-One local preview.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Netra-One preview: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

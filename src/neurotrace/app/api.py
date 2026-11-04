"""Local server for the visualizer: the exported JSON as an API plus the built site.

    make app            # builds web/ and serves http://localhost:8000

The public site on GitHub Pages reads the same JSON files statically; this server is for local use
(e.g. after re-running ``scripts/export_explanations.py`` on new recordings).
Research prototype - not for clinical use.
"""

from __future__ import annotations

import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from neurotrace import DISCLAIMER
from neurotrace.utils import ROOT

DATA = ROOT / "web" / "public" / "data"
DIST = ROOT / "web" / "dist"

app = FastAPI(title="NEUROTRACE", description=DISCLAIMER, version="0.1.0")


@app.middleware("http")
async def disclaimer_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Disclaimer"] = DISCLAIMER
    return response


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "disclaimer": DISCLAIMER, "data": (DATA / "index.json").exists()}


@app.get("/api/index")
def index() -> JSONResponse:
    path = DATA / "index.json"
    if not path.exists():
        raise HTTPException(404, "no exported data; run scripts/export_explanations.py")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/segments/{seg_id}")
def segment(seg_id: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", seg_id):
        raise HTTPException(400, "bad segment id")
    path = DATA / "seg" / f"{seg_id}.json"
    if not path.exists():
        raise HTTPException(404, f"unknown segment {seg_id}")
    return FileResponse(path, media_type="application/json")


# the built site expects to live under /neurotrace/ (its GitHub Pages path)
if DIST.exists():
    app.mount("/neurotrace", StaticFiles(directory=DIST, html=True), name="site")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(DIST / "index.html")

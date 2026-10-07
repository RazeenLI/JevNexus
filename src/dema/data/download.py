"""Download and extract the raw Magneto benchmark collection into ``data/raw``.

Layout produced (relative to ``raw_data``)::

    downloads/                     original archives, never modified
    gdc/data/ground-truth/*.csv    from gdc-sm-data.zip
    gdc/data/target-tables/...     from gdc-sm-data.zip
    gdc/data/source-tables/*.csv   rebuilt from the papers' supplementary files
    gdc/downloads/*.xlsx           supplementary files as downloaded
    valentine/Valentine-datasets/  from Valentine-datasets.zip

The GDC-SM release does not redistribute the study tables. They are rebuilt with
the official GDC-SM procedure (``gdc_download.py`` shipped with the Zenodo
record, re-implemented below without modification of its logic, including its
four documented fixes). The raw step is idempotent: files that already exist are
kept as they are.

Usage::

    python -m dema.data.download [--only gdc|valentine]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

from ..utils.config import load_config
from ..utils.logging import get_console_logger

log = get_console_logger("dema.download")

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) JevNexus-benchmark-downloader"


def _download(url: str, dest: Path, timeout: int = 600, attempts: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("exists, skipping download: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        log.info("downloading %s -> %s (attempt %d)", url, dest, attempt)
        try:
            with requests.get(
                url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
            ) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException:
            if tmp.exists():
                tmp.unlink()
            if attempt == attempts:
                raise


def _extract(archive: Path, dest: Path, marker_name: str) -> None:
    marker = dest / marker_name
    if marker.exists():
        log.info("already extracted: %s", archive.name)
        return
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"not a valid zip archive: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in {archive}")
        for member in zf.namelist():
            # Skip macOS metadata that is not part of the benchmark.
            if member.startswith("__MACOSX/") or Path(member).name == ".DS_Store":
                continue
            zf.extract(member, dest)
    marker.write_text(archive.name + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- GDC
def build_gdc_sources(gdc_dir: Path) -> None:
    """Re-implementation of GDC-SM ``gdc_download.py`` (official procedure)."""
    import pandas as pd

    papers_info_path = gdc_dir / "papers_info.json"
    papers_info = json.loads(papers_info_path.read_text(encoding="utf-8"))
    download_dir = gdc_dir / "downloads"
    out_dir = gdc_dir / "data" / "source-tables"
    download_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for paper, info in papers_info.items():
        out_file = out_dir / paper
        if out_file.exists():
            log.info("GDC source exists: %s", paper)
            continue
        url = info.get("Dataset URL")
        if not url:
            failures.append(paper)
            continue
        xlsx = download_dir / Path(url).name
        try:
            _download(url, xlsx, timeout=120)
        except requests.RequestException as exc:
            log.error("failed to download %s: %s", url, exc)
            failures.append(paper)
            continue
        if not xlsx.name.endswith(".xlsx"):
            failures.append(paper)
            continue
        df = pd.read_excel(xlsx, sheet_name=info["Sheet Name"])
        # --- official fixes from gdc_download.py (verbatim logic) ---
        if paper == "Vasaikar.csv":
            df = df.iloc[1:-1]
        if paper == "McDermott.csv":
            df.rename(
                columns={
                    "Age in Months at Time of Tissue Procurement \t": "Age in Months at Time of Tissue Procurement",
                },
                inplace=True,
            )
        if paper == "Clark.csv":
            if "Ethnicity_Self_Identify" not in df.columns:
                raise ValueError(
                    f"Expected column Ethnicity_Self_Identify but not found in table from paper {paper}"
                )
            df["Ethnicity_Self_Identify"] = df["Ethnicity_Self_Identify"].replace("Whte", "White")
        if paper == "Wang.csv":
            col = "secondhand_smoke_exposure"
            df[col] = df[col].replace("No or minimal exposure to secondhand smoke", "No")
            df[col] = df[col].replace("Exposed in childhood houshold", "Yes")
            df[col] = df[col].replace("Exposed in current household", "Yes")
        df.to_csv(out_file, index=False)
        log.info("created GDC source table %s (%d rows, %d cols)", paper, len(df), df.shape[1])
    if failures:
        raise RuntimeError(f"could not build GDC source tables for: {failures}")


def prepare_gdc(config) -> None:
    spec = config.paths["downloads"]["gdc"]["files"]
    downloads = config.raw_dir / "downloads"
    for name, url in spec.items():
        _download(url, downloads / name)
    gdc_dir = config.raw_dir / "gdc"
    _extract(downloads / "gdc-sm-data.zip", gdc_dir, ".extracted")
    for name in ("papers_info.json", "gdc_download.py"):
        if not (gdc_dir / name).exists():
            shutil.copy2(downloads / name, gdc_dir / name)
    build_gdc_sources(gdc_dir)


def prepare_valentine(config) -> None:
    spec = config.paths["downloads"]["valentine"]["files"]
    downloads = config.raw_dir / "downloads"
    for name, url in spec.items():
        _download(url, downloads / name, timeout=3600)
    _extract(downloads / "Valentine-datasets.zip", config.raw_dir / "valentine", ".extracted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["gdc", "valentine"], default=None)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)
    config = load_config(config_dir=args.config_dir)
    if args.only in (None, "gdc"):
        prepare_gdc(config)
    if args.only in (None, "valentine"):
        prepare_valentine(config)
    log.info("raw benchmark ready under %s", config.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

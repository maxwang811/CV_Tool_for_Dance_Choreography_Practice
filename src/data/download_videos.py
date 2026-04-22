"""Phase 2.4: filtered downloader.

Downloads every manifest row matching the configured ``download.filter``
(e.g. ``cameras: ['01']`` and ``situations: ['FM', 'BM']``), updates the
manifest with local paths, and preserves resumability via ``.part`` files.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm import tqdm

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table


def _download_one(url: str, out_path: Path, timeout: float, max_retries: int) -> tuple[bool, str | None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return True, None
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False, last_err


def _select_indices(
    rows: List[dict],
    cameras: Iterable[str] | None,
    situations: Iterable[str] | None,
) -> List[int]:
    """Return manifest indices matching the camera/situation filter.

    An empty/None list for a dimension means "no constraint on that dimension".
    Rows with ``parse_ok`` not true are skipped because their metadata is
    unreliable.
    """
    cams = set(cameras or [])
    sits = set(situations or [])
    out: List[int] = []
    for i, r in enumerate(rows):
        if not r.get("parse_ok"):
            continue
        if cams and r.get("camera") not in cams:
            continue
        if sits and r.get("situation") not in sits:
            continue
        out.append(i)
    return out


def download_pilot(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    manifest_path = Path(cfg["manifest_out"])
    rows = read_table(manifest_path)

    d = cfg.get("download", {})
    out_dir = Path(d.get("out_dir", "data/raw_videos"))
    timeout = float(d.get("timeout_sec", 60))
    max_retries = int(d.get("max_retries", 3))
    f = d.get("filter", {}) or {}

    indices = _select_indices(rows, f.get("cameras"), f.get("situations"))
    print(
        f"Selected {len(indices)} / {len(rows)} rows matching filter "
        f"cameras={f.get('cameras')} situations={f.get('situations')}"
    )

    for row in rows:
        row.setdefault("local_path", None)
        row.setdefault("download_status", None)
        row.setdefault("download_error", None)

    for idx in tqdm(indices, desc="download"):
        row = rows[idx]
        filename = row["filename"]
        out_path = out_dir / filename
        if out_path.exists():
            row["local_path"] = str(out_path)
            row["download_status"] = "already_present"
            continue
        ok, err = _download_one(row["url"], out_path, timeout=timeout, max_retries=max_retries)
        row["local_path"] = str(out_path) if ok else None
        row["download_status"] = "ok" if ok else "failed"
        row["download_error"] = err

    write_table(manifest_path, rows)
    return manifest_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Download manifest videos matching the configured filter.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    out = download_pilot(args.config)
    print(f"Download finished. Manifest updated -> {out}")


if __name__ == "__main__":
    _main()

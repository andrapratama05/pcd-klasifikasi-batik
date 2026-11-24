"""Download, extract, and verify the Batik Banyumasan dataset (idempotent).

Usage:
    uv run python -m src.download_data            # download if needed + extract + verify
    uv run python -m src.download_data --force    # redo extraction from scratch

Flow:
    1. data/raw/batik_banyumasan/ exists and passes verification -> done.
    2. ZIP already in data/zip/ -> extract it (no re-download).
    3. Otherwise download DATASET_URL to data/zip/ (via a .part file).
Only the standard library and tqdm are used.
"""

import argparse
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src import config

JUNK_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"}
FILENAME_RE = re.compile(config.FILENAME_PATTERN)


# --- Verification ----------------------------------------------------------
def verify(raw_dir: Path = config.RAW_DIR, verbose: bool = True) -> bool:
    """Check class folders and file counts. Return True if everything matches."""
    if not raw_dir.is_dir():
        if verbose:
            print(f"[verify] Folder belum ada: {raw_dir}")
        return False

    class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    names = [p.name for p in class_dirs]
    ok = True
    missing = sorted(set(config.CLASSES) - set(names))
    extra = sorted(set(names) - set(config.CLASSES))
    if missing or extra:
        ok = False
        if verbose:
            print(f"[verify] PERINGATAN: folder kelas tidak sesuai. Hilang={missing} Lebih={extra}")

    per_class, originals, unmatched = Counter(), Counter(), 0
    for d in class_dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() == ".jpg":
                per_class[d.name] += 1
                m = FILENAME_RE.match(f.name)
                if m is None:
                    unmatched += 1
                elif m.group("aug") is None:
                    originals[d.name] += 1

    total, n_orig = sum(per_class.values()), sum(originals.values())
    if verbose:
        print(f"[verify] {len(class_dirs)} folder kelas, {total} file .jpg, {n_orig} foto asli")
        for name in names:
            print(f"    {name:<16} {per_class[name]:>4} file  ({originals[name]} asli)")
        if unmatched:
            print(f"[verify] PERINGATAN: {unmatched} file tidak cocok pola nama")

    if total != config.EXPECTED_TOTAL or n_orig != config.EXPECTED_ORIGINALS:
        ok = False
        if verbose:
            print(
                f"[verify] PERINGATAN: diharapkan {config.EXPECTED_TOTAL} citra dan "
                f"{config.EXPECTED_ORIGINALS} foto asli, ditemukan {total} dan {n_orig}. "
                "Periksa di audit (notebook 01) sebelum lanjut."
            )
    if verbose and ok:
        print("[verify] OK: jumlah sesuai harapan.")
    return ok


# --- Download --------------------------------------------------------------
def manual_instructions() -> None:
    print(
        "\n=== Unduhan otomatis gagal: lakukan langkah manual ===\n"
        f"1. Buka {config.DATASET_PAGE}\n"
        "2. Klik 'Download All' (ZIP).\n"
        f"3. Letakkan file ZIP di: {config.ZIP_DIR}\n"
        "4. Jalankan lagi: uv run python -m src.download_data\n"
    )


def download(url: str, dest: Path) -> None:
    """Stream url to dest via a .part file so an interrupted download is never treated as complete.

    If a .part file already exists, resume it with an HTTP Range request. Stopping the
    script (Ctrl+C, closing the laptop, network loss) is therefore safe: the next run
    continues where it left off.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "python-urllib/dataset-downloader"}
    if done:
        headers["Range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if done and resp.status == 206:
            # "Content-Range: bytes <start>-<end>/<total>"
            total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
            mode = "ab"
            print(f"[download] Melanjutkan dari {done / 1e6:.0f} MB")
        else:
            if done:
                print("[download] Server tidak mendukung resume; mengunduh dari awal.")
            done = 0
            total = int(resp.headers.get("Content-Length") or 0) or None
            mode = "wb"
        with open(part, mode) as fh, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    if total is not None and part.stat().st_size != total:
        raise IOError(
            f"Unduhan belum lengkap ({part.stat().st_size} dari {total} byte). "
            "Jalankan lagi perintah yang sama untuk melanjutkan."
        )
    part.replace(dest)


def find_zip() -> Path | None:
    """Prefer the standard name; otherwise accept any ZIP placed manually (e.g. 'vsjrkhs9km-2.zip')."""
    preferred = config.ZIP_DIR / config.ZIP_NAME
    if preferred.is_file():
        return preferred
    zips = sorted(config.ZIP_DIR.glob("*.zip")) if config.ZIP_DIR.is_dir() else []
    return zips[0] if zips else None


# --- Extraction ------------------------------------------------------------
def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract while skipping junk entries and blocking paths that escape out_dir."""
    out_root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.infolist()
            if not any(part in JUNK_NAMES for part in Path(m.filename).parts)
        ]
        for m in tqdm(members, desc=f"ekstrak {zip_path.name}", unit="file"):
            target = (out_dir / m.filename).resolve()
            if not target.is_relative_to(out_root):
                raise ValueError(f"Entri ZIP mencurigakan: {m.filename}")
            zf.extract(m, out_dir)


def extract_nested(zip_path: Path, work_dir: Path) -> None:
    """Extract zip_path into work_dir, then keep extracting any ZIPs found inside."""
    safe_extract(zip_path, work_dir)
    done: set[Path] = set()
    while True:
        inner = [z for z in work_dir.rglob("*.zip") if z not in done]
        if not inner:
            break
        for z in inner:
            safe_extract(z, z.parent / z.stem)
            done.add(z)
            z.unlink()


def collect_class_dirs(work_dir: Path) -> dict[str, Path]:
    """Find one folder per class name (folder containing .jpg files)."""
    found: dict[str, Path] = {}
    for d in work_dir.rglob("*"):
        if d.is_dir() and d.name in config.CLASSES and any(
            f.suffix.lower() == ".jpg" for f in d.iterdir()
        ):
            if d.name in found:
                print(f"[extract] PERINGATAN: folder kelas ganda '{d.name}', memakai {found[d.name]}")
                continue
            found[d.name] = d
    return found


def extract(zip_path: Path) -> None:
    config.RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.DATA_DIR, prefix="tmp_extract_") as tmp:
        work = Path(tmp)
        extract_nested(zip_path, work)
        class_dirs = collect_class_dirs(work)
        missing = sorted(set(config.CLASSES) - set(class_dirs))
        if missing:
            print(f"[extract] PERINGATAN: folder kelas tidak ditemukan di ZIP: {missing}")
        if config.RAW_DIR.exists():
            shutil.rmtree(config.RAW_DIR)
        config.RAW_DIR.mkdir(parents=True)
        for name, src_dir in sorted(class_dirs.items()):
            shutil.move(str(src_dir), str(config.RAW_DIR / name))
        # Remove junk files that slipped through inside class folders.
        for f in config.RAW_DIR.rglob("*"):
            if f.is_file() and f.name in JUNK_NAMES:
                f.unlink()
    print(f"[extract] Selesai: {len(class_dirs)} folder kelas -> {config.RAW_DIR}")


# --- Main ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="ulangi ekstraksi dari awal")
    args = parser.parse_args(argv)

    if not args.force and config.RAW_DIR.is_dir():
        if verify(verbose=False):
            print("[main] Dataset sudah ada dan lolos verifikasi. Tidak ada yang dikerjakan.")
            verify()
            return 0
        print("[main] Dataset ada tetapi tidak lolos verifikasi; mengekstrak ulang.")

    zip_path = find_zip()
    if zip_path is None:
        if config.DATASET_URL is None:
            print("[main] DATASET_URL belum diset.")
            manual_instructions()
            return 1
        zip_path = config.ZIP_DIR / config.ZIP_NAME
        print(f"[main] Mengunduh {config.DATASET_URL}")
        try:
            download(config.DATASET_URL, zip_path)
        except KeyboardInterrupt:
            print("\n[main] Unduhan dijeda. Jalankan lagi perintah yang sama untuk melanjutkan.")
            return 130
        except Exception as exc:  # network, HTTP error, auth wall, size mismatch
            print(f"[main] Gagal mengunduh: {exc}")
            manual_instructions()
            return 1
    else:
        print(f"[main] Memakai ZIP yang sudah ada: {zip_path}")

    try:
        extract(zip_path)
    except zipfile.BadZipFile as exc:
        print(f"[main] ZIP rusak ({exc}). Hapus {zip_path} lalu ulangi.")
        manual_instructions()
        return 1

    return 0 if verify() else 2


if __name__ == "__main__":
    sys.exit(main())

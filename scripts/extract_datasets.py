from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "DATASET"
RAW_DIR = ROOT / "data" / "raw"
TMP_DIR = RAW_DIR / "_tmp"

ARCHIVES = {
    "dentex": ARCHIVE_DIR / "7812323.zip",
    "dentalai": ARCHIVE_DIR / "dentalai-DatasetNinja.tar",
    "tufts": ARCHIVE_DIR / "tufts-dental-database.zip",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def marker_path(target_dir: Path) -> Path:
    return target_dir / ".extracted"


def is_done(target_dir: Path) -> bool:
    return marker_path(target_dir).exists()


def mark_done(target_dir: Path, text: str) -> None:
    marker_path(target_dir).write_text(text, encoding="utf-8")


def extract_tar(archive: Path, target_dir: Path, force: bool = False) -> None:
    if is_done(target_dir) and not force:
        print(f"[skip] {archive.name} already extracted to {target_dir}")
        return
    ensure_dir(target_dir)
    print(f"[extract] {archive.name} -> {target_dir}")
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(target_dir)
    mark_done(target_dir, f"Extracted from {archive}\n")


def extract_zip(archive: Path, target_dir: Path, force: bool = False) -> None:
    if is_done(target_dir) and not force:
        print(f"[skip] {archive.name} already extracted to {target_dir}")
        return
    ensure_dir(target_dir)
    print(f"[extract] {archive.name} -> {target_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    mark_done(target_dir, f"Extracted from {archive}\n")


def extract_nested_dentex(archive: Path, target_dir: Path, force: bool = False) -> None:
    if is_done(target_dir) and not force:
        print(f"[skip] {archive.name} already extracted to {target_dir}")
        return

    ensure_dir(target_dir)
    ensure_dir(TMP_DIR)

    print(f"[extract] {archive.name} nested archives -> {target_dir}")
    with zipfile.ZipFile(archive) as outer:
        members = [m for m in outer.namelist() if m.endswith(".zip")]
        if not members:
            raise RuntimeError(f"No nested zip files found in {archive}")

        for member in members:
            tmp_zip = TMP_DIR / Path(member).name
            print(f"  [stage] {member} -> {tmp_zip}")
            with outer.open(member) as src, tmp_zip.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)

            print(f"  [unzip] {tmp_zip.name} -> {target_dir}")
            with zipfile.ZipFile(tmp_zip) as inner:
                inner.extractall(target_dir)

            tmp_zip.unlink(missing_ok=True)

    mark_done(target_dir, f"Extracted nested zips from {archive}\n")


def validate_archives() -> None:
    missing = [name for name, path in ARCHIVES.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name}: {ARCHIVES[name]}" for name in missing)
        raise FileNotFoundError(f"Missing dataset archive(s): {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw dental AI datasets.")
    parser.add_argument(
        "--only",
        choices=["dentex", "dentalai", "tufts"],
        help="Extract only one dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if marker exists.")
    args = parser.parse_args()

    validate_archives()
    ensure_dir(RAW_DIR)

    tasks = {
        "dentalai": lambda: extract_tar(ARCHIVES["dentalai"], RAW_DIR / "dentalai", args.force),
        "tufts": lambda: extract_zip(ARCHIVES["tufts"], RAW_DIR / "tufts", args.force),
        "dentex": lambda: extract_nested_dentex(ARCHIVES["dentex"], RAW_DIR / "dentex", args.force),
    }

    selected = [args.only] if args.only else ["dentalai", "tufts", "dentex"]
    for name in selected:
        tasks[name]()

    print("[done] extraction complete")


if __name__ == "__main__":
    main()

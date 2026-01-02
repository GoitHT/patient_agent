from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Chroma index from ./kb")
    parser.add_argument("--kb", type=Path, default=ROOT / "kb", help="KB root directory")
    parser.add_argument(
        "--persist",
        type=Path,
        default=ROOT / ".chroma",
        help="Chroma persist directory (default: <repo>/.chroma)",
    )
    parser.add_argument("--collection", type=str, default="hospital_kb", help="Collection name")
    args = parser.parse_args()

    kb_root = Path(args.kb)
    persist_dir = Path(args.persist)
    if not kb_root.is_absolute():
        kb_root = (ROOT / kb_root).resolve()
    if not persist_dir.is_absolute():
        persist_dir = (ROOT / persist_dir).resolve()

    info = build_index(kb_root=kb_root, persist_dir=persist_dir, collection_name=args.collection)
    print(info)


if __name__ == "__main__":
    main()

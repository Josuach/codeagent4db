"""
Incremental update tracker.

Tracks file content hashes to determine which source files have changed
since the last preprocessing run. Only changed files need re-processing.

Usage:
    from incremental import IncrementalTracker
    tracker = IncrementalTracker("cache/file_hashes.json")
    changed = tracker.get_changed_files(file_list)
    tracker.update(changed)
    tracker.save()
"""

import hashlib
import json
import os
from typing import Optional


class IncrementalTracker:
    """
    Tracks SHA-256 hashes of source files to detect changes.
    """

    def __init__(self, hash_file: str):
        self.hash_file = hash_file
        self._hashes: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.hash_file), exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2)

    @staticmethod
    def _file_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_changed(self, filepath: str) -> bool:
        """Return True if the file has changed since last recorded hash."""
        try:
            current = self._file_hash(filepath)
        except OSError:
            return True
        return self._hashes.get(filepath) != current

    def get_changed_files(self, filepaths: list[str]) -> list[str]:
        """Return the subset of filepaths that have changed."""
        return [f for f in filepaths if self.is_changed(f)]

    def update(self, filepaths: list[str]):
        """Update stored hashes for the given files."""
        for fp in filepaths:
            try:
                self._hashes[fp] = self._file_hash(fp)
            except OSError:
                pass

    def remove_deleted(self, existing_filepaths: set[str]):
        """Remove hash entries for files that no longer exist."""
        to_remove = [k for k in self._hashes if k not in existing_filepaths]
        for k in to_remove:
            del self._hashes[k]

    def get_all_tracked(self) -> list[str]:
        return list(self._hashes.keys())


def find_changed_files(
    project_root: str,
    hash_file: str,
    extensions: tuple[str, ...] = (".c", ".h"),
) -> tuple[list[str], IncrementalTracker]:
    """
    Walk project_root, find all source files, and return those that have changed.

    Returns:
        (changed_file_paths, tracker) — caller should call tracker.save() after processing
    """
    tracker = IncrementalTracker(hash_file)

    all_files = []
    for dirpath, _, filenames in os.walk(project_root):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(dirpath, fname))

    tracker.remove_deleted(set(all_files))
    changed = tracker.get_changed_files(all_files)
    return changed, tracker

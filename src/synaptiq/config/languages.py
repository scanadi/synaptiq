"""Language detection based on file extensions."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".ru": "ruby",
    ".rbi": "ruby",
}

# Ruby (and a few other ecosystems) use suffix-less or non-``.rb`` files that are
# nonetheless Ruby source.  These are matched by basename before falling back to
# the extension lookup.
SPECIAL_FILENAMES: dict[str, str] = {
    "Rakefile": "ruby",
    "Gemfile": "ruby",
    "Guardfile": "ruby",
    "Capfile": "ruby",
    "Vagrantfile": "ruby",
    "Brewfile": "ruby",
    "Podfile": "ruby",
}

def get_language(file_path: str | Path) -> str | None:
    """Return the language name for *file_path*.

    Special Ruby filenames (e.g. ``Rakefile``, ``Gemfile``) are matched by
    basename first; otherwise the file's extension is looked up in
    :data:`SUPPORTED_EXTENSIONS`.

    Returns ``None`` when neither the name nor the extension is recognized.
    """
    path = Path(file_path)
    special = SPECIAL_FILENAMES.get(path.name)
    if special is not None:
        return special
    return SUPPORTED_EXTENSIONS.get(path.suffix)

def is_supported(file_path: str | Path) -> bool:
    """Return ``True`` if *file_path* has a supported name or extension."""
    path = Path(file_path)
    return path.name in SPECIAL_FILENAMES or path.suffix in SUPPORTED_EXTENSIONS

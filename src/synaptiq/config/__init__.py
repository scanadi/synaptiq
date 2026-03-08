"""Synaptiq configuration — ignore patterns and language detection."""

from synaptiq.config.ignore import DEFAULT_IGNORE_PATTERNS, load_gitignore, should_ignore
from synaptiq.config.languages import SUPPORTED_EXTENSIONS, get_language, is_supported

__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "SUPPORTED_EXTENSIONS",
    "get_language",
    "is_supported",
    "load_gitignore",
    "should_ignore",
]

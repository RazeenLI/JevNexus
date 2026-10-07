"""Compatibility imports for the model's former DeMa name."""

from .jevnexus import JevNexusMatcher

DeMaMatcher = JevNexusMatcher

__all__ = ["JevNexusMatcher", "DeMaMatcher"]

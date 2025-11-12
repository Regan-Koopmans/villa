"""Mesh processing utilities for Vesuvius."""

from .inpaint import (
    inpaint_mesh,
    detect_holes,
    validate_mesh,
    load_mesh,
    save_mesh,
    HoleInfo,
    MeshValidationResult,
)

__all__ = [
    "inpaint_mesh",
    "detect_holes",
    "validate_mesh",
    "load_mesh",
    "save_mesh",
    "HoleInfo",
    "MeshValidationResult",
]

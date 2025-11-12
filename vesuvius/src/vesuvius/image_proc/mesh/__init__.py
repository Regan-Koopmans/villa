"""Mesh processing utilities for Vesuvius."""

from .inpaint import (
    inpaint_mesh,
    validate_mesh,
    MeshValidationResult,
)

__all__ = [
    "inpaint_mesh",
    "validate_mesh",
    "MeshValidationResult",
]

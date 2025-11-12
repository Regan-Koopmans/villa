import numpy as np
import open3d as o3d
import pymeshlab
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class MeshValidationResult:
    """Results from mesh validation checks."""
    is_watertight: bool
    has_self_intersections: bool
    has_degenerate_triangles: bool
    has_normals: bool
    num_holes: int
    num_vertices: int
    num_faces: int
    bounding_box_volume: float
    issues: List[str]




def validate_mesh(vertices: np.ndarray, faces: np.ndarray,
                 check_intersections: bool = True) -> MeshValidationResult:
    """
    Validate mesh quality and detect issues.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        check_intersections: Whether to check for self-intersections (expensive)

    Returns:
        MeshValidationResult with detailed validation information
    """
    issues = []

    # Create Open3D mesh for validation
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Check if mesh is watertight
    is_watertight = mesh.is_watertight()
    if not is_watertight:
        issues.append("Mesh is not watertight (has holes or boundaries)")

    # Note: Hole counting is done via watertight check
    # If not watertight, there are holes - but we don't count them individually
    num_holes = 0 if is_watertight else -1  # -1 indicates "unknown number of holes"

    # Check for degenerate triangles
    has_degenerate = False
    degenerate_count = 0
    min_area = 1e-10
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        if area < min_area:
            has_degenerate = True
            degenerate_count += 1

    if has_degenerate:
        issues.append(f"Found {degenerate_count} degenerate triangle(s)")

    # Check for normals
    has_normals = mesh.has_vertex_normals()
    if not has_normals:
        issues.append("Mesh does not have vertex normals")

    # Check for self-intersections (expensive)
    has_self_intersections = False
    if check_intersections:
        try:
            if mesh.is_self_intersecting():
                has_self_intersections = True
                issues.append("Mesh has self-intersections")
        except Exception as e:
            logging.warning(f"Could not check self-intersections: {e}")

    # Calculate bounding box volume
    bbox = mesh.get_axis_aligned_bounding_box()
    bb_volume = bbox.volume()

    return MeshValidationResult(
        is_watertight=is_watertight,
        has_self_intersections=has_self_intersections,
        has_degenerate_triangles=has_degenerate,
        has_normals=has_normals,
        num_holes=num_holes,
        num_vertices=len(vertices),
        num_faces=len(faces),
        bounding_box_volume=bb_volume,
        issues=issues
    )


def inpaint_mesh(mesh_path: Union[str, Path],
                output_path: Optional[Union[str, Path]] = None,
                validate: bool = True,
                max_hole_size: int = 1000000,
                smoothing_steps: int = 3,
                simplify: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[MeshValidationResult]]:
    """
    Inpaint holes in a triangle mesh using PyMeshLab's hole filling algorithms.

    This method uses MeshLab's robust hole filling implementation, which handles
    complex geometries better than custom implementations.

    Args:
        mesh_path: Path to input .obj mesh file with holes
        output_path: Optional path to save the inpainted mesh
        validate: Whether to validate the mesh after inpainting
        max_hole_size: Maximum hole size (edge count) to fill (default: 1000000 for all holes)
        smoothing_steps: Number of Laplacian smoothing steps to apply (default: 3)
        simplify: Whether to simplify the mesh after hole filling

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        validation: MeshValidationResult if validate=True, else None

    Example:
        >>> from vesuvius.image_proc.mesh.inpaint import inpaint_mesh
        >>> vertices, faces, validation = inpaint_mesh(
        ...     "surface_with_holes.obj",
        ...     "surface_inpainted.obj"
        ... )
        >>> print(f"Filled {validation.num_holes} holes")
    """
    mesh_path = Path(mesh_path)

    logging.info(f"Loading mesh from {mesh_path}...")

    # Use PyMeshLab for hole filling (it handles hole detection internally)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    # Get mesh info for logging
    current_mesh = ms.current_mesh()
    num_vertices = current_mesh.vertex_number()
    num_faces = current_mesh.face_number()
    logging.info(f"Loaded mesh with {num_vertices} vertices and {num_faces} faces")

    logging.info(f"Filling holes using PyMeshLab's refinement-based algorithm...")

    # Compute vertex normals if not present (needed for better hole filling)
    try:
        ms.compute_normal_per_vertex()
        logging.info("Computed vertex normals")
    except Exception as e:
        logging.warning(f"Could not compute normals: {e}")

    # Close holes using MeshLab's hole filling
    try:
        # Use the standard hole closing algorithm
        ms.meshing_close_holes(maxholesize=max_hole_size)
        logging.info("Successfully closed holes")
    except Exception as e:
        logging.error(f"Hole filling failed: {e}")
        logging.warning("Returning original mesh")
        # Get original mesh data for error return
        current_mesh = ms.current_mesh()
        vertices = current_mesh.vertex_matrix()
        faces = current_mesh.face_matrix()
        if validate:
            validation_result = validate_mesh(vertices, faces)
            return vertices, faces, validation_result
        return vertices, faces, None

    # Refine the filled regions for better quality
    # This approach uses subdivision and smoothing to improve the filled patches
    try:
        logging.info("Refining filled regions with subdivision and smoothing...")

        # Step 1: Apply one iteration of Loop subdivision to the newly filled faces
        # This creates a smoother, more refined mesh in the filled regions
        try:
            ms.meshing_surface_subdivision_loop(iterations=1)
            logging.info("Applied Loop subdivision for refinement")
        except Exception as e:
            logging.warning(f"Loop subdivision failed: {e}")

        # Step 2: Apply HC Laplacian smoothing (better than standard Laplacian)
        # HC smoothing preserves features better while smoothing the surface
        if smoothing_steps > 0:
            try:
                # HC Laplacian: alternates between Laplacian smoothing and inverse smoothing
                ms.apply_coord_hc_laplacian_smoothing(stepsmoothnum=smoothing_steps)
                logging.info(f"Applied HC Laplacian smoothing ({smoothing_steps} steps)")
            except Exception as e:
                logging.warning(f"HC Laplacian smoothing failed, trying Taubin: {e}")
                try:
                    # Fall back to Taubin smoothing (volume-preserving)
                    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=smoothing_steps)
                    logging.info(f"Applied Taubin smoothing ({smoothing_steps} steps)")
                except Exception as e2:
                    logging.warning(f"Taubin smoothing failed, trying basic Laplacian: {e2}")
                    try:
                        # Last resort: basic Laplacian
                        ms.apply_coord_laplacian_smoothing(stepsmoothnum=smoothing_steps)
                        logging.info(f"Applied Laplacian smoothing ({smoothing_steps} steps)")
                    except Exception as e3:
                        logging.warning(f"All smoothing methods failed: {e3}")

        # Step 3: Recompute normals after refinement
        try:
            ms.compute_normal_per_vertex()
            logging.info("Recomputed normals after refinement")
        except Exception as e:
            logging.warning(f"Could not recompute normals: {e}")

        logging.info("Refinement complete")

    except Exception as e:
        logging.warning(f"Refinement process failed (continuing with basic fill): {e}")

    # Optional: Simplify the mesh
    if simplify:
        logging.info("Simplifying mesh...")
        try:
            original_faces = ms.current_mesh().face_number()
            target_faces = int(original_faces * 0.95)  # Reduce by 5%
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            logging.info(f"Simplified from {original_faces} to {ms.current_mesh().face_number()} faces")
        except Exception as e:
            logging.warning(f"Simplification failed (continuing anyway): {e}")

    # Extract the result
    current_mesh = ms.current_mesh()
    new_vertices = current_mesh.vertex_matrix()
    new_faces = current_mesh.face_matrix()

    logging.info(f"Inpainting complete: {len(new_vertices)} vertices, {len(new_faces)} faces")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        logging.info(f"Saving result to {output_path}...")
        ms.save_current_mesh(str(output_path), save_vertex_normal=True)
        logging.info(f"Saved inpainted mesh to {output_path}")

    # Validate result
    validation_result = None
    if validate:
        logging.info("Validating result...")
        validation_result = validate_mesh(new_vertices, new_faces, check_intersections=False)
        logging.info(f"Validation: watertight={validation_result.is_watertight}, "
                    f"holes={validation_result.num_holes}")
        if validation_result.issues:
            logging.warning("Validation issues found:")
            for issue in validation_result.issues:
                logging.warning(f"  - {issue}")

    return new_vertices, new_faces, validation_result

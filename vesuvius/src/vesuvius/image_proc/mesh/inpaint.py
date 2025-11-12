import numpy as np
import open3d as o3d
import igl
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import logging
from scipy.sparse.linalg import spsolve


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
    Inpaint holes in a triangle mesh using libigl.

    Uses libigl for hole detection, filling, and mesh refinement.

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
        >>> print(f"Inpainted mesh: {len(vertices)} vertices, {len(faces)} faces")
    """
    mesh_path = Path(mesh_path)

    logging.info(f"Loading mesh from {mesh_path}...")

    # Load mesh with igl
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    num_vertices = len(vertices)
    num_faces = len(faces)
    logging.info(f"Loaded mesh with {num_vertices} vertices and {num_faces} faces")

    # Detect boundary loops (holes) using igl
    logging.info("Detecting boundary loops...")
    boundary_loops = igl.boundary_loop(faces)

    if len(boundary_loops) == 0:
        logging.info("No holes detected - mesh is already watertight")
        # Still save if output path provided
        if output_path:
            output_path = Path(output_path)
            logging.info(f"Saving mesh to {output_path}...")
            igl.write_triangle_mesh(str(output_path), vertices, faces)

        validation_result = None
        if validate:
            logging.info("Validating mesh...")
            validation_result = validate_mesh(vertices, faces, check_intersections=False)

        return vertices, faces, validation_result

    logging.info(f"Found {len(boundary_loops)} boundary loops (holes)")

    # Fill holes using igl
    logging.info("Filling holes using igl...")

    # Use libigl to fill holes
    try:
        # Fill all holes that are within max_hole_size
        filled_vertices = vertices.copy()
        new_faces_list = [faces]

        # For each boundary loop, check size and fill if needed
        for i, loop in enumerate(boundary_loops):
            if len(loop) > max_hole_size:
                logging.warning(f"Skipping hole {i+1} (size {len(loop)} > max {max_hole_size})")
                continue

            logging.info(f"Filling hole {i+1} with {len(loop)} boundary edges")

            # Vectorized fan triangulation from first vertex
            if len(loop) >= 3:
                # Create fan triangles: (v0, v1, v2), (v0, v2, v3), ...
                n_triangles = len(loop) - 2
                fan_faces = np.zeros((n_triangles, 3), dtype=np.int32)
                fan_faces[:, 0] = loop[0]  # All triangles share first vertex
                fan_faces[:, 1] = loop[1:-1]  # Sequential vertices
                fan_faces[:, 2] = loop[2:]  # Next vertices
                new_faces_list.append(fan_faces)

        # Combine all faces
        filled_faces = np.vstack(new_faces_list)
        logging.info(f"Successfully filled holes: {len(filled_faces) - len(faces)} new faces added")

    except Exception as e:
        logging.error(f"Hole filling failed: {e}")
        logging.warning("Returning original mesh")
        if validate:
            validation_result = validate_mesh(vertices, faces)
            return vertices, faces, validation_result
        return vertices, faces, None

    # Refine the filled regions for better quality
    new_vertices = filled_vertices
    new_faces = filled_faces

    if smoothing_steps > 0 or simplify:
        try:
            logging.info("Refining filled regions with subdivision and smoothing...")

            # Step 1: Apply Loop subdivision using igl
            try:
                new_vertices, new_faces = igl.upsample(new_vertices, new_faces, number_of_subdivs=1)
                logging.info("Applied Loop subdivision for refinement")
            except Exception as e:
                logging.warning(f"Loop subdivision failed: {e}")

            # Step 2: Apply Laplacian smoothing using igl
            if smoothing_steps > 0:
                try:
                    # igl Laplacian smoothing
                    # Build the cotangent Laplacian matrix (computed once)
                    L = igl.cotmatrix(new_vertices, new_faces)

                    # Mass matrix for normalization (computed once)
                    M = igl.massmatrix(new_vertices, new_faces, igl.MASSMATRIX_TYPE_VORONOI)

                    # Precompute system matrix (computed once)
                    lambda_smooth = 0.5  # Smoothing parameter
                    A = M - lambda_smooth * L

                    # Apply iterative smoothing
                    smoothed_vertices = new_vertices.copy()
                    for _ in range(smoothing_steps):
                        # Solve Laplacian smoothing: (M - lambda*L)*V' = M*V
                        # Solve for each coordinate dimension independently
                        for dim in range(3):
                            b = M @ smoothed_vertices[:, dim]
                            # Convert to dense if sparse
                            b_dense = b.toarray().ravel() if hasattr(b, 'toarray') else b
                            smoothed_vertices[:, dim] = spsolve(A, b_dense)

                    new_vertices = smoothed_vertices
                    logging.info(f"Applied Laplacian smoothing ({smoothing_steps} steps)")
                except Exception as e:
                    logging.warning(f"Laplacian smoothing failed: {e}")

            logging.info("Refinement complete")

        except Exception as e:
            logging.warning(f"Refinement process failed (continuing with basic fill): {e}")

    # Optional: Simplify the mesh using igl
    if simplify:
        logging.info("Simplifying mesh...")
        try:
            original_faces = len(new_faces)
            target_faces = int(original_faces * 0.95)  # Reduce by 5%

            # Use igl's edge collapse decimation
            new_vertices, new_faces, _, _ = igl.decimate(new_vertices, new_faces, target_faces)
            logging.info(f"Simplified from {original_faces} to {len(new_faces)} faces")
        except Exception as e:
            logging.warning(f"Simplification failed (continuing anyway): {e}")

    logging.info(f"Inpainting complete: {len(new_vertices)} vertices, {len(new_faces)} faces")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        logging.info(f"Saving result to {output_path}...")
        igl.write_triangle_mesh(str(output_path), new_vertices, new_faces)
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

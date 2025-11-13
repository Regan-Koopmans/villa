import numpy as np
import open3d as o3d
import igl
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
                 check_intersections: bool = True,
                 use_streaming: bool = False) -> MeshValidationResult:
    """
    Validate mesh quality and detect issues.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        check_intersections: Whether to check for self-intersections (expensive)
        use_streaming: Use memory-efficient streaming validation (for large meshes)

    Returns:
        MeshValidationResult with detailed validation information
    """
    issues = []

    # For large meshes, use streaming/chunked approach
    if use_streaming or len(faces) > 5_000_000:
        logging.info("Using streaming validation for large mesh")

        # Check watertight using igl (more memory efficient than Open3D)
        try:
            boundary_loops = igl.boundary_loop_all(faces)
            is_watertight = len(boundary_loops) == 0
            num_holes = len(boundary_loops)
            if not is_watertight:
                issues.append(f"Mesh is not watertight ({num_holes} holes)")
        except Exception as e:
            logging.warning(f"Could not check watertight: {e}")
            is_watertight = False
            num_holes = -1
            issues.append("Could not determine watertight status")

        # Check for degenerate triangles in chunks to save memory
        has_degenerate = False
        degenerate_count = 0
        min_area = 1e-10
        chunk_size = 100_000

        for i in range(0, len(faces), chunk_size):
            chunk_faces = faces[i:i+chunk_size]
            # Vectorized computation per chunk
            v0 = vertices[chunk_faces[:, 0]]
            v1 = vertices[chunk_faces[:, 1]]
            v2 = vertices[chunk_faces[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            chunk_degenerate = np.sum(areas < min_area)
            if chunk_degenerate > 0:
                has_degenerate = True
                degenerate_count += chunk_degenerate

        if has_degenerate:
            issues.append(f"Found {degenerate_count} degenerate triangle(s)")

        # Calculate bounding box
        bb_min = np.min(vertices, axis=0)
        bb_max = np.max(vertices, axis=0)
        bb_volume = np.prod(bb_max - bb_min)

        # Skip expensive checks for streaming
        has_normals = False
        has_self_intersections = False

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

    # Original Open3D-based validation for smaller meshes
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
                simplify: bool = False,
                use_pymeshlab: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[MeshValidationResult]]:
    """
    Inpaint holes in a triangle mesh using PyMeshLab or libigl.

    Uses PyMeshLab's robust hole filling (default) or libigl for hole detection and basic filling.

    Args:
        mesh_path: Path to input .obj mesh file with holes
        output_path: Optional path to save the inpainted mesh
        validate: Whether to validate the mesh after inpainting
        max_hole_size: Maximum hole size (edge count) to fill (default: 100000 edges, essentially all holes)
        smoothing_steps: Number of Taubin smoothing iterations to apply to hole boundaries (default: 3)
        simplify: Whether to simplify the mesh after hole filling
        use_pymeshlab: Use PyMeshLab's robust hole filling (recommended for complex 3D holes)

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

    # Use PyMeshLab for robust hole filling
    if use_pymeshlab:
        logging.info(f"Using PyMeshLab for hole filling: {mesh_path}")
        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_path))

            mesh = ms.current_mesh()
            logging.info(f"Loaded mesh: {mesh.vertex_number():,} vertices, {mesh.face_number():,} faces")

            # Step 1: Repair non-manifold geometry (preprocessing)
            logging.info("Repairing non-manifold geometry...")
            try:
                ms.meshing_repair_non_manifold_edges()
                logging.info("  Repaired non-manifold edges")
            except Exception as e:
                logging.warning(f"  Non-manifold edge repair failed: {e}")

            try:
                ms.meshing_repair_non_manifold_vertices()
                logging.info("  Repaired non-manifold vertices")
            except Exception as e:
                logging.warning(f"  Non-manifold vertex repair failed: {e}")

            try:
                ms.meshing_remove_duplicate_vertices()
                logging.info("  Removed duplicate vertices")
            except Exception as e:
                logging.warning(f"  Duplicate vertex removal failed: {e}")

            try:
                ms.meshing_remove_duplicate_faces()
                logging.info("  Removed duplicate faces")
            except Exception as e:
                logging.warning(f"  Duplicate face removal failed: {e}")

            mesh = ms.current_mesh()
            logging.info(f"After preprocessing: {mesh.vertex_number():,} vertices, {mesh.face_number():,} faces")

            # Step 2: Detect and log holes before filling
            logging.info("Detecting holes in mesh...")
            try:
                # Get boundary edges to estimate hole count
                vertices_np = mesh.vertex_matrix()
                faces_np = mesh.face_matrix()
                boundary_loops = igl.boundary_loop_all(faces_np)
                num_holes = len(boundary_loops)
                logging.info(f"  Found {num_holes} boundary loops (holes)")

                if num_holes > 0:
                    hole_sizes = [len(loop) for loop in boundary_loops]
                    logging.info(f"  Hole sizes (boundary edges): min={min(hole_sizes)}, max={max(hole_sizes)}, mean={sum(hole_sizes)/len(hole_sizes):.1f}")
                    # Log largest holes
                    largest_holes = sorted(enumerate(hole_sizes), key=lambda x: x[1], reverse=True)[:5]
                    logging.info(f"  Top 5 largest holes:")
                    for idx, (hole_idx, size) in enumerate(largest_holes):
                        logging.info(f"    Hole {hole_idx}: {size} edges")
            except Exception as e:
                logging.warning(f"  Hole detection failed: {e}")

            # Step 3: Close holes using PyMeshLab's robust algorithm
            logging.info(f"Filling holes iteratively...")

            # Run hole filling iteratively - sometimes holes are only detected after previous ones are filled
            max_iterations = 10
            for iteration in range(max_iterations):
                logging.info(f"Hole filling iteration {iteration + 1}/{max_iterations}")

                # Check how many holes exist before filling
                initial_mesh = ms.current_mesh()
                initial_faces = initial_mesh.face_number()

                try:
                    # Fill all holes (no size limit)
                    ms.meshing_close_holes()
                except Exception as e:
                    logging.error(f"Hole filling failed in iteration {iteration + 1}: {e}")
                    break

                # Check if anything changed
                current_mesh = ms.current_mesh()
                new_vertices = current_mesh.vertex_number()
                new_faces = current_mesh.face_number()

                if new_faces == initial_faces:
                    logging.info(f"No new faces added in iteration {iteration + 1}, stopping")
                    break
                else:
                    logging.info(f"Added {new_faces - initial_faces} faces in iteration {iteration + 1}")

                # Check remaining holes after this iteration
                try:
                    vertices_iter = current_mesh.vertex_matrix()
                    faces_iter = current_mesh.face_matrix()
                    boundary_loops_iter = igl.boundary_loop_all(faces_iter)
                    remaining_holes = len(boundary_loops_iter)
                    logging.info(f"  Remaining boundary loops after iteration {iteration + 1}: {remaining_holes}")

                    # Analyze hole size distribution
                    if remaining_holes > 0:
                        hole_sizes_iter = [len(loop) for loop in boundary_loops_iter]
                        small_holes = sum(1 for s in hole_sizes_iter if s < 10)
                        medium_holes = sum(1 for s in hole_sizes_iter if 10 <= s < 100)
                        large_holes = sum(1 for s in hole_sizes_iter if s >= 100)
                        logging.info(f"    Small (<10 edges): {small_holes}, Medium (10-99): {medium_holes}, Large (>=100): {large_holes}")
                        if remaining_holes <= 20:
                            logging.info(f"    Exact sizes: {sorted(hole_sizes_iter, reverse=True)}")
                except Exception as e:
                    logging.warning(f"  Could not check remaining holes: {e}")

            logging.info(f"Hole filling complete after {min(iteration + 1, max_iterations)} iterations")

            # Get the result
            mesh = ms.current_mesh()
            vertices = mesh.vertex_matrix()
            faces = mesh.face_matrix()

            logging.info(f"After hole filling: {len(vertices):,} vertices, {len(faces):,} faces")

            # Save if requested
            if output_path:
                output_path = Path(output_path)
                logging.info(f"Saving to {output_path}...")
                ms.save_current_mesh(str(output_path))

            # Validate if requested
            validation_result = None
            if validate:
                logging.info("Validating result...")
                use_streaming = len(faces) > 5_000_000
                validation_result = validate_mesh(vertices, faces,
                                                 check_intersections=False,
                                                 use_streaming=use_streaming)
                logging.info(f"Validation: watertight={validation_result.is_watertight}, "
                            f"holes={validation_result.num_holes}")
                if validation_result.issues:
                    logging.warning("Validation issues found:")
                    for issue in validation_result.issues:
                        logging.warning(f"  - {issue}")

            return vertices, faces, validation_result

        except Exception as e:
            logging.error(f"PyMeshLab hole filling failed: {e}")
            logging.info("Falling back to basic igl approach...")
            use_pymeshlab = False

    # Fallback: Use igl for basic hole detection and filling
    logging.info(f"Loading mesh from {mesh_path}...")

    # Load mesh with igl
    try:
        vertices, faces = igl.read_triangle_mesh(str(mesh_path))
        logging.info(f"Successfully loaded mesh, vertices shape: {vertices.shape}, faces shape: {faces.shape}")
    except Exception as e:
        logging.error(f"Failed to load mesh: {e}")
        raise
    num_vertices = len(vertices)
    num_faces = len(faces)
    logging.info(f"Loaded mesh with {num_vertices} vertices and {num_faces} faces")

    # Detect boundary loops (holes) using igl
    logging.info("Detecting boundary loops...")
    try:
        boundary_loops = igl.boundary_loop_all(faces)
        logging.info(f"boundary_loop_all returned type: {type(boundary_loops)}")
    except Exception as e:
        logging.error(f"Failed to detect boundary loops: {e}")
        raise

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

        # Filter and categorize boundary loops
        real_holes = []
        degenerate_holes = 0
        hole_vertex_indices = set()  # Track vertices involved in holes for selective smoothing

        for i, loop in enumerate(boundary_loops):
            # Convert to numpy array if it's a list
            if isinstance(loop, list):
                loop = np.array(loop, dtype=np.int32)

            # Skip degenerate holes (< 3 vertices can't form a triangle)
            if len(loop) < 3:
                degenerate_holes += 1
                continue

            # Skip holes larger than max size
            if len(loop) > max_hole_size:
                logging.warning(f"Skipping hole {i+1} (size {len(loop)} > max {max_hole_size})")
                continue

            real_holes.append((i, loop))
            # Track vertices that are part of hole boundaries
            hole_vertex_indices.update(loop.tolist())

        logging.info(f"Found {len(real_holes)} fillable holes, {degenerate_holes} degenerate holes (skipped)")
        logging.info(f"Total vertices in hole boundaries: {len(hole_vertex_indices)}")

        # For each real hole, fill it
        for i, loop in real_holes:
            logging.info(f"Filling hole {i+1} with {len(loop)} boundary edges")

            # Vectorized fan triangulation from first vertex
            # Create fan triangles: (v0, v1, v2), (v0, v2, v3), ...
            n_triangles = len(loop) - 2
            fan_faces = np.zeros((n_triangles, 3), dtype=np.int32)
            fan_faces[:, 0] = loop[0]  # All triangles share first vertex
            fan_faces[:, 1] = loop[1:-1]  # Sequential vertices
            fan_faces[:, 2] = loop[2:]  # Next vertices
            new_faces_list.append(fan_faces)

        # Combine all faces
        if len(new_faces_list) > 1:  # More than just the original faces
            filled_faces = np.vstack(new_faces_list)
            logging.info(f"Successfully filled {len(real_holes)} holes: {len(filled_faces) - len(faces)} new faces added")
        else:
            filled_faces = faces
            logging.info("No holes were filled (all were degenerate)")

    except Exception as e:
        logging.error(f"Hole filling failed: {e}")
        logging.warning("Returning original mesh")
        if validate:
            validation_result = validate_mesh(vertices, faces)
            return vertices, faces, validation_result
        return vertices, faces, None

    # Store hole vertex indices for selective smoothing
    hole_vertices_to_smooth = np.array(list(hole_vertex_indices), dtype=np.int32) if hole_vertex_indices else np.array([], dtype=np.int32)

    # Clean up mesh: remove unreferenced vertices
    logging.info("Cleaning up mesh (removing unreferenced vertices)...")
    try:
        # Find which vertices are actually used in faces
        used_vertices = np.unique(filled_faces.ravel())
        n_removed = len(filled_vertices) - len(used_vertices)

        if n_removed > 0:
            logging.info(f"Removing {n_removed} unreferenced vertices")

            # Create mapping from old to new vertex indices
            vertex_map = np.full(len(filled_vertices), -1, dtype=np.int32)
            vertex_map[used_vertices] = np.arange(len(used_vertices))

            # Remap faces to new vertex indices
            new_faces = vertex_map[filled_faces]

            # Keep only used vertices
            new_vertices = filled_vertices[used_vertices]

            # Remap hole vertex indices for smoothing
            hole_vertices_to_smooth = vertex_map[hole_vertices_to_smooth]
            # Remove any vertices that were unreferenced (-1)
            hole_vertices_to_smooth = hole_vertices_to_smooth[hole_vertices_to_smooth >= 0]

            logging.info(f"Cleaned mesh: {len(new_vertices)} vertices, {len(new_faces)} faces")
            logging.info(f"Hole vertices for smoothing (after cleanup): {len(hole_vertices_to_smooth)}")
        else:
            logging.info("No unreferenced vertices found")
            new_vertices = filled_vertices
            new_faces = filled_faces

    except Exception as e:
        logging.warning(f"Mesh cleanup failed: {e}, continuing with original")
        new_vertices = filled_vertices
        new_faces = filled_faces

    # Refine the filled regions for better quality

    if smoothing_steps > 0 or simplify:
        try:
            logging.info("Refining filled regions with subdivision and smoothing...")

            # Step 1: Apply Loop subdivision using igl
            try:
                new_vertices, new_faces = igl.upsample(new_vertices, new_faces, number_of_subdivs=1)
                logging.info("Applied Loop subdivision for refinement")
            except Exception as e:
                logging.warning(f"Loop subdivision failed: {e}")

            # Step 2: Apply Taubin smoothing (only on hole vertices)
            if smoothing_steps > 0 and len(hole_vertices_to_smooth) > 0:
                logging.info(f"Applying Taubin smoothing to {len(hole_vertices_to_smooth)} hole boundary vertices")
                try:
                    # Taubin smoothing: alternating shrink/inflate to preserve volume
                    # More robust than naive Laplacian smoothing which causes shrinkage
                    lambda_val = 0.5  # Shrink factor
                    mu_val = -0.53  # Inflate factor (must be negative, slightly > -lambda)

                    # Create mask for hole vertices
                    hole_vertex_mask = np.zeros(len(new_vertices), dtype=bool)
                    hole_vertex_mask[hole_vertices_to_smooth] = True

                    # Build edge adjacency once (reused across iterations)
                    edges = igl.edges(new_faces)

                    for step in range(smoothing_steps):
                        # Create vertex-to-vertex adjacency
                        n_verts = len(new_vertices)
                        neighbor_sums = np.zeros_like(new_vertices)
                        neighbor_counts = np.zeros(n_verts, dtype=np.int32)

                        # Accumulate neighbor positions
                        for e in edges:
                            v0, v1 = e[0], e[1]
                            neighbor_sums[v0] += new_vertices[v1]
                            neighbor_sums[v1] += new_vertices[v0]
                            neighbor_counts[v0] += 1
                            neighbor_counts[v1] += 1

                        # Only update hole vertices
                        mask = neighbor_counts > 0
                        smooth_mask = mask & hole_vertex_mask

                        # Lambda step (shrink) - only for hole vertices
                        laplacian = np.zeros_like(new_vertices)
                        laplacian[smooth_mask] = (neighbor_sums[smooth_mask] / neighbor_counts[smooth_mask, np.newaxis]) - new_vertices[smooth_mask]
                        new_vertices[smooth_mask] = new_vertices[smooth_mask] + lambda_val * laplacian[smooth_mask]

                        # Mu step (inflate) - only for hole vertices
                        # Recompute neighbors with updated positions
                        neighbor_sums = np.zeros_like(new_vertices)
                        for e in edges:
                            v0, v1 = e[0], e[1]
                            neighbor_sums[v0] += new_vertices[v1]
                            neighbor_sums[v1] += new_vertices[v0]

                        laplacian = np.zeros_like(new_vertices)
                        laplacian[smooth_mask] = (neighbor_sums[smooth_mask] / neighbor_counts[smooth_mask, np.newaxis]) - new_vertices[smooth_mask]
                        new_vertices[smooth_mask] = new_vertices[smooth_mask] + mu_val * laplacian[smooth_mask]

                        if (step + 1) % max(1, smoothing_steps // 3) == 0:
                            logging.info(f"Smoothing progress: {step + 1}/{smoothing_steps}")

                    logging.info(f"Taubin smoothing complete: {smoothing_steps} iterations on {len(hole_vertices_to_smooth)} vertices")
                except Exception as e:
                    logging.warning(f"Taubin smoothing failed: {e}")

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
        try:
            logging.info("Validating result...")
            # Use streaming validation for large meshes
            use_streaming = len(new_faces) > 5_000_000
            validation_result = validate_mesh(new_vertices, new_faces,
                                             check_intersections=False,
                                             use_streaming=use_streaming)
            logging.info(f"Validation: watertight={validation_result.is_watertight}, "
                        f"holes={validation_result.num_holes}")
            if validation_result.issues:
                logging.warning("Validation issues found:")
                for issue in validation_result.issues:
                    logging.warning(f"  - {issue}")
        except Exception as e:
            logging.warning(f"Validation failed: {e}")
            validation_result = None

    return new_vertices, new_faces, validation_result

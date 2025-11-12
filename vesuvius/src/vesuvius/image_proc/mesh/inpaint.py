"""
Mesh hole inpainting for scroll surface reconstruction.

This module provides algorithms to detect and fill holes in triangle meshes
resulting from surface tracing. Multiple inpainting strategies are provided:
1. Advancing Front - iteratively fills holes by advancing boundary vertices
2. Poisson Reconstruction - uses implicit surface reconstruction
3. Fairing-based - fills holes with smooth minimal surfaces
4. Context-aware - uses neighboring surface information for better interpolation
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    logging.warning("libigl not available. Some features will be disabled.")

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False
    logging.warning("pymeshlab not available. Some features will be disabled.")


@dataclass
class HoleInfo:
    """Information about a detected hole in the mesh."""
    boundary_indices: np.ndarray  # Vertex indices forming the hole boundary
    boundary_vertices: np.ndarray  # 3D coordinates of boundary vertices
    area: float  # Approximate area of the hole
    perimeter: float  # Length of the hole boundary
    center: np.ndarray  # Approximate center of the hole


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


def load_mesh(mesh_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh from an OBJ file.

    Args:
        mesh_path: Path to the .obj file

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    if not mesh_path.suffix.lower() == '.obj':
        raise ValueError(f"Only .obj files are supported, got: {mesh_path.suffix}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not mesh.has_vertices():
        raise ValueError("Mesh has no vertices")

    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    return vertices, faces


def save_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: Union[str, Path],
              normals: Optional[np.ndarray] = None) -> None:
    """
    Save a mesh to an OBJ file.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        output_path: Path where to save the mesh
        normals: Optional (N, 3) array of vertex normals
    """
    output_path = Path(output_path)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    else:
        mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=True)


def detect_holes(vertices: np.ndarray, faces: np.ndarray) -> List[HoleInfo]:
    """
    Detect holes (boundaries) in a triangle mesh.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices

    Returns:
        List of HoleInfo objects describing each detected hole
    """
    if not HAS_IGL:
        raise ImportError("libigl is required for hole detection. Install with: pip install libigl")

    # Find all boundary loops
    boundary_loops = igl.boundary_loop(faces)

    holes = []
    for loop in boundary_loops if isinstance(boundary_loops[0], (list, np.ndarray)) else [boundary_loops]:
        loop = np.array(loop)
        boundary_verts = vertices[loop]

        # Calculate perimeter
        perimeter = 0.0
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            perimeter += np.linalg.norm(boundary_verts[i] - boundary_verts[j])

        # Estimate hole area using boundary vertices
        center = np.mean(boundary_verts, axis=0)
        area = 0.0
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            # Area of triangle formed by center and edge
            v1 = boundary_verts[i] - center
            v2 = boundary_verts[j] - center
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))

        holes.append(HoleInfo(
            boundary_indices=loop,
            boundary_vertices=boundary_verts,
            area=area,
            perimeter=perimeter,
            center=center
        ))

    return holes


def _advancing_front_fill(vertices: np.ndarray, faces: np.ndarray,
                          hole: HoleInfo) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill a hole using advancing front method.

    This creates new vertices and faces by iteratively connecting boundary vertices.
    """
    if not HAS_IGL:
        raise ImportError("libigl is required for advancing front fill")

    # Use libigl's hole filling
    filled_vertices, filled_faces = igl.fill_holes(vertices, faces)

    return filled_vertices, filled_faces


def _fair_fill(vertices: np.ndarray, faces: np.ndarray,
               hole: HoleInfo, iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill holes with smooth, fair surfaces using Laplacian smoothing.

    This method creates a patch that minimizes bending energy.
    """
    # First, close the hole with a simple filling
    new_vertices = vertices.copy()
    new_faces = faces.copy().tolist()

    boundary = hole.boundary_indices
    n = len(boundary)

    if n < 3:
        return vertices, faces

    # For small holes, use simple fan triangulation from center
    if n <= 4:
        center_idx = len(new_vertices)
        new_vertices = np.vstack([new_vertices, hole.center.reshape(1, 3)])

        for i in range(n):
            j = (i + 1) % n
            new_faces.append([boundary[i], boundary[j], center_idx])
    else:
        # For larger holes, create internal vertices
        # Use a radial pattern with multiple rings
        num_rings = max(1, int(np.sqrt(n / 3)))
        new_face_list = []

        # Create rings of vertices
        rings = [boundary]
        for ring_idx in range(num_rings):
            t = (ring_idx + 1) / (num_rings + 1)
            ring_verts = []
            ring_indices = []

            prev_ring = rings[-1]
            for i in range(len(prev_ring)):
                # Interpolate towards center
                vert = (1 - t) * new_vertices[prev_ring[i]] + t * hole.center
                new_idx = len(new_vertices)
                new_vertices = np.vstack([new_vertices, vert.reshape(1, 3)])
                ring_indices.append(new_idx)
            rings.append(np.array(ring_indices))

        # Create center vertex
        center_idx = len(new_vertices)
        new_vertices = np.vstack([new_vertices, hole.center.reshape(1, 3)])

        # Connect rings with triangles
        for ring_idx in range(len(rings) - 1):
            outer_ring = rings[ring_idx]
            inner_ring = rings[ring_idx + 1]

            for i in range(len(outer_ring)):
                j = (i + 1) % len(outer_ring)
                k = i % len(inner_ring)
                l = (i + 1) % len(inner_ring)

                new_faces.append([outer_ring[i], outer_ring[j], inner_ring[k]])
                if k != l:
                    new_faces.append([outer_ring[j], inner_ring[l], inner_ring[k]])

        # Connect innermost ring to center
        innermost = rings[-1]
        for i in range(len(innermost)):
            j = (i + 1) % len(innermost)
            new_faces.append([innermost[i], innermost[j], center_idx])

    new_faces = np.array(new_faces, dtype=np.int32)

    # Apply Laplacian smoothing to the filled region
    if HAS_IGL and iterations > 0:
        # Identify vertices to smooth (new vertices only)
        smooth_mask = np.zeros(len(new_vertices), dtype=bool)
        smooth_mask[len(vertices):] = True

        # Apply Laplacian smoothing
        for _ in range(iterations):
            new_vertices = _laplacian_smooth_selective(new_vertices, new_faces, smooth_mask)

    return new_vertices, new_faces


def _laplacian_smooth_selective(vertices: np.ndarray, faces: np.ndarray,
                                mask: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """
    Apply Laplacian smoothing only to selected vertices.

    Args:
        vertices: Vertex array
        faces: Face array
        mask: Boolean mask indicating which vertices to smooth
        weight: Smoothing weight (0-1)
    """
    smoothed = vertices.copy()

    # Build adjacency
    adjacency = {}
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)

    # Smooth masked vertices
    for i in range(len(vertices)):
        if mask[i] and i in adjacency:
            neighbors = adjacency[i]
            if len(neighbors) > 0:
                centroid = np.mean(vertices[neighbors], axis=0)
                smoothed[i] = (1 - weight) * vertices[i] + weight * centroid

    return smoothed


def _poisson_fill(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill holes using Poisson surface reconstruction.

    This is most effective when the mesh has good normal information.
    """
    if not HAS_PYMESHLAB:
        raise ImportError("pymeshlab is required for Poisson reconstruction. "
                         "Install with: pip install pymeshlab")

    ms = pymeshlab.MeshSet()

    # Create pymeshlab mesh
    m = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(m)

    # Compute normals if not present
    ms.compute_normal_for_point_clouds()

    # Apply Poisson reconstruction
    ms.generate_surface_reconstruction_screened_poisson(
        depth=10,
        pointweight=4.0,
        threads=4
    )

    # Get reconstructed mesh
    mesh = ms.current_mesh()
    new_vertices = mesh.vertex_matrix()
    new_faces = mesh.face_matrix()

    return new_vertices, new_faces


def _context_aware_fill(vertices: np.ndarray, faces: np.ndarray,
                       hole: HoleInfo, k_neighbors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill holes using local surface context (normals, curvature).

    This method analyzes the surrounding surface to create a patch that
    better respects the underlying geometry.
    """
    # Get boundary normals by analyzing adjacent faces
    boundary_normals = np.zeros((len(hole.boundary_indices), 3))

    # Build vertex-to-face mapping
    vert_to_faces = {}
    for face_idx, face in enumerate(faces):
        for v in face:
            if v not in vert_to_faces:
                vert_to_faces[v] = []
            vert_to_faces[v].append(face_idx)

    # Calculate boundary vertex normals
    for i, v_idx in enumerate(hole.boundary_indices):
        if v_idx in vert_to_faces:
            face_normals = []
            for f_idx in vert_to_faces[v_idx]:
                face = faces[f_idx]
                v0, v1, v2 = vertices[face]
                normal = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    face_normals.append(normal / norm)

            if len(face_normals) > 0:
                boundary_normals[i] = np.mean(face_normals, axis=0)
                boundary_normals[i] /= np.linalg.norm(boundary_normals[i])

    # Estimate average normal for the hole
    hole_normal = np.mean(boundary_normals, axis=0)
    if np.linalg.norm(hole_normal) > 0:
        hole_normal /= np.linalg.norm(hole_normal)
    else:
        hole_normal = np.array([0, 0, 1])

    # Use fair filling with normal-aware smoothing
    new_vertices, new_faces = _fair_fill(vertices, faces, hole, iterations=30)

    # Project new vertices slightly along the estimated normal direction
    # to better match the surface
    for i in range(len(vertices), len(new_vertices)):
        # Find nearest boundary point
        dists = np.linalg.norm(hole.boundary_vertices - new_vertices[i], axis=1)
        nearest_idx = np.argmin(dists)

        # Blend with normal direction
        offset = hole_normal * dists[nearest_idx] * 0.1
        new_vertices[i] += offset

    return new_vertices, new_faces


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

    # Detect holes
    num_holes = 0
    if HAS_IGL:
        try:
            holes = detect_holes(vertices, faces)
            num_holes = len(holes)
            if num_holes > 0:
                issues.append(f"Found {num_holes} hole(s) in the mesh")
        except Exception as e:
            logging.warning(f"Could not detect holes: {e}")

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
                method: str = "context_aware",
                smooth_iterations: int = 10,
                validate: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[MeshValidationResult]]:
    """
    Inpaint holes in a triangle mesh.

    Args:
        mesh_path: Path to input .obj mesh file with holes
        output_path: Optional path to save the inpainted mesh
        method: Inpainting method to use. Options:
            - "context_aware": Use local surface information (default, best quality)
            - "fair": Minimal surface with Laplacian smoothing (good for clean holes)
            - "advancing_front": Quick iterative filling (fast, lower quality)
            - "poisson": Poisson reconstruction (requires pymeshlab)
        smooth_iterations: Number of smoothing iterations for "fair" method
        validate: Whether to validate the mesh after inpainting

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        validation: MeshValidationResult if validate=True, else None

    Example:
        >>> from vesuvius.image_proc.mesh.inpaint import inpaint_mesh
        >>> vertices, faces, validation = inpaint_mesh(
        ...     "surface_with_holes.obj",
        ...     "surface_inpainted.obj",
        ...     method="context_aware"
        ... )
        >>> print(f"Filled {validation.num_holes} holes")
    """
    # Load mesh
    vertices, faces = load_mesh(mesh_path)

    logging.info(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")

    # Detect holes
    holes = detect_holes(vertices, faces)
    logging.info(f"Detected {len(holes)} hole(s)")

    if len(holes) == 0:
        logging.info("No holes detected, mesh is already complete")
        if output_path:
            save_mesh(vertices, faces, output_path)
        if validate:
            validation_result = validate_mesh(vertices, faces)
            return vertices, faces, validation_result
        return vertices, faces, None

    # Fill holes using selected method
    new_vertices, new_faces = vertices, faces

    if method == "advancing_front":
        if not HAS_IGL:
            raise ImportError("libigl required for advancing_front method")
        logging.info("Using advancing front method...")
        new_vertices, new_faces = igl.fill_holes(vertices, faces)

    elif method == "fair":
        logging.info(f"Using fair filling method with {smooth_iterations} iterations...")
        for hole in holes:
            new_vertices, new_faces = _fair_fill(new_vertices, new_faces, hole,
                                                 iterations=smooth_iterations)

    elif method == "poisson":
        logging.info("Using Poisson reconstruction method...")
        new_vertices, new_faces = _poisson_fill(vertices, faces)

    elif method == "context_aware":
        logging.info("Using context-aware filling method...")
        for hole in holes:
            logging.info(f"  Filling hole with {len(hole.boundary_indices)} boundary vertices, "
                        f"area ~{hole.area:.2f}, perimeter ~{hole.perimeter:.2f}")
            new_vertices, new_faces = _context_aware_fill(new_vertices, new_faces, hole)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: "
                        "context_aware, fair, advancing_front, poisson")

    logging.info(f"Inpainting complete: {len(new_vertices)} vertices, {len(new_faces)} faces")

    # Compute normals
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(new_faces)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # Save if output path provided
    if output_path:
        save_mesh(new_vertices, new_faces, output_path, normals)
        logging.info(f"Saved inpainted mesh to {output_path}")

    # Validate result
    validation_result = None
    if validate:
        validation_result = validate_mesh(new_vertices, new_faces)
        logging.info(f"Validation: watertight={validation_result.is_watertight}, "
                    f"holes={validation_result.num_holes}")
        if validation_result.issues:
            logging.warning("Validation issues found:")
            for issue in validation_result.issues:
                logging.warning(f"  - {issue}")

    return new_vertices, new_faces, validation_result

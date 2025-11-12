#!/usr/bin/env python3
"""
CLI tool for mesh hole inpainting using PyMeshLab's hole filling.

Usage:
    vesuvius.inpaint_mesh input.obj output.obj
    vesuvius.inpaint_mesh input.obj --validate-only
"""

import argparse
import logging
import sys
from pathlib import Path

from vesuvius.image_proc.mesh import inpaint_mesh, validate_mesh
import pymeshlab


def main():
    parser = argparse.ArgumentParser(
        description="Inpaint holes in triangle meshes using PyMeshLab's hole filling. "
                    "This method uses MeshLab's robust algorithms to fill holes in mesh geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inpaint mesh with holes
  vesuvius.inpaint_mesh surface_holes.obj surface_filled.obj

  # Only validate without inpainting
  vesuvius.inpaint_mesh input.obj --validate-only

  # Verbose output with custom smoothing
  vesuvius.inpaint_mesh input.obj output.obj --verbose --smoothing-steps 5

  # Fill only small holes and simplify result
  vesuvius.inpaint_mesh input.obj output.obj --max-hole-size 1000 --simplify
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input .obj mesh file with holes"
    )

    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output .obj mesh file (optional if --validate-only)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the mesh without inpainting"
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after inpainting"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--max-hole-size",
        type=int,
        default=1000000,
        help="Maximum hole size (edge count) to fill (default: 1000000 for all holes)"
    )

    parser.add_argument(
        "--smoothing-steps",
        type=int,
        default=3,
        help="Number of Laplacian smoothing steps (default: 3, use 0 to skip)"
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the mesh after hole filling"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s"
    )

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if not input_path.suffix.lower() == '.obj':
        print(f"Error: Input must be a .obj file, got: {input_path.suffix}", file=sys.stderr)
        return 1

    # Validate-only mode
    if args.validate_only:
        print(f"Validating mesh: {input_path}")

        try:
            # Use PyMeshLab to load the mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(input_path))
            current_mesh = ms.current_mesh()

            vertices = current_mesh.vertex_matrix()
            faces = current_mesh.face_matrix()

            print(f"  Loaded: {len(vertices)} vertices, {len(faces)} faces")

            # Validate using Open3D
            validation = validate_mesh(vertices, faces)
            print(f"\nValidation Results:")
            print(f"  Watertight: {validation.is_watertight}")
            print(f"  Self-intersections: {validation.has_self_intersections}")
            print(f"  Degenerate triangles: {validation.has_degenerate_triangles}")
            print(f"  Has normals: {validation.has_normals}")
            print(f"  Number of holes: {validation.num_holes}")
            print(f"  Bounding box volume: {validation.bounding_box_volume:.2f}")

            if validation.issues:
                print(f"\n  Issues found:")
                for issue in validation.issues:
                    print(f"    - {issue}")

            return 0

        except Exception as e:
            print(f"Error during validation: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Check output path
    if not args.output:
        print("Error: Output path required (or use --validate-only)", file=sys.stderr)
        parser.print_help()
        return 1

    output_path = Path(args.output)
    if output_path.exists():
        response = input(f"Output file exists: {output_path}. Overwrite? [y/N] ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 0

    # Run inpainting
    print(f"Inpainting mesh: {input_path}")
    print(f"  Method: PyMeshLab hole filling")
    print(f"  Max hole size: {args.max_hole_size}")
    print(f"  Smoothing steps: {args.smoothing_steps}")
    print(f"  Simplify: {args.simplify}")
    print(f"  Output: {output_path}")

    try:
        vertices, faces, validation = inpaint_mesh(
            mesh_path=input_path,
            output_path=output_path,
            validate=not args.no_validate,
            max_hole_size=args.max_hole_size,
            smoothing_steps=args.smoothing_steps,
            simplify=args.simplify
        )

        print(f"\nSuccess!")
        print(f"  Output vertices: {len(vertices)}")
        print(f"  Output faces: {len(faces)}")

        if validation:
            print(f"\nValidation:")
            print(f"  Watertight: {validation.is_watertight}")
            print(f"  Remaining holes: {validation.num_holes}")

            if validation.issues:
                print(f"  Issues:")
                for issue in validation.issues:
                    print(f"    - {issue}")

        return 0

    except Exception as e:
        print(f"\nError during inpainting: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

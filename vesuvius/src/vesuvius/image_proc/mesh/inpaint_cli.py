#!/usr/bin/env python3
"""
CLI tool for mesh hole inpainting.

Usage:
    vesuvius.inpaint_mesh input.obj output.obj --method context_aware
    vesuvius.inpaint_mesh input.obj output.obj --method fair --smooth-iterations 50
"""

import argparse
import logging
import sys
from pathlib import Path

from vesuvius.image_proc.mesh import inpaint_mesh, validate_mesh, detect_holes, load_mesh


def main():
    parser = argparse.ArgumentParser(
        description="Inpaint holes in triangle meshes from surface tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inpaint using context-aware method (best quality)
  vesuvius.inpaint_mesh surface_holes.obj surface_filled.obj

  # Use fair filling with more smoothing
  vesuvius.inpaint_mesh input.obj output.obj --method fair --smooth-iterations 100

  # Quick filling for testing
  vesuvius.inpaint_mesh input.obj output.obj --method advancing_front

  # Only validate without inpainting
  vesuvius.inpaint_mesh input.obj --validate-only

Methods:
  context_aware   - Uses local surface normals and curvature (default, best)
  fair            - Minimal surface with Laplacian smoothing
  advancing_front - Quick iterative filling (requires libigl)
  poisson         - Poisson surface reconstruction (requires pymeshlab)
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
        "--method",
        type=str,
        choices=["context_aware", "fair", "advancing_front", "poisson"],
        default="context_aware",
        help="Inpainting method (default: context_aware)"
    )

    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=10,
        help="Number of smoothing iterations for 'fair' method (default: 10)"
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
            vertices, faces = load_mesh(input_path)
            print(f"  Loaded: {len(vertices)} vertices, {len(faces)} faces")

            # Detect holes
            holes = detect_holes(vertices, faces)
            print(f"  Holes detected: {len(holes)}")
            for i, hole in enumerate(holes):
                print(f"    Hole {i+1}: {len(hole.boundary_indices)} boundary vertices, "
                      f"area ~{hole.area:.2f}, perimeter ~{hole.perimeter:.2f}")

            # Validate
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
    print(f"  Method: {args.method}")
    print(f"  Output: {output_path}")

    try:
        vertices, faces, validation = inpaint_mesh(
            mesh_path=input_path,
            output_path=output_path,
            method=args.method,
            smooth_iterations=args.smooth_iterations,
            validate=not args.no_validate
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

"""Run inside Blender with --python, passing input and output after --."""

import sys
from pathlib import Path

import bpy


def has_faces():
    return any(obj.type == "MESH" and len(obj.data.polygons) for obj in bpy.context.scene.objects)


def main():
    source, destination = map(Path, sys.argv[sys.argv.index("--") + 1 :])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=str(source))
    if not has_faces():
        raise ValueError(f"PLY contains no mesh surface: {source}")
    bpy.ops.export_scene.gltf(filepath=str(destination), export_format="GLB")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(destination))
    if not has_faces():
        raise ValueError(f"Exported GLB contains no editable mesh surface: {destination}")


if __name__ == "__main__":
    main()

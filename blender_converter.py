"""
blender_converter.py
Translate XMVP/Nicotime data into Blender 3D Scenes.
"""
import sys
import os
import logging
import random
import glob
import math
from pathlib import Path
import xml.etree.ElementTree as ET

# Attempt extraction of shared logic
# Removed mvp_shared import to avoid dependency hell inside Blender
# We will pass raw data (dicts) instead of pydantic models.
# from mvp_shared import load_manifest, Manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- BPY IMPORT HANDLER ---
try:
    import bpy
    import mathutils
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False
    logging.warning("⚠️ 'bpy' module not found. Blender operations will fail if executed.")

def check_bpy():
    if not BPY_AVAILABLE:
        raise ImportError("Blender Python (bpy) is not installed. Please install it or run within Blender.")

def clear_scene():
    """Wipes the current scene clean."""
    check_bpy()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Purge orphan data
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0: bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)

def create_material(name, hex_color=None):
    """Creates a simple material."""
    check_bpy()
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    if hex_color:
        # crude hex to rgb
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], 1)
    else:
        # Random colorful
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
            
    return mat

def load_nicotime_entities(nicotime_dir: str):
    """Parses all XMLs in directory and returns entity dicts."""
    entities = []
    path = Path(nicotime_dir)
    if not path.exists():
        logging.warning(f"Nicotime dir not found: {nicotime_dir}")
        return []
        
    for f in path.glob("*.xml"):
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = root.find("Noosphere")
            if ns:
                for ent in ns.findall("Entity"):
                    e_data = {
                        "name": ent.findtext("Name"),
                        "type": ent.get("type"),
                        "vibe": ent.findtext("VisualSemiotics")
                    }
                    entities.append(e_data)
        except Exception as e:
            logging.warning(f"Failed to parse {f}: {e}")
    return entities

def build_library(nicotime_dir: str, out_blend_path: str):
    """
    Generates a library.blend file containing objects for all Nicotime entities.
    """
    check_bpy()
    logging.info(f"🏗️ Building Asset Library from {nicotime_dir}...")
    
    # Reset
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    entities = load_nicotime_entities(nicotime_dir)
    if not entities:
        logging.warning("No entities found. Creating Empty Library.")
    
    # Grid layout for library (for visual inspection)
    grid_size = int(math.ceil(math.sqrt(len(entities)))) if entities else 1
    spacing = 5.0
    
    for i, ent in enumerate(entities):
        name = ent['name']
        etype = ent['type']
        
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        
        # Create Primitive based on Type
        if etype == "Vibe":
            bpy.ops.mesh.primitive_ico_sphere_add(location=(x, y, 0), radius=1)
        elif etype == "Object":
            bpy.ops.mesh.primitive_cube_add(location=(x, y, 0), size=2)
        elif etype == "Social":
            bpy.ops.mesh.primitive_monkey_add(location=(x, y, 1)) # Suzanne represents people
        else:
            bpy.ops.mesh.primitive_cone_add(location=(x, y, 0))
            
        obj = bpy.context.active_object
        obj.name = name
        
        # Apply Material
        mat = create_material(f"Mat_{name}")
        obj.data.materials.append(mat)
        
        # Add Text Label
        bpy.ops.object.text_add(location=(x, y-1.5, 0))
        txt = bpy.context.active_object
        txt.data.body = name
        txt.name = f"LBL_{name}"
        txt.scale = (0.5, 0.5, 0.5)

    # Save
    logging.info(f"💾 Saving Library: {out_blend_path}")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend_path)

def setup_scene_for_shot(shot_data, library_path):
    """
    Sets up a scene for a specific shot by linking objects from the library.
    """
    check_bpy()
    
    # Load Library
    # Logic: We are in a fresh file or existing file?
    # Usually we start fresh for each shot to avoid clutter, or clear scene.
    clear_scene()
    
    # Create Camera
    bpy.ops.object.camera_add(location=(0, -10, 5), rotation=(math.radians(60), 0, 0))
    cam = bpy.context.active_object
    cam.name = "ShotCamera"
    bpy.context.scene.camera = cam
    
    # Append Objects mentioned in Prompt?
    # This requires NLP mapping which is hard in pure python without the TextEngine.
    # For now, let's just append random 3 objects from library for "chaos" visualization :)
    
    if os.path.exists(library_path):
        with bpy.data.libraries.load(library_path) as (data_from, data_to):
            # Potential objects
            all_objs = data_from.objects
            # Pick 3 random
            if all_objs:
                selection = random.sample(all_objs, min(3, len(all_objs)))
                data_to.objects = selection
        
        # Link them to scene
        for obj in data_to.objects:
            if obj:
                bpy.context.collection.objects.link(obj)
                # Randomize placement
                obj.location = (random.uniform(-5, 5), random.uniform(-5, 5), 0)

def render_frame(out_path, resolution_x=720, resolution_y=480):
    check_bpy()
    scene = bpy.context.scene
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-lib", help="Nicotime XML Dir")
    parser.add_argument("--out", help="Output .blend file")
    args = parser.parse_args()
    
    if args.build_lib and args.out:
        if BPY_AVAILABLE:
            build_library(args.build_lib, args.out)
        else:
            print("BPY not found.")

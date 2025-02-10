

# python recipes_image_making_tool.py --texture-dir "D:/MC/bedrock-samples/resource_pack/textures,D:\MC\NiaServer-Extra\development_resource_packs\NiaServer-Extra-RP\textures" --recipes-dir "D:\MC\NiaServer-Extra\development_behavior_packs\NiaServer-Extra-BP\recipes" --output-dir "./output"


import cv2
import numpy as np
import os
import json
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="自动生成合成配方图片 (支持多目录纹理搜索)")
    parser.add_argument("--texture-dir", type=str, default="textures",
                        help="材质图片文件夹路径，多个目录请用逗号分隔，例如：'dir1,dir2,dir3'")
    parser.add_argument("--recipes-dir", type=str, default="recipes",
                        help="配方 JSON 文件夹路径（资源包配方位置），支持子文件夹")
    parser.add_argument("--output-dir", type=str, default="output_recipes",
                        help="生成的配方图片输出文件夹路径")
    args = parser.parse_args()
    # 将 --texture-dir 按逗号分割，去掉空格，生成列表
    args.texture_dir = [d.strip() for d in args.texture_dir.split(",")]
    return args


def load_texture_mappings(texture_dirs):
    mapping = {}
    for texture_dir in texture_dirs:
        for fname in ["item_texture.json", "terrain_texture.json"]:
            filepath = os.path.join(texture_dir, fname)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "texture_data" in data:
                        mapping.update(data["texture_data"])
                        print(f"[DEBUG] Loaded mapping from {fname} in {texture_dir}, {len(data['texture_data'])} entries")
                except Exception as e:
                    print(f"[DEBUG] Error loading {fname} from {texture_dir}: {e}")
    return mapping


def find_texture_file(filename, texture_dirs):
    """
    在多个 texture 目录中递归搜索指定文件，返回第一个找到的完整路径。
    texture_dirs 为一个目录列表。
    """
    for texture_dir in texture_dirs:
        matches = glob.glob(os.path.join(texture_dir, '**', filename), recursive=True)
        if matches:
            return matches[0]
    return None

def load_texture(item_id, texture_dirs, cell_size=None, texture_map=None):
    key = item_id.replace("mcnia:", "").replace("minecraft:", "").lower()
    rel_path = None
    if texture_map and key in texture_map:
        mapping_entry = texture_map[key]
        if "textures" in mapping_entry:
            rel_path = mapping_entry["textures"]

    filename = None
    filepath = None
    if rel_path:
        # 构造候选文件名列表
        candidates = [rel_path + ".png"]
        if rel_path.startswith("textures/"):
            candidates.append(rel_path[len("textures/"):] + ".png")
        for cand in candidates:
            filepath_candidate = find_texture_file(cand, texture_dirs)
            if filepath_candidate and os.path.exists(filepath_candidate):
                filename = cand
                filepath = filepath_candidate
                break
        if not filename:
            print(f"[DEBUG] Mapping for key '{key}' provided candidates {candidates} but none were found.")
    if not filename:
        filename = key + ".png"
        filepath = find_texture_file(filename, texture_dirs)
    
    if not filepath or not os.path.exists(filepath):
        print(f"[DEBUG] Texture file not found for item '{item_id}', expected at '{filepath}'. Using red placeholder.")
        if cell_size is None:
            cell_size = (64, 64)
        img = np.zeros((cell_size[1], cell_size[0], 4), dtype=np.uint8)
        img[:] = (0, 0, 255, 255)
        return img

    print(f"[DEBUG] Loaded texture for item '{item_id}' from '{filepath}'")
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[DEBUG] Failed to read image file '{filepath}'. Using red placeholder.")
        if cell_size is None:
            cell_size = (64, 64)
        img = np.zeros((cell_size[1], cell_size[0], 4), dtype=np.uint8)
        img[:] = (0, 0, 255, 255)
        return img

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        print(f"[DEBUG] Converted texture for item '{item_id}' to 4 channels.")

    if cell_size is not None:
        img = cv2.resize(img, cell_size, interpolation=cv2.INTER_AREA)
    return img



def determine_cell_size_from_recipe(recipe, texture_dir, texture_map):
    """
    根据配方中任一物品的纹理自动识别单元格尺寸（假定正方形）。
    优先检测 shaped 配方的 key，如果没有，再检测 shapeless 配方的 ingredients。
    """
    if "pattern" in recipe and "key" in recipe:
        for ch, value in recipe["key"].items():
            item_id = value.get("item", "")
            if item_id:
                tex = load_texture(item_id, texture_dir, None, texture_map)
                if tex is not None:
                    h, w = tex.shape[:2]
                    print(f"[DEBUG] Detected cell size from item '{item_id}': {w}x{h}")
                    return (w, h)
    if "ingredients" in recipe:
        ingredients = recipe["ingredients"]
        if ingredients:
            item_id = ingredients[0].get("item", "")
            if item_id:
                tex = load_texture(item_id, texture_dir, None, texture_map)
                if tex is not None:
                    h, w = tex.shape[:2]
                    print(f"[DEBUG] Detected cell size from item '{item_id}': {w}x{h}")
                    return (w, h)
    print("[DEBUG] Defaulting cell size to 64x64")
    return (64, 64)

def compose_shaped_recipe(recipe, output_path, texture_dir, cell_size, texture_map):
    pattern = recipe.get("pattern", [])
    key = recipe.get("key", {})
    result = recipe.get("result", {})

    grid_rows = len(pattern)
    grid_cols = max(len(row) for row in pattern) if grid_rows > 0 else 0
    grid_height = grid_rows * cell_size[1]
    grid_width = grid_cols * cell_size[0]

    grid_img = np.full((grid_height, grid_width, 4), 255, dtype=np.uint8)

    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            if ch == " ":
                continue
            if ch in key:
                item_id = key[ch].get("item", "")
                tex = load_texture(item_id, texture_dir, cell_size, texture_map)
                y = r * cell_size[1]
                x = c * cell_size[0]
                if tex.shape != (cell_size[1], cell_size[0], 4):
                    print(f"[DEBUG] Texture shape mismatch for item '{item_id}': got {tex.shape}, expected {(cell_size[1], cell_size[0], 4)}")
                grid_img[y:y+cell_size[1], x:x+cell_size[0]] = tex

    result_item = result.get("item", "")
    result_tex = load_texture(result_item, texture_dir, cell_size, texture_map)

    arrow_width = int(cell_size[0] * 0.625)
    arrow_img_small = np.full((cell_size[1], arrow_width, 4), 255, dtype=np.uint8)
    cv2.arrowedLine(arrow_img_small,
                    (5, cell_size[1] // 2),
                    (arrow_width - 5, cell_size[1] // 2),
                    (0, 0, 0, 255),
                    thickness=2)

    arrow_canvas = np.full((grid_height, arrow_width, 4), 255, dtype=np.uint8)
    result_canvas = np.full((grid_height, cell_size[0], 4), 255, dtype=np.uint8)
    offset_y_arrow = (grid_height - cell_size[1]) // 2
    offset_y_result = (grid_height - cell_size[1]) // 2
    arrow_canvas[offset_y_arrow:offset_y_arrow+cell_size[1], :] = arrow_img_small
    result_canvas[offset_y_result:offset_y_result+cell_size[1], :] = result_tex

    final_img = np.hstack([grid_img, arrow_canvas, result_canvas])
    cv2.imwrite(output_path, final_img)
    print("Saved shaped recipe image to", output_path)

def compose_shapeless_recipe(recipe, output_path, texture_dir, cell_size, texture_map):
    ingredients = recipe.get("ingredients", [])
    result = recipe.get("result", {})

    num = len(ingredients)
    if num == 0:
        return
    ingredients_img = np.full((cell_size[1], num * cell_size[0], 4), 255, dtype=np.uint8)

    for i, ing in enumerate(ingredients):
        item_id = ing.get("item", "")
        tex = load_texture(item_id, texture_dir, cell_size, texture_map)
        x = i * cell_size[0]
        ingredients_img[:, x:x+cell_size[0]] = tex

    result_item = result.get("item", "")
    result_tex = load_texture(result_item, texture_dir, cell_size, texture_map)

    arrow_width = int(cell_size[0] * 0.625)
    arrow_img = np.full((cell_size[1], arrow_width, 4), 255, dtype=np.uint8)
    cv2.arrowedLine(arrow_img,
                    (5, cell_size[1] // 2),
                    (arrow_width - 5, cell_size[1] // 2),
                    (0, 0, 0, 255),
                    thickness=2)

    final_img = np.hstack([ingredients_img, arrow_img, result_tex])
    cv2.imwrite(output_path, final_img)
    print("Saved shapeless recipe image to", output_path)

def process_recipe_file(recipe_file, output_dir, texture_dir, texture_map):
    with open(recipe_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 判断配方格式：Bedrock 行为包配方顶级键通常为 "minecraft:recipe_shaped" 或 "minecraft:recipe_shapeless"
    if "minecraft:recipe_shaped" in data:
        recipe = data["minecraft:recipe_shaped"]
        recipe_type = "shaped"
    elif "minecraft:recipe_shapeless" in data:
        recipe = data["minecraft:recipe_shapeless"]
        recipe_type = "shapeless"
    else:
        print("Unknown recipe type in", recipe_file)
        return

    cell_size = determine_cell_size_from_recipe(recipe, texture_dir, texture_map)
    base_name = os.path.splitext(os.path.basename(recipe_file))[0]
    output_path = os.path.join(output_dir, base_name + ".png")

    if recipe_type == "shaped":
        compose_shaped_recipe(recipe, output_path, texture_dir, cell_size, texture_map)
    else:
        compose_shapeless_recipe(recipe, output_path, texture_dir, cell_size, texture_map)

def process_recipes_folder(recipes_dir, output_dir, texture_dir, texture_map):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(recipes_dir):
        for file in files:
            if file.endswith(".json"):
                recipe_file = os.path.join(root, file)
                process_recipe_file(recipe_file, output_dir, texture_dir, texture_map)

if __name__ == "__main__":
    args = parse_args()
    texture_map = load_texture_mappings(args.texture_dir)
    process_recipes_folder(args.recipes_dir, args.output_dir, args.texture_dir, texture_map)

import numpy as np
import cv2
import math
import argparse

LABELS = {
    "t1": "由日田正口",
    "t2": "正月日口田田由",
    "t3": "月由田由口田正日正",
    "t4": "由口由田正月正口正田日",
    "t5": "日口由口月月日由正正田田正",
    "t6": "正口日日口月口由正正由田由田月",
    "t7": "田正正日日田口月月由由由口正正田口"
}
def algorithm_density(image, char_count=None, min_pixels_per_char=50,
                      margin_ratio=1.5, dark_text=True, min_scale=0.4,
                      bboxes=None, min_char_width=3, min_char_height=8):
    """
    Scale image based on text pixel density.

    Core idea (user's algorithm):
      1. Binarise and count foreground (text) pixels
      2. Compute average pixels per character from ground-truth char count
      3. Apply a margin to account for non-text foreground noise
      4. scale = sqrt(min_pixels_per_char / effective_pixels_per_char)
      5. Clamp with min_scale and bbox floor

    Returns:
        (scaled_image, scale, info_dict)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarise (assume dark text on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not dark_text:
        binary = cv2.bitwise_not(binary)

    text_pixels = int(np.sum(binary == 255))
    total_pixels = h * w
    text_ratio = text_pixels / total_pixels

    # Estimate char count from connected components when GT is not available
    if char_count is None or char_count <= 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)
                 if stats[i, cv2.CC_STAT_AREA] > 20]
        if areas:
            avg_area = np.mean(areas)
            if avg_area > 400:
                char_count = max(1, int(sum(areas) / 200))
            else:
                char_count = max(1, len(areas))
        else:
            char_count = 1

    # Compute density-based scale
    avg_pixels_per_char = text_pixels / char_count if char_count > 0 else total_pixels

    # Dynamic margin: denser text = more aggressive scaling
    if text_ratio > 0.3:
        dynamic_margin = margin_ratio * 0.8
    elif text_ratio > 0.1:
        dynamic_margin = margin_ratio * 1.0
    else:
        dynamic_margin = margin_ratio * 1.4

    effective_pixels_per_char = avg_pixels_per_char / dynamic_margin
    scale_density = 1.0
    if effective_pixels_per_char > 0:
        scale_density = math.sqrt(min_pixels_per_char / effective_pixels_per_char)

    # Bbox floor: protect very thin characters (punctuation etc.)
    scale_floor = 0.0
    if bboxes and len(bboxes) > 0:
        min_bbox_w = min(b[2] - b[0] for b in bboxes if b[2] > b[0])
        min_bbox_h = min(b[3] - b[1] for b in bboxes if b[3] > b[1])
        need_protect_w = min_bbox_w < min_char_width * 2
        need_protect_h = min_bbox_h < min_char_height * 2
        floor_w = (1.0 / min_bbox_w) if (min_bbox_w > 0 and need_protect_w) else 0.0
        floor_h = (min_char_height / 2 / min_bbox_h) if (min_bbox_h > 0 and need_protect_h) else 0.0
        scale_floor = max(floor_w, floor_h)

    # Fuse
    scale = max(scale_density, scale_floor)
    scale = max(scale, min_scale)
    scale = min(scale, 1.0)

    # Round to multiple of 32
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    scale_actual = new_w / w

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    info = {
        "algorithm": "density",
        "scale": round(scale_actual, 4),
        "new_size": (new_w, new_h),
        "text_pixels": text_pixels,
        "char_count": char_count,
        "avg_pixels_per_char": round(avg_pixels_per_char, 2),
        "text_ratio": round(text_ratio, 4),
    }
    return resized, scale_actual, info


def analyze_char_pixels(image_name):
    """
    Analyze average pixel count per character using LABELS.
    
    Args:
        image_name: Image filename (e.g., 't1.png')
    """
    import os
    
    image_path = os.path.join(os.path.dirname(__file__), image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    text_pixels = int(np.sum(binary == 255))
    
    # Get the key without extension (e.g., 't1' from 't1.png')
    key = os.path.splitext(image_name)[0]
    target_chars = LABELS.get(key, "")
    char_count = len(target_chars)
    
    avg_pixels = text_pixels / char_count if char_count > 0 else 0
    
    print(f"{image_name}: {char_count} chars, {text_pixels} total pixels, {avg_pixels:.2f} avg pixels/char")


def analyze_edit_distance(pred_strings):
    """
    Calculate edit distance ratio between predicted strings and LABELS using Levenshtein.ratio.
    
    Args:
        pred_strings: List of predicted strings for t1-t7
    """
    import Levenshtein
    
    for i, pred in enumerate(pred_strings):
        key = f"t{i+1}"
        gt = LABELS.get(key, "")
        ratio = Levenshtein.ratio(pred, gt)
        print(f"{key}: GT='{gt}' | Pred='{pred}' | Ratio={ratio:.2f}")


def show_usage():
    print("""
Usage: python test.py -m <mode> [options]

Modes:
  -m pixels              Analyze pixel density for t1-t7 images
  -m distance            Calculate edit distance between predictions and LABELS
                         (requires --pred argument)

Examples:
  # Analyze pixel density
  python test.py -m pixels

  # Calculate edit distance with predictions
  python test.py -m distance --pred "由日田正口" "正月日口田由" "月由田由口田正日正"

  # Show this help
  python test.py -h
""")


def main():
    parser = argparse.ArgumentParser(description="OCR Test Script")
    parser.add_argument("-m", "--mode", choices=["pixels", "distance"], 
                        help="Mode: 'pixels' for pixel analysis, 'distance' for edit distance")
    parser.add_argument("--pred", nargs="+", 
                        help="Predicted strings for edit distance calculation (for 'distance' mode)")
    parser.add_argument("-u", "--usage", action="store_true", help="Show usage examples")
    args = parser.parse_args()
    
    if args.usage:
        show_usage()
    elif args.mode == "pixels":
        for i in range(1, 8):
            image_name = f"t{i}.png"
            analyze_char_pixels(image_name)
    elif args.mode == "distance":
        if args.pred:
            analyze_edit_distance(args.pred)
        else:
            print("Please provide --pred for edit distance mode")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
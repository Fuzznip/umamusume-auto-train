from scenarios.base import ScenarioHandler
import core.config as config
import utils.constants as constants
import utils.device_action_wrapper as device_action
import json
import os
import re
import cv2
import numpy as np
import hashlib
import difflib
from collections import Counter, defaultdict
from core.ocr import extract_number, extract_text
from utils.screenshot import enhance_image_for_ocr
from utils.screenshot import are_screenshots_same
from utils.log import info, warning, debug, debug_window, args
from PIL import Image
from time import sleep, perf_counter

# Baseline Trackblazer fallback uses URA-compatible stat gain region until TB-specific extraction is implemented.
TRACKBLAZER_STAT_GAINS_BBOX = constants.add_tuple_elements(constants.GAME_WINDOW_BBOX, (-30, 667, -264, -378))
TRACKBLAZER_STAT_GAINS_REGION = constants.convert_xyxy_to_xywh(TRACKBLAZER_STAT_GAINS_BBOX)
TRACKBLAZER_STAT_GAINS_2_BBOX = constants.add_tuple_elements(TRACKBLAZER_STAT_GAINS_BBOX, (0, -35, 0, -35))
TRACKBLAZER_STAT_GAINS_2_REGION = constants.convert_xyxy_to_xywh(TRACKBLAZER_STAT_GAINS_2_BBOX)
TRACKBLAZER_ITEM_CATALOG_PATH = os.path.join("data", "trackblazer_items.json")
TRACKBLAZER_QTY_DIGIT_TEMPLATE_DIR = os.path.join("assets", "trackblazer", "qty_digits")
TRACKBLAZER_QTY_DIGIT_TEMPLATE_THRESHOLD = 0.45
TRACKBLAZER_QTY_DIGIT_TEMPLATE_HIGH_CONFIDENCE = 0.85
TRACKBLAZER_QTY_DIGIT_TEMPLATE_SCALES = (0.9, 1.0, 1.1, 1.2)
TRACKBLAZER_CLOSE_BTN_PATH = "assets/buttons/close_btn.png"
TRACKBLAZER_BACK_BTN_PATH = "assets/buttons/back_btn.png"
TRACKBLAZER_INVENTORY_BTN_PATH = "assets/trackblazer/training_items_btn.png"
TRACKBLAZER_INVENTORY_USE_ITEM_PATH = "assets/trackblazer/use_item.png"
TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_PATH = "assets/trackblazer/confirm_item_use.png"
TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_POSITION = (530, 1000)
TRACKBLAZER_SHOP_BTN_PATH = "assets/trackblazer/shop_btn.png"
TRACKBLAZER_CLIMAX_RACE_BTN_PATH = "assets/trackblazer/climax_race_btn.png"
TRACKBLAZER_SHOP_CHECKMARK_DISABLED_PATH = "assets/trackblazer/checkmark_disabled.png"
TRACKBLAZER_SHOP_CHECKMARK_SELECTED_PATH = "assets/trackblazer/checkmark_selected.png"
TRACKBLAZER_SHOP_PURCHASED_PATH = "assets/trackblazer/purchased.png"
TRACKBLAZER_CONFIRM_BTN_PATH = "assets/buttons/confirm_btn.png"
TRACKBLAZER_OK_BTN_PATH = "assets/buttons/ok_btn.png"
TRACKBLAZER_SHOP_COINS_FIXED_OCR_REGION_FROM_SCREEN = (555, 336, 627, 373)
TRACKBLAZER_TEMPLATE_SCALES = (1.0, 0.95, 1.05, 0.9, 1.1)
TRACKBLAZER_SHOP_TEMPLATE_SCALES = (1.0, 0.98, 1.02)
TRACKBLAZER_INVENTORY_TEMPLATE_SCALES = (1.0, 0.98, 1.02)
TRACKBLAZER_SHOP_ROW_ANCHOR_THRESHOLD = 0.82
TRACKBLAZER_SHOP_ROW_ANCHOR_SCALES = (1.0, 0.98, 1.02)
TRACKBLAZER_INVENTORY_QTY_MIN = 1
TRACKBLAZER_INVENTORY_QTY_MAX = 5
TRACKBLAZER_INVENTORY_QTY_RESOLVED_MAX = TRACKBLAZER_INVENTORY_QTY_MAX
TRACKBLAZER_MAX_REASONABLE_COST = 999
TRACKBLAZER_MAX_REASONABLE_SHOP_COINS = 9999
TRACKBLAZER_SHOP_SCROLL_START = (560, 700)
TRACKBLAZER_SHOP_SCROLL_END = (560, 580)  # exactly 1 row height (104px) per scroll
TRACKBLAZER_SHOP_NEW_ROW_ANCHOR_TOLERANCE = 14
TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_MAX = 48
TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_GAIN = 0.8
TRACKBLAZER_SHOP_REOPEN_ATTEMPTS = 4
TRACKBLAZER_SHOP_REENTRY_LOAD_WAIT = 1.0
TRACKBLAZER_SHOP_SCROLL_SETTLE_WAIT = 0.8
TRACKBLAZER_SHOP_BACK_BUTTON_POLL_INTERVAL = 1.0
TRACKBLAZER_SHOP_BACK_BUTTON_MAX_WAIT = 30.0
TRACKBLAZER_SHOP_SELECTION_EXHAUSTIVE_SCROLL_LIMIT = 80
TRACKBLAZER_INVENTORY_OPEN_LOAD_WAIT = 1.0
TRACKBLAZER_INVENTORY_SCROLL_SETTLE_WAIT = 3.0
TRACKBLAZER_INVENTORY_ROW_BUCKET_HEIGHT = 28
TRACKBLAZER_INVENTORY_MIN_ROW_CENTER_Y_RATIO = 0.06
TRACKBLAZER_INVENTORY_MAX_ROW_CENTER_Y_RATIO = 0.96
TRACKBLAZER_INVENTORY_QTY_MAX_ROW_CENTER_Y_RATIO = 0.92
TRACKBLAZER_INVENTORY_PANEL_LTRB = (111, 87, 691, 909)
TRACKBLAZER_INVENTORY_ROW_WIDTH = 550
TRACKBLAZER_INVENTORY_ROW_HEIGHT = 100
TRACKBLAZER_INVENTORY_USE_ITEM_THRESHOLD = 0.82
TRACKBLAZER_INVENTORY_USE_ITEM_SCALES = (1.0, 0.98, 1.02)
TRACKBLAZER_INVENTORY_QTY_CROP_LEFT_FROM_USE_ITEM = 290
TRACKBLAZER_INVENTORY_QTY_CROP_WIDTH = 74
TRACKBLAZER_INVENTORY_QTY_CROP_Y_PAD = 10
TRACKBLAZER_SHOP_PANEL_LTRB = (104, 396, 697, 805)
TRACKBLAZER_SHOP_ROW_BUCKET_HEIGHT = 28
TRACKBLAZER_SHOP_ROW_WIDTH = 620
TRACKBLAZER_SHOP_ROW_HEIGHT = 104
TRACKBLAZER_SHOP_ROW_ANCHOR_MIN_X_RATIO = 0.56
TRACKBLAZER_SHOP_ICON_MIN_X_RATIO = 0.01
TRACKBLAZER_SHOP_ICON_MAX_X_RATIO = 0.4
TRACKBLAZER_ITEM_CATALOG_CACHE = None
TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE = None

_TRACKBLAZER_OFFSET_APPLIED = False

def adjust_trackblazer_x_coords(offset):
  """Align trackblazer module-level coordinates with the active game window.

  Must be called AFTER constants.adjust_constants_x_coords (or instead of it
  for the Steam-native path, where constants receives no shift).

  Two coordinate categories exist in this module:

  1. BBOX/REGION values are derived from constants.GAME_WINDOW_BBOX at import
     time, so they already embed the Steam-default x=155 origin.  They need the
     same *delta* that was applied to constants (i.e. `offset`).

  2. Raw device-relative values (LTRB panels, scroll positions, click positions)
     are calibrated for game content at x=0.  They need to be shifted to the
     absolute game window origin, which is constants.GAME_WINDOW_BBOX[0] after
     constants has been updated.

  Do NOT call this for ADB (offset=-155): the BBOX delta would produce x=-30
  (outside device bounds) and the raw values are already valid at x=0-based
  device coordinates.
  """
  global _TRACKBLAZER_OFFSET_APPLIED
  if _TRACKBLAZER_OFFSET_APPLIED:
    return

  # Read game window origin AFTER constants has been updated by the caller.
  game_window_x = constants.GAME_WINDOW_BBOX[0]

  import sys
  g = sys.modules[__name__].__dict__

  for name, value in list(g.items()):
    if not isinstance(value, tuple):
      continue

    # --- BBOX/REGION: Steam-calibrated, apply the same delta as constants ---
    # (x, y, w, h) — shift x only
    if name.endswith("_REGION") and len(value) == 4:
      g[name] = (value[0] + offset, value[1], value[2], value[3])

    # (x1, y1, x2, y2) — shift both x coords
    elif name.endswith("_BBOX") and len(value) == 4:
      g[name] = (value[0] + offset, value[1], value[2] + offset, value[3])

    # --- Raw device-relative: shift to absolute game window origin ---
    # (left, top, right, bottom)
    elif (name.endswith("_LTRB") or name.endswith("_FROM_SCREEN")) and len(value) == 4:
      g[name] = (value[0] + game_window_x, value[1], value[2] + game_window_x, value[3])

    # (x, y)
    elif (
      name.endswith("_MOUSE_POS")
      or name.endswith("_POSITION")
      or name.endswith("_SCROLL_START")
      or name.endswith("_SCROLL_END")
    ) and len(value) == 2:
      g[name] = (value[0] + game_window_x, value[1])

  _TRACKBLAZER_OFFSET_APPLIED = True

def _dedup_boxes(boxes, min_dist=8):
  filtered = []
  for x, y, w, h in boxes:
    cx = x + w // 2
    cy = y + h // 2
    too_close = False
    for fx, fy, fw, fh in filtered:
      fcx = fx + fw // 2
      fcy = fy + fh // 2
      if abs(cx - fcx) <= min_dist and abs(cy - fcy) <= min_dist:
        too_close = True
        break
    if not too_close:
      filtered.append((x, y, w, h))
  return filtered


def _find_item_matches(image_path, screenshot, base_threshold, scales=None, use_grayscale=True):
  if scales is None:
    scales = TRACKBLAZER_TEMPLATE_SCALES
  all_matches = []

  for scale in scales:
    all_matches.extend(
      device_action.match_template(
        image_path,
        screenshot,
        threshold=base_threshold,
        template_scaling=scale,
      )
    )

  # Fallback to grayscale with a slightly lower threshold for dim/filtered inventory UI.
  if use_grayscale:
    gray_threshold = max(0.72, base_threshold - 0.08)
    for scale in scales:
      all_matches.extend(
        device_action.match_template(
          image_path,
          screenshot,
          threshold=gray_threshold,
          grayscale=True,
          template_scaling=scale,
        )
      )

  return _dedup_boxes(all_matches)


def _read_trackblazer_item_catalog_from_disk():
  if not os.path.exists(TRACKBLAZER_ITEM_CATALOG_PATH):
    warning(f"Trackblazer item catalog not found: {TRACKBLAZER_ITEM_CATALOG_PATH}")
    return []

  try:
    with open(TRACKBLAZER_ITEM_CATALOG_PATH, "r", encoding="utf-8") as f:
      payload = json.load(f)
  except (OSError, json.JSONDecodeError) as exc:
    warning(f"Failed to load Trackblazer item catalog: {exc}")
    return []

  items = payload.get("items", []) if isinstance(payload, dict) else payload
  if not isinstance(items, list):
    warning("Trackblazer item catalog malformed: 'items' must be a list.")
    return []
  return items


def _read_trackblazer_qty_digit_templates_from_disk():
  if not os.path.isdir(TRACKBLAZER_QTY_DIGIT_TEMPLATE_DIR):
    return {}

  templates = {}
  for qty in range(TRACKBLAZER_INVENTORY_QTY_MIN, TRACKBLAZER_INVENTORY_QTY_MAX + 1):
    image_path = os.path.join(TRACKBLAZER_QTY_DIGIT_TEMPLATE_DIR, f"{qty}.png")
    if not os.path.exists(image_path):
      continue
    template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
      warning(f"[TRACKBLAZER] Could not load qty template image: {image_path}")
      continue

    _, template_bin = cv2.threshold(template, 180, 255, cv2.THRESH_BINARY_INV)
    templates[qty] = template_bin

  if templates:
    debug(f"[TRACKBLAZER] Loaded quantity digit templates: {sorted(templates.keys())}")
  return templates


def preload_trackblazer_qty_digit_templates(force_reload=False):
  global TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE
  if TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE is not None and not force_reload:
    return TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE

  TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE = _read_trackblazer_qty_digit_templates_from_disk()
  return TRACKBLAZER_QTY_DIGIT_TEMPLATE_CACHE


def preload_trackblazer_item_catalog(force_reload=False):
  global TRACKBLAZER_ITEM_CATALOG_CACHE
  if TRACKBLAZER_ITEM_CATALOG_CACHE is not None and not force_reload:
    return TRACKBLAZER_ITEM_CATALOG_CACHE

  TRACKBLAZER_ITEM_CATALOG_CACHE = _read_trackblazer_item_catalog_from_disk()
  return TRACKBLAZER_ITEM_CATALOG_CACHE


def _extract_price_from_match_region(screenshot, match, fallback_cost):
  x, y, w, h = match
  image_h, image_w = screenshot.shape[:2]

  def _extract_valid_price_from_crop(crop):
    if crop is None or crop.size == 0:
      return None

    direct_number = extract_number(crop)
    if 0 < direct_number <= TRACKBLAZER_MAX_REASONABLE_COST:
      return int(direct_number)

    enhanced_crop = enhance_image_for_ocr(crop, resize_factor=3)
    enhanced_number = extract_number(enhanced_crop)
    if 0 < enhanced_number <= TRACKBLAZER_MAX_REASONABLE_COST:
      return int(enhanced_number)

    text = extract_text(enhanced_crop)
    for group in re.findall(r"\d{1,3}", str(text or "")):
      try:
        value = int(group)
      except ValueError:
        continue
      if 0 < value <= TRACKBLAZER_MAX_REASONABLE_COST:
        return value

    return None

  # Sale price may appear to the right of the normal cost.
  sale_left = min(max(x + w + 188, 0), image_w)
  sale_right = min(max(x + w + 320, 0), image_w)
  sale_top = min(max(y - 10, 0), image_h)
  sale_bottom = min(max(y + h + 20, 0), image_h)
  sale_crop = None
  if sale_right > sale_left and sale_bottom > sale_top:
    sale_crop = screenshot[sale_top:sale_bottom, sale_left:sale_right]
    debug_window(sale_crop, save_name="trackblazer_shop_sale_price_crop", force_save=True)
    debug_window(
      enhance_image_for_ocr(sale_crop, resize_factor=3),
      save_name="trackblazer_shop_sale_price_crop_enhanced",
      force_save=True,
    )

  # Shop price is shown in a dedicated "Cost <value>" field on the row.
  left = min(max(x + w + 28, 0), image_w)
  right = min(max(x + w + 235, 0), image_w)
  top = min(max(y - 10, 0), image_h)
  bottom = min(max(y + h + 20, 0), image_h)

  regular_price = int(fallback_cost)
  if right <= left or bottom <= top:
    right = left
    bottom = top

  price_crop = screenshot[top:bottom, left:right] if right > left and bottom > top else None
  if price_crop is not None and price_crop.size > 0:
    debug_window(price_crop, save_name="trackblazer_shop_regular_price_crop", force_save=True)
    debug_window(
      enhance_image_for_ocr(price_crop, resize_factor=3),
      save_name="trackblazer_shop_regular_price_crop_enhanced",
      force_save=True,
    )

    enhanced = enhance_image_for_ocr(price_crop, resize_factor=3)
    price_text = extract_text(enhanced)
    if price_text:
      cost_match = re.search(r"(?i)cost[^0-9]{0,6}(\d{1,3})", price_text)
      if cost_match:
        try:
          parsed_cost = int(cost_match.group(1))
          if 0 < parsed_cost <= TRACKBLAZER_MAX_REASONABLE_COST:
            regular_price = parsed_cost
        except ValueError:
          pass

    if regular_price <= 0:
      normal_price = _extract_valid_price_from_crop(price_crop)
      if normal_price is not None:
        regular_price = normal_price

  # Fallback to a tighter numeric slot immediately to the right of the Cost label.
  if regular_price <= 0:
    narrow_left = min(max(x + w + 118, 0), image_w)
    narrow_right = min(max(x + w + 188, 0), image_w)
    narrow_top = min(max(y - 2, 0), image_h)
    narrow_bottom = min(max(y + h + 10, 0), image_h)
    if narrow_right > narrow_left and narrow_bottom > narrow_top:
      narrow_crop = screenshot[narrow_top:narrow_bottom, narrow_left:narrow_right]
      detected_price = extract_number(narrow_crop)
      if 0 < detected_price <= TRACKBLAZER_MAX_REASONABLE_COST:
        regular_price = int(detected_price)

  if regular_price <= 0:
    regular_price = int(fallback_cost)

  if sale_crop is not None and sale_crop.size > 0 and regular_price > 0:
    sale_price = _extract_valid_price_from_crop(sale_crop)
    if sale_price is not None and sale_price < regular_price:
      expected_floor = (regular_price * 9) // 10
      expected_ceil = (regular_price * 9 + 9) // 10
      if sale_price in {expected_floor, expected_ceil}:
        return int(sale_price)
      debug(
        f"[TRACKBLAZER] Ignoring sale OCR value {sale_price} for regular price {regular_price}: "
        "not a valid 10% discount."
      )

  return int(regular_price)


def _extract_shop_coin_count(threshold=0.85):
  total_started_at = perf_counter()
  _ = threshold  # kept for backward-compatible signature

  left, top, right, bottom = TRACKBLAZER_SHOP_COINS_FIXED_OCR_REGION_FROM_SCREEN
  region_w = right - left
  region_h = bottom - top
  if region_w <= 0 or region_h <= 0:
    warning("[TRACKBLAZER] shop coin OCR region is invalid.")
    _log_shop_timing("shop coins OCR total", total_started_at, "invalid_fixed_region")
    return None

  full_screen = device_action.screenshot(region_ltrb=constants.FULL_SCREEN_LANDSCAPE)
  image_h, image_w = full_screen.shape[:2]

  c_left = min(max(left, 0), image_w)
  c_right = min(max(right, 0), image_w)
  c_top = min(max(top, 0), image_h)
  c_bottom = min(max(bottom, 0), image_h)
  if c_right <= c_left or c_bottom <= c_top:
    warning("[TRACKBLAZER] shop coin OCR crop resolved outside screen bounds.")
    _log_shop_timing("shop coins OCR total", total_started_at, "out_of_bounds")
    return None

  crop = full_screen[c_top:c_bottom, c_left:c_right]
  if crop.size == 0:
    warning("[TRACKBLAZER] shop coin OCR crop is empty.")
    _log_shop_timing("shop coins OCR total", total_started_at, "empty_crop")
    return None

  # Always persist coin OCR crops for troubleshooting misreads.
  debug_window(crop, save_name="trackblazer_shop_coins_crop_raw", force_save=True)

  enhanced = enhance_image_for_ocr(crop, resize_factor=4)
  debug_window(enhanced, save_name="trackblazer_shop_coins_crop_enhanced", force_save=True)

  # Alternative high-contrast path for coin OCR when generic enhancement over-binarizes.
  gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
  sobel_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
  sobel_mag = cv2.magnitude(sobel_x, sobel_y)
  sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
  sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
  sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
  high_contrast = cv2.addWeighted(sharpened, 0.80, sobel_norm, 0.55, 0)
  high_contrast = cv2.resize(high_contrast, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
  _, high_contrast_bin = cv2.threshold(
    high_contrast,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
  )

  debug_window(gray, save_name="trackblazer_shop_coins_crop_gray", force_save=True)
  debug_window(clahe, save_name="trackblazer_shop_coins_crop_clahe", force_save=True)
  debug_window(sobel_norm, save_name="trackblazer_shop_coins_crop_sobel", force_save=True)
  debug_window(high_contrast, save_name="trackblazer_shop_coins_crop_hicontrast", force_save=True)
  debug_window(high_contrast_bin, save_name="trackblazer_shop_coins_crop_hicontrast_bin", force_save=True)

  ocr_variants = [
    ("enhanced", enhanced),
    ("raw", Image.fromarray(crop)),
    ("hicontrast", Image.fromarray(high_contrast)),
    ("hicontrast_bin", Image.fromarray(high_contrast_bin)),
  ]

  candidates = []
  variant_votes = []
  weighted_variant_votes = []
  raw_variant_vote = None
  ocr_digit_texts = []

  for variant_name, image in ocr_variants:
    variant_candidates = []
    number = extract_number(image)
    if 0 < number <= TRACKBLAZER_MAX_REASONABLE_SHOP_COINS:
      candidates.append(int(number))
      variant_candidates.append(int(number))

    text_digits = extract_text(image, allowlist="0123456789")
    ocr_digit_texts.append(f"{variant_name}:{str(text_digits).strip()}")
    for group in re.findall(r"\d{1,4}", text_digits):
      try:
        value = int(group)
      except ValueError:
        continue
      if 0 < value <= TRACKBLAZER_MAX_REASONABLE_SHOP_COINS:
        candidates.append(value)
        variant_candidates.append(value)

    if variant_candidates:
      local_counts = Counter(variant_candidates)
      local_vote = sorted(local_counts.keys(), key=lambda value: (-local_counts[value], value))[0]
      variant_votes.append(local_vote)
      if variant_name == "raw":
        raw_variant_vote = local_vote
        weighted_variant_votes.extend([local_vote, local_vote])
      else:
        weighted_variant_votes.append(local_vote)

  detected = None
  if weighted_variant_votes:
    vote_counts = Counter(weighted_variant_votes)
    highest_votes = max(vote_counts.values())
    tied_values = [value for value, count in vote_counts.items() if count == highest_votes]
    if raw_variant_vote is not None and raw_variant_vote in tied_values:
      detected = raw_variant_vote
    else:
      detected = sorted(tied_values)[0]
  elif candidates:
    counts = Counter(candidates)
    detected = sorted(counts.keys(), key=lambda value: (-counts[value], value))[0]

  debug(
    "[TRACKBLAZER] shop coins OCR fixed region: "
    f"ltrb=({c_left}, {c_top}, {c_right}, {c_bottom}), size={c_right - c_left}x{c_bottom - c_top}, "
    f"coins={detected if detected is not None else 'unknown'}, "
    f"variant_votes={variant_votes}, "
    f"weighted_variant_votes={weighted_variant_votes}, "
    f"coin_candidates={sorted(set(candidates)) if candidates else []}, "
    f"ocr_digit_texts={ocr_digit_texts}"
  )

  if detected is None:
    _log_shop_timing("shop coins OCR total", total_started_at, "no_numeric_candidate")
    return None

  _log_shop_timing("shop coins OCR total", total_started_at)
  return int(detected)


def _is_inventory_quantity_resolved(quantity):
  value = int(quantity)
  return TRACKBLAZER_INVENTORY_QTY_MIN <= value <= TRACKBLAZER_INVENTORY_QTY_RESOLVED_MAX


def _is_inventory_row_well_positioned(screenshot, match):
  image_h, _ = screenshot.shape[:2]
  _, y, _, h = match
  center_y = y + h // 2
  min_y = int(image_h * TRACKBLAZER_INVENTORY_MIN_ROW_CENTER_Y_RATIO)
  max_y = int(image_h * TRACKBLAZER_INVENTORY_MAX_ROW_CENTER_Y_RATIO)
  return min_y <= center_y <= max_y


def _is_inventory_quantity_row_reliable(screenshot, match):
  image_h, _ = screenshot.shape[:2]
  _, y, _, h = match
  center_y = y + h // 2
  min_y = int(image_h * TRACKBLAZER_INVENTORY_MIN_ROW_CENTER_Y_RATIO)
  max_y = int(image_h * TRACKBLAZER_INVENTORY_QTY_MAX_ROW_CENTER_Y_RATIO)
  return min_y <= center_y <= max_y


def _match_quantity_from_digit_templates(crop, item_name="unknown_item"):
  templates = preload_trackblazer_qty_digit_templates()
  if not templates:
    return None, -1.0

  if crop is None or crop.size == 0:
    return None, -1.0

  gray = crop
  if len(crop.shape) == 3:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

  _, crop_bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

  best_qty = None
  best_score = -1.0
  crop_h, crop_w = crop_bin.shape[:2]
  for qty, template in templates.items():
    th, tw = template.shape[:2]
    for scale in TRACKBLAZER_QTY_DIGIT_TEMPLATE_SCALES:
      scaled_w = max(1, int(tw * scale))
      scaled_h = max(1, int(th * scale))
      if scaled_w > crop_w or scaled_h > crop_h:
        continue

      scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
      result = cv2.matchTemplate(crop_bin, scaled_template, cv2.TM_CCOEFF_NORMED)
      _, max_val, _, _ = cv2.minMaxLoc(result)
      if max_val > best_score:
        best_score = float(max_val)
        best_qty = int(qty)

  if best_qty is None:
    return None, -1.0

  if args.debug is not None:
    info(f"[TRACKBLAZER] Qty template match for '{item_name}': qty={best_qty}, score={best_score:.3f}")

  if best_score >= TRACKBLAZER_QTY_DIGIT_TEMPLATE_THRESHOLD:
    return best_qty, best_score
  return None, best_score


def _match_name_hint_from_item(item):
  hint = item.get("match_name_hint")
  if isinstance(hint, str) and hint.strip():
    return hint.strip().lower()

  name = item.get("name", "")
  tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", name)]
  if not tokens:
    return ""
  return tokens[0]


def _candidate_items_from_row_ocr(row_text, item_catalog, max_candidates=6):
  normalized = (row_text or "").lower()
  if not normalized:
    return []

  primary_segment = re.split(r"(?i)held|effect", normalized)[0].strip()
  lookup_text = primary_segment if primary_segment else normalized

  candidates = []
  seen_names = set()

  # Fast hint contains pass.
  for item in item_catalog:
    hint = _match_name_hint_from_item(item)
    name = item.get("name", "unknown_item")
    if hint and hint in lookup_text and name not in seen_names:
      candidates.append(item)
      seen_names.add(name)

  # Fuzzy fallback using item names from JSON.
  names_lookup = {item.get("name", "unknown_item").lower(): item for item in item_catalog}
  fuzzy_hits = difflib.get_close_matches(lookup_text, list(names_lookup.keys()), n=max_candidates, cutoff=0.45)
  for fuzzy_name in fuzzy_hits:
    item = names_lookup.get(fuzzy_name)
    if not item:
      continue
    name = item.get("name", "unknown_item")
    if name in seen_names:
      continue
    candidates.append(item)
    seen_names.add(name)

  return candidates


def _extract_quantity_from_use_item_anchor(panel_screenshot, anchor_match, item_name="unknown_item"):
  x, y, w, h = anchor_match
  image_h, image_w = panel_screenshot.shape[:2]
  safe_item_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(item_name))

  left = min(max(x - TRACKBLAZER_INVENTORY_QTY_CROP_LEFT_FROM_USE_ITEM, 0), image_w)
  right = min(max(left + TRACKBLAZER_INVENTORY_QTY_CROP_WIDTH, 0), image_w)
  top = min(max(y - TRACKBLAZER_INVENTORY_QTY_CROP_Y_PAD, 0), image_h)
  bottom = min(max(y + h + TRACKBLAZER_INVENTORY_QTY_CROP_Y_PAD, 0), image_h)

  if right <= left or bottom <= top:
    return 0

  qty_crop = panel_screenshot[top:bottom, left:right]
  if qty_crop.size == 0:
    return 0

  if args.debug is not None:
    debug_window(qty_crop, save_name=f"trackblazer_qty_anchor_{safe_item_name}")
    info(
      f"[TRACKBLAZER] Qty anchor bbox for '{item_name}': "
      f"(x1={left}, y1={top}, x2={right}, y2={bottom})"
    )

  qty, score = _match_quantity_from_digit_templates(qty_crop, item_name=item_name)
  if qty is None:
    return 0
  return int(qty)


def _find_inventory_row_anchors(panel_screenshot):
  if not os.path.exists(TRACKBLAZER_INVENTORY_USE_ITEM_PATH):
    warning(f"[TRACKBLAZER] use_item template missing: {TRACKBLAZER_INVENTORY_USE_ITEM_PATH}")
    return []

  matches = _find_item_matches(
    TRACKBLAZER_INVENTORY_USE_ITEM_PATH,
    panel_screenshot,
    TRACKBLAZER_INVENTORY_USE_ITEM_THRESHOLD,
    scales=TRACKBLAZER_INVENTORY_USE_ITEM_SCALES,
    use_grayscale=False,
  )
  return sorted(matches, key=lambda box: (box[1], box[0]))


def _row_text_contains_item_hint(screenshot, match, item):
  x, y, w, h = match
  image_h, image_w = screenshot.shape[:2]

  # Name is usually around the same row, right of icon.
  left = min(max(x + w + 4, 0), image_w)
  right = min(max(x + w + 280, 0), image_w)
  top = min(max(y - 8, 0), image_h)
  bottom = min(max(y + h + 24, 0), image_h)
  if right <= left or bottom <= top:
    return True

  row_crop = screenshot[top:bottom, left:right]
  row_text = extract_text(Image.fromarray(row_crop)).lower()
  hint = _match_name_hint_from_item(item)
  if not hint:
    return True
  return hint in row_text


def _build_item_entry(item, count=1, already_bought=False):
  return {
    "name": item.get("name", "unknown_item"),
    "image_path": item.get("image_path", ""),
    "match_threshold": item.get("match_threshold", 0.85),
    "priority": item.get("priority", 0),
    "usage_effects": item.get("usage_effects", {}),
    "count": count,
    "already_bought": already_bought,
  }


def _is_junior_or_classic_first_half_timeline(timeline_token):
  token = str(timeline_token or "").strip()
  if not token:
    return False
  if token.startswith("Junior Year"):
    return True

  classic_first_half_tokens = {
    "Classic Year Early Jan",
    "Classic Year Late Jan",
    "Classic Year Early Feb",
    "Classic Year Late Feb",
    "Classic Year Early Mar",
    "Classic Year Late Mar",
    "Classic Year Early Apr",
    "Classic Year Late Apr",
    "Classic Year Early May",
    "Classic Year Late May",
    "Classic Year Early Jun",
    "Classic Year Late Jun",
  }
  return token in classic_first_half_tokens


def _apply_shop_priority_overrides(item_entry, timeline_token="", inventory_counts=None):
  if not isinstance(item_entry, dict):
    return item_entry

  counts = inventory_counts if isinstance(inventory_counts, dict) else {}
  name = str(item_entry.get("name", "") or "")
  priority = int(item_entry.get("priority", 0) or 0)

  if name == "Grilled Carrots" and not _is_junior_or_classic_first_half_timeline(timeline_token):
    priority = 0

  ankle_weight_names = {
    "Speed Ankle Weights",
    "Stamina Ankle Weights",
    "Power Ankle Weights",
    "Guts Ankle Weights",
  }
  if name in ankle_weight_names and int(counts.get(name, 0) or 0) > 3:
    priority = min(priority, 2)

  # If currently held mood-up cupcakes already sum to +4 or more mood,
  # de-prioritize buying more mood items (cupcakes).
  cupcake_mood_points = (
    int(counts.get("Plain Cupcake", 0) or 0) * 1
    + int(counts.get("Berry Sweet Cupcake", 0) or 0) * 2
  )
  if cupcake_mood_points >= 4 and name in {"Plain Cupcake", "Berry Sweet Cupcake"}:
    priority = 0

  item_entry["priority"] = priority
  return item_entry


def _log_shop_timing(step, started_at, extra=""):
  elapsed = perf_counter() - started_at
  suffix = f" | {extra}" if extra else ""
  debug(f"[TRACKBLAZER][TIMING] {step}: {elapsed:.3f}s{suffix}")


def _find_button_center_on_full_screen(template_path, threshold=0.85, full_screen=None):
  if full_screen is None:
    full_screen = device_action.screenshot(region_ltrb=constants.FULL_SCREEN_LANDSCAPE)
  matches = _find_item_matches(
    template_path,
    full_screen,
    threshold,
    scales=TRACKBLAZER_TEMPLATE_SCALES,
    use_grayscale=True,
  )
  if not matches:
    return None
  x, y, w, h = sorted(matches, key=lambda box: (box[1], box[0]))[0]
  return (x + w // 2, y + h // 2)


def _open_trackblazer_inventory(threshold=0.75):
  if not device_action.locate_and_click(
    TRACKBLAZER_INVENTORY_BTN_PATH,
    confidence=threshold,
    min_search_time=1,
    region_ltrb=constants.FULL_SCREEN_LANDSCAPE,
  ):
    warning("[TRACKBLAZER] training_items_btn not found on Trackblazer main screen.")
    return False
  sleep(TRACKBLAZER_INVENTORY_OPEN_LOAD_WAIT)
  return True


def _open_trackblazer_shop(threshold=0.85):
  if not device_action.locate_and_click(
    TRACKBLAZER_SHOP_BTN_PATH,
    confidence=threshold,
    min_search_time=2,
    region_ltrb=constants.FULL_SCREEN_LANDSCAPE,
  ):
    warning("[TRACKBLAZER] shop_btn not found on Trackblazer main screen.")
    return False

  poll_interval = max(0.0, TRACKBLAZER_SHOP_BACK_BUTTON_POLL_INTERVAL)
  if poll_interval <= 0:
    poll_interval = 1.0
  max_wait = max(0.0, TRACKBLAZER_SHOP_BACK_BUTTON_MAX_WAIT)
  waited = 0.0
  relaxed_threshold = max(0.72, threshold - 0.10)

  while waited <= max_wait:
    device_action.flush_screenshot_cache()
    full_screen = device_action.screenshot(region_ltrb=constants.FULL_SCREEN_LANDSCAPE)
    back_btn_center = _find_button_center_on_full_screen(
      TRACKBLAZER_BACK_BTN_PATH,
      threshold=threshold,
      full_screen=full_screen,
    )
    if back_btn_center is None and relaxed_threshold < threshold:
      back_btn_center = _find_button_center_on_full_screen(
        TRACKBLAZER_BACK_BTN_PATH,
        threshold=relaxed_threshold,
        full_screen=full_screen,
      )
    if back_btn_center:
      return True

    if waited >= max_wait:
      break

    sleep_for = min(poll_interval, max_wait - waited)
    if sleep_for <= 0:
      break
    sleep(sleep_for)
    waited += sleep_for

  warning(f"[TRACKBLAZER] Shop opened but back button was not detected after {waited:.1f}s.")
  return False


def _reopen_trackblazer_shop_for_purchase(threshold=0.85):
  for attempt in range(TRACKBLAZER_SHOP_REOPEN_ATTEMPTS):
    if _open_trackblazer_shop(threshold=threshold):
      return True

    # Recover from transitional popups/screens before retrying.
    device_action.locate_and_click(TRACKBLAZER_OK_BTN_PATH, min_search_time=0.5, region_ltrb=constants.SCREEN_MIDDLE_BBOX)
    device_action.locate_and_click(TRACKBLAZER_BACK_BTN_PATH, min_search_time=0.5, region_ltrb=constants.SCREEN_BOTTOM_BBOX)
    info(f"[TRACKBLAZER] Retrying shop re-open ({attempt + 2}/{TRACKBLAZER_SHOP_REOPEN_ATTEMPTS}).")
  return False


def _dismiss_trackblazer_post_purchase_popup():
  # After checkout, Trackblazer may show a prompt asking whether to use purchased items now.
  # Dismiss it to return to normal flow.
  if device_action.locate_and_click(TRACKBLAZER_CLOSE_BTN_PATH, min_search_time=1, region_ltrb=constants.FULL_SCREEN_LANDSCAPE):
    info("[TRACKBLAZER] Dismissed post-purchase item-use popup via close button.")
    return True

  if device_action.locate_and_click(TRACKBLAZER_OK_BTN_PATH, min_search_time=0.75, region_ltrb=constants.SCREEN_MIDDLE_BBOX):
    info("[TRACKBLAZER] Dismissed post-purchase item-use popup via OK button.")
    return True

  if device_action.locate_and_click(TRACKBLAZER_BACK_BTN_PATH, min_search_time=0.75, region_ltrb=constants.SCREEN_BOTTOM_BBOX):
    info("[TRACKBLAZER] Dismissed post-purchase item-use popup via back button.")
    return True

  debug("[TRACKBLAZER] No post-purchase item-use popup detected.")
  return False


def _scroll_trackblazer_shop_once(extra_scroll_pixels=0):
  before_scroll = device_action.screenshot(region_ltrb=TRACKBLAZER_SHOP_PANEL_LTRB)

  scroll_x = TRACKBLAZER_SHOP_SCROLL_START[0]
  shop_top = TRACKBLAZER_SHOP_PANEL_LTRB[1]

  # Start just inside the top edge of the shop panel.  With any clamped delta
  # >= 20 the computed end_y lands above shop_top, so the finger rests outside
  # the panel and cannot accidentally trigger a purchase on release.
  base_start_y = shop_top + 10
  base_delta = TRACKBLAZER_SHOP_ROW_HEIGHT  # one row per scroll
  adjusted_delta = base_delta + int(extra_scroll_pixels)
  adjusted_delta = max(20, min(TRACKBLAZER_SHOP_ROW_HEIGHT + 70, adjusted_delta))
  target_end = (scroll_x, int(base_start_y - adjusted_delta))

  # Swipe with a 1-second hold at the end before releasing.
  # For ADB: total duration includes travel + hold, so we use a long duration.
  # For pyautogui: manually manage hold/moveTo/sleep/release.
  if device_action.bot.use_adb:
    import utils.adb_actions as _adb
    _adb.swipe(
      scroll_x, base_start_y,
      target_end[0], target_end[1],
      duration=1.5,  # ~0.5s travel + 1s hold baked into ADB gesture duration
    )
  else:
    import utils.pyautogui_actions as _pg
    _pg.moveTo(scroll_x, base_start_y, duration=0.1)
    _pg.hold()
    _pg.moveTo(target_end[0], target_end[1], duration=0.5)
    sleep(1.0)  # hold at end before releasing; end is above shop panel so no accidental selection
    _pg.release()
  device_action.flush_screenshot_cache()
  sleep(TRACKBLAZER_SHOP_SCROLL_SETTLE_WAIT)
  after_scroll = device_action.screenshot(region_ltrb=TRACKBLAZER_SHOP_PANEL_LTRB)
  return are_screenshots_same(before_scroll, after_scroll, diff_threshold=5)


def _scroll_trackblazer_inventory_once():
  left, top, right, bottom = TRACKBLAZER_INVENTORY_PANEL_LTRB
  panel_w = max(1, right - left)
  panel_h = max(1, bottom - top)
  scroll_x = left + panel_w // 2
  start_y = top + int(panel_h * 0.82)
  end_y = top + int(panel_h * 0.62)

  before_scroll = device_action.screenshot(region_ltrb=TRACKBLAZER_INVENTORY_PANEL_LTRB)
  device_action.swipe((scroll_x, start_y), (scroll_x, end_y), duration=0.28)
  sleep(TRACKBLAZER_INVENTORY_SCROLL_SETTLE_WAIT)
  after_scroll = device_action.screenshot(region_ltrb=TRACKBLAZER_INVENTORY_PANEL_LTRB)
  return are_screenshots_same(before_scroll, after_scroll, diff_threshold=5)


def _close_trackblazer_overlay(threshold=0.85, context_label="overlay"):
  close_btn_center = _find_button_center_on_full_screen(TRACKBLAZER_CLOSE_BTN_PATH, threshold=threshold)
  if close_btn_center:
    debug(f"[TRACKBLAZER] close_btn center coordinates ({context_label}): {close_btn_center}")
    device_action.click(close_btn_center)
  else:
    warning(f"[TRACKBLAZER] close_btn not found for {context_label}.")
    device_action.locate_and_click(TRACKBLAZER_CLOSE_BTN_PATH, min_search_time=1, region_ltrb=constants.SCREEN_BOTTOM_BBOX)


def _back_out_of_trackblazer_shop(threshold=0.85):
  back_btn_center = _find_button_center_on_full_screen(TRACKBLAZER_BACK_BTN_PATH, threshold=threshold)
  if back_btn_center:
    debug(f"[TRACKBLAZER] back_btn center coordinates (shop): {back_btn_center}")
    device_action.click(back_btn_center)
  else:
    warning("[TRACKBLAZER] back_btn not found for shop.")
    device_action.locate_and_click(TRACKBLAZER_BACK_BTN_PATH, min_search_time=1, region_ltrb=constants.SCREEN_BOTTOM_BBOX)


def _is_probably_purchased_text(text):
  if not text:
    return False
  normalized = re.sub(r"[^a-z]", "", text.lower())
  if not normalized:
    return False

  # EasyOCR may insert punctuation/spaces; match on "purch" family to be tolerant.
  purchased_tokens = (
    "purchased",
    "purchase",
    "purch",
    "purc",
    "purchas",
  )
  return any(token in normalized for token in purchased_tokens)


def _row_contains_purchased(screenshot, match):
  x, y, w, h = match
  image_h, image_w = screenshot.shape[:2]

  # Purchased/checkmark status is on the far right side of the row.
  left = min(max(x + w + 260, 0), image_w)
  right = min(max(x + w + 430, 0), image_w)
  top = min(max(y - 10, 0), image_h)
  bottom = min(max(y + h + 24, 0), image_h)
  if right <= left or bottom <= top:
    return False

  row_crop = screenshot[top:bottom, left:right]

  plain_text = extract_text(Image.fromarray(row_crop))
  if _is_probably_purchased_text(plain_text):
    return True

  # Retry with OCR-enhanced image for low-contrast rows.
  enhanced_crop = enhance_image_for_ocr(row_crop, resize_factor=3)
  enhanced_text = extract_text(enhanced_crop)
  if _is_probably_purchased_text(enhanced_text):
    return True

  return False


def _row_contains_selected_checkout(screenshot, match):
  x, y, w, h = match
  image_h, image_w = screenshot.shape[:2]

  # Selected-checkmark status is on the far right side of the row.
  left = min(max(x + w + 260, 0), image_w)
  right = min(max(x + w + 430, 0), image_w)
  top = min(max(y - 10, 0), image_h)
  bottom = min(max(y + h + 24, 0), image_h)
  if right <= left or bottom <= top:
    return False

  row_crop = screenshot[top:bottom, left:right]
  if row_crop.size == 0:
    return False

  template_path = TRACKBLAZER_SHOP_CHECKMARK_SELECTED_PATH
  if not os.path.exists(template_path):
    return False

  # Use a stricter threshold than generic row-anchor detection to avoid
  # misclassifying unchecked rows as already selected.
  matches = _find_item_matches(
    template_path,
    row_crop,
    max(TRACKBLAZER_SHOP_ROW_ANCHOR_THRESHOLD, 0.86),
    scales=TRACKBLAZER_SHOP_ROW_ANCHOR_SCALES,
    use_grayscale=False,
  )
  if matches:
    return True

  return False


def _is_shop_left_column_match(screenshot, match):
  _, image_w = screenshot.shape[:2]
  x, _, w, _ = match
  center_x = x + w // 2
  min_x = int(image_w * TRACKBLAZER_SHOP_ICON_MIN_X_RATIO)
  max_x = int(image_w * TRACKBLAZER_SHOP_ICON_MAX_X_RATIO)
  return min_x <= center_x <= max_x


def _find_shop_row_anchors(shop_screenshot):
  image_h, image_w = shop_screenshot.shape[:2]
  min_anchor_x = int(image_w * TRACKBLAZER_SHOP_ROW_ANCHOR_MIN_X_RATIO)

  template_specs = (
    (TRACKBLAZER_SHOP_CHECKMARK_DISABLED_PATH, "available", 1),
    (TRACKBLAZER_SHOP_CHECKMARK_SELECTED_PATH, "selected", 2),
    (TRACKBLAZER_SHOP_PURCHASED_PATH, "purchased", 3),
  )

  buckets = {}
  for template_path, row_state, priority in template_specs:
    if not os.path.exists(template_path):
      warning(f"[TRACKBLAZER] shop row anchor template missing: {template_path}")
      continue

    matches = _find_item_matches(
      template_path,
      shop_screenshot,
      TRACKBLAZER_SHOP_ROW_ANCHOR_THRESHOLD,
      scales=TRACKBLAZER_SHOP_ROW_ANCHOR_SCALES,
      use_grayscale=False,
    )
    for x, y, w, h in matches:
      center_x = x + w // 2
      if center_x < min_anchor_x:
        continue

      row_bucket = (y + h // 2) // TRACKBLAZER_SHOP_ROW_BUCKET_HEIGHT
      existing = buckets.get(row_bucket)
      entry = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "state": row_state,
        "priority": priority,
      }
      if existing is None or priority > existing["priority"]:
        buckets[row_bucket] = entry

  rows = [buckets[key] for key in sorted(buckets.keys())]
  return rows


def _identify_shop_item_from_row(row_crop, row_text, item_catalog, threshold=0.85):
  row_candidates = _candidate_items_from_row_ocr(row_text, item_catalog)
  search_pools = [row_candidates, item_catalog] if row_candidates else [item_catalog]
  seen_names = set()

  for item_pool in search_pools:
    for item in item_pool:
      name = item.get("name", "unknown_item")
      if name in seen_names:
        continue
      seen_names.add(name)

      image_path = item.get("image_path", "")
      if not image_path or not os.path.exists(image_path):
        continue

      match_threshold = item.get("match_threshold", threshold)
      matches = _find_item_matches(
        image_path,
        row_crop,
        match_threshold,
        scales=TRACKBLAZER_SHOP_TEMPLATE_SCALES,
        use_grayscale=False,
      )
      if not matches:
        matches = _find_item_matches(
          image_path,
          row_crop,
          max(0.72, match_threshold - 0.06),
          scales=(1.0,),
          use_grayscale=True,
        )
      if not matches:
        continue

      local_match = sorted(matches, key=lambda box: (box[0], box[1]))[0]
      return item, local_match

  return None, None


def _confirm_trackblazer_item_use(threshold=0.85):
  """Confirms item usage after a Use button has been clicked.

  Tries to locate and click the confirm_item_use template first.  Falls back
  to clicking the fixed coordinate (530, 1000) if the template is not found.
  Then taps the same location again 2 seconds later for the follow-up
  confirmation screen, and waits for animations to finish.
  """
  confirmation_position = TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_POSITION
  if os.path.exists(TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_PATH):
    confirmed = device_action.locate_and_click(
      TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_PATH,
      confidence=threshold,
      min_search_time=1.5,
      region_ltrb=constants.FULL_SCREEN_LANDSCAPE,
    )
    if confirmed:
      info("[TRACKBLAZER] Confirmed item use via template match.")
    else:
      debug("[TRACKBLAZER] confirm_item_use template not matched; falling back to fixed coordinate.")
      device_action.click(confirmation_position, text="[TRACKBLAZER] Confirming item use at fixed position")
      info("[TRACKBLAZER] Confirmed item use at fixed position (530, 1000).")
  else:
    warning(f"[TRACKBLAZER] confirm_item_use template missing: {TRACKBLAZER_INVENTORY_CONFIRM_ITEM_USE_PATH}")
    device_action.click(confirmation_position, text="[TRACKBLAZER] Confirming item use at fixed position")
    info("[TRACKBLAZER] Confirmed item use at fixed position (530, 1000).")

  sleep(2.0)
  device_action.click(confirmation_position, text="[TRACKBLAZER] Confirming follow-up item-use prompt")
  info("[TRACKBLAZER] Confirmed follow-up item-use prompt at fixed position (530, 1000).")

  sleep(7.0)
  debug("[TRACKBLAZER] Waited 7.0s for item-use animations to finish.")

  return True


def _collect_trackblazer_visible_inventory_items(item_catalog, threshold=0.85):
  list_screen = device_action.screenshot(region_ltrb=TRACKBLAZER_INVENTORY_PANEL_LTRB)
  list_h, list_w = list_screen.shape[:2]
  row_anchors = _find_inventory_row_anchors(list_screen)
  visible_items = []
  seen_names = set()

  for anchor_x, anchor_y, anchor_w, anchor_h in row_anchors:
    anchor_center_y = anchor_y + anchor_h // 2
    row_top = max(0, anchor_center_y - TRACKBLAZER_INVENTORY_ROW_HEIGHT // 2)
    row_bottom = min(list_h, row_top + TRACKBLAZER_INVENTORY_ROW_HEIGHT)
    row_right = min(list_w, anchor_x + anchor_w + 14)
    row_left = max(0, row_right - TRACKBLAZER_INVENTORY_ROW_WIDTH)
    if row_bottom - row_top < 60 or row_right - row_left < 220:
      continue

    row_crop = list_screen[row_top:row_bottom, row_left:row_right]
    if row_crop.size == 0:
      continue

    row_text = extract_text(Image.fromarray(row_crop)).lower()
    row_text_candidates = _candidate_items_from_row_ocr(row_text, item_catalog)
    candidate_items = row_text_candidates if row_text_candidates else item_catalog

    def _try_match_items(item_pool):
      for item in item_pool:
        name = item.get("name", "unknown_item")
        image_path = item.get("image_path", "")
        match_threshold = item.get("match_threshold", threshold)

        if not image_path or not os.path.exists(image_path):
          continue

        matches = _find_item_matches(
          image_path,
          row_crop,
          match_threshold,
          scales=TRACKBLAZER_INVENTORY_TEMPLATE_SCALES,
          use_grayscale=False,
        )
        if not matches:
          matches = _find_item_matches(
            image_path,
            row_crop,
            max(0.72, match_threshold - 0.06),
            scales=(1.0,),
            use_grayscale=True,
          )
        if not matches:
          continue

        x, y, w, h = sorted(matches, key=lambda box: (box[1], box[0]))[0]
        global_match = (row_left + x, row_top + y, w, h)
        if not _is_shop_left_column_match(list_screen, global_match):
          continue

        item_hint = _match_name_hint_from_item(item)
        if item_pool is row_text_candidates and item_hint and item_hint not in row_text:
          continue

        return item, global_match
      return None, None

    matched_item, global_match = _try_match_items(candidate_items)
    if matched_item is None and row_text_candidates:
      matched_item, global_match = _try_match_items(item_catalog)
    if matched_item is None or global_match is None:
      continue

    name = matched_item.get("name", "unknown_item")
    if name in seen_names:
      continue
    seen_names.add(name)

    use_center_x = TRACKBLAZER_INVENTORY_PANEL_LTRB[0] + min(max(anchor_x + anchor_w // 2, 0), list_w - 1)
    use_center_y = TRACKBLAZER_INVENTORY_PANEL_LTRB[1] + min(max(anchor_y + anchor_h // 2, 0), list_h - 1)
    visible_items.append({
      "name": name,
      "item": matched_item,
      "match": global_match,
      "use_center": (use_center_x, use_center_y),
    })

  return visible_items


def _select_trackblazer_shop_item_for_checkout(item_entry, threshold=0.85):
  step_started_at = perf_counter()
  image_path = item_entry.get("image_path", "")
  name = item_entry.get("name", "unknown_item")
  match_threshold = item_entry.get("match_threshold", threshold)
  expected_price = int(item_entry.get("price", 0))

  if not image_path or not os.path.exists(image_path):
    warning(f"[TRACKBLAZER] Cannot purchase '{name}' because template is missing: {image_path}")
    _log_shop_timing("shop select candidate", step_started_at, f"item={name}, selected=False, reason=missing_template")
    return False

  list_screen = device_action.screenshot(region_ltrb=constants.GAME_WINDOW_BBOX)
  row_anchors = _find_shop_row_anchors(list_screen)

  def _match_is_in_selected_row(match):
    if not row_anchors:
      return False

    _, y, _, h = match
    match_center_y = y + h // 2
    nearest_row = min(
      row_anchors,
      key=lambda row: abs((row.get("y", 0) + row.get("h", 0) // 2) - match_center_y),
    )
    nearest_row_center_y = nearest_row.get("y", 0) + nearest_row.get("h", 0) // 2
    if abs(nearest_row_center_y - match_center_y) > TRACKBLAZER_SHOP_ROW_HEIGHT:
      return False
    return nearest_row.get("state") == "selected"

  matches = _find_item_matches(
    image_path,
    list_screen,
    match_threshold,
    scales=TRACKBLAZER_SHOP_TEMPLATE_SCALES,
    use_grayscale=False,
  )
  valid_matches = []
  for match in matches:
    if not _is_shop_left_column_match(list_screen, match):
      continue
    if _match_is_in_selected_row(match):
      debug(f"[TRACKBLAZER] Skipping '{name}' row because row-anchor state is already selected.")
      continue
    if not _row_text_contains_item_hint(list_screen, match, item_entry):
      continue
    if _row_contains_selected_checkout(list_screen, match):
      debug(f"[TRACKBLAZER] Skipping '{name}' row because it is already selected for checkout.")
      continue
    if _row_contains_purchased(list_screen, match):
      continue
    if expected_price > 0:
      detected_price = _extract_price_from_match_region(list_screen, match, expected_price)
      if detected_price != expected_price:
        debug(
          f"[TRACKBLAZER] Rejecting selection row for '{name}' because price {detected_price} != expected {expected_price}."
        )
        continue
    valid_matches.append(match)

  if not valid_matches:
    debug(f"[TRACKBLAZER] Could not find purchasable row for '{name}' during selection pass.")
    _log_shop_timing("shop select candidate", step_started_at, f"item={name}, selected=False, reason=no_valid_match")
    return False

  list_h, list_w = list_screen.shape[:2]
  x, y, w, h = sorted(valid_matches, key=lambda box: (box[1], box[0]))[0]
  # Select item for checkout by clicking on the right side of the row.
  right_side_x = min(max(x + w + 300, 0), list_w - 5)
  center_x = constants.GAME_WINDOW_BBOX[0] + right_side_x
  center_y = constants.GAME_WINDOW_BBOX[1] + y + h // 2
  device_action.click((center_x, center_y), text=f"[TRACKBLAZER] Selecting '{name}' for checkout")
  sleep(1.0)
  info(f"[TRACKBLAZER] Added '{name}' to checkout.")
  _log_shop_timing("shop select candidate", step_started_at, f"item={name}, selected=True")
  return True


def collect_trackblazer_shop_snapshot(item_catalog, threshold=0.85, max_scrolls=20, timeline_token="", inventory_counts=None):
  total_started_at = perf_counter()
  debug("[TRACKBLAZER][TIMING] shop flow started")

  open_started_at = perf_counter()
  if not _open_trackblazer_shop(threshold=threshold):
    _log_shop_timing("shop open", open_started_at, "success=False")
    _log_shop_timing("shop flow total", total_started_at, "open_failed")
    return [], None, []
  _log_shop_timing("shop open", open_started_at, "success=True")

  back_btn_started_at = perf_counter()
  back_btn_center = _find_button_center_on_full_screen(TRACKBLAZER_BACK_BTN_PATH, threshold=threshold)
  if back_btn_center:
    debug(f"[TRACKBLAZER] back_btn center coordinates (shop): {back_btn_center}")
  _log_shop_timing("shop locate back button", back_btn_started_at, f"found={back_btn_center is not None}")

  coin_ocr_started_at = perf_counter()
  shop_coins = _extract_shop_coin_count(threshold=threshold)
  _log_shop_timing("shop coin OCR call", coin_ocr_started_at, f"coins={shop_coins}")

  found_shop_items = []  # list; preserves duplicate slots (same item can stock multiple rows)
  seen_row_hashes = set()  # MD5 of row pixel content; belt-and-suspenders against imperfect scroll alignment
  no_movement_scrolls = 0
  hit_purchased_tail = False
  expected_new_anchor_y = None
  pending_scroll_adjust = 0

  GREEN  = (0, 220, 60)
  ORANGE = (255, 140, 0)
  RED    = (220, 40, 40)
  YELLOW = (255, 220, 0)
  CYAN   = (0, 200, 200)

  def _process_row(row, screenshot, panel_h, panel_w, dbg, label=""):
    """Crop, hash-check, OCR, identify and append one row anchor. Returns True if purchased."""
    nonlocal hit_purchased_tail
    anchor_x  = row.get("x", 0)
    anchor_y  = row.get("y", 0)
    anchor_w  = row.get("w", 0)
    anchor_h  = row.get("h", 0)
    row_state = row.get("state", "available")

    if row_state == "purchased":
      debug("[TRACKBLAZER] Encountered purchased row anchor — stopping discovery.")
      hit_purchased_tail = True
      return True

    anchor_center_y = anchor_y + anchor_h // 2
    row_top    = max(0, anchor_center_y - TRACKBLAZER_SHOP_ROW_HEIGHT // 2)
    row_bottom = min(panel_h, row_top + TRACKBLAZER_SHOP_ROW_HEIGHT)
    row_right  = min(panel_w, anchor_x + anchor_w + 14)
    row_left   = max(0, row_right - TRACKBLAZER_SHOP_ROW_WIDTH)

    if row_bottom - row_top < 20 or row_right - row_left < 100:
      cv2.rectangle(dbg, (row_left, row_top), (row_right, row_bottom), RED, 2)
      cv2.putText(dbg, f"{label}too small", (row_left + 4, max(row_top + 14, anchor_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1)
      return False

    row_crop = screenshot[row_top:row_bottom, row_left:row_right]
    if row_crop.size == 0:
      cv2.rectangle(dbg, (row_left, row_top), (row_right, row_bottom), RED, 2)
      cv2.putText(dbg, f"{label}empty", (row_left + 4, max(row_top + 14, anchor_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1)
      return False

    row_hash = hashlib.md5(row_crop.tobytes()).hexdigest()
    if row_hash in seen_row_hashes:
      debug(f"[TRACKBLAZER] {label}Duplicate pixel hash at y={anchor_center_y} — skipping.")
      cv2.rectangle(dbg, (row_left, row_top), (row_right, row_bottom), YELLOW, 2)
      cv2.putText(dbg, f"{label}DUP", (row_left + 4, max(row_top + 14, anchor_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1)
      return False

    row_text = extract_text(Image.fromarray(row_crop)).lower()
    item, local_match = _identify_shop_item_from_row(row_crop, row_text, item_catalog, threshold=threshold)
    if item is None or local_match is None:
      debug(f"[TRACKBLAZER] {label}No ID (OCR: '{' '.join(row_text.split())[:60]}')")
      cv2.rectangle(dbg, (row_left, row_top), (row_right, row_bottom), ORANGE, 2)
      cv2.putText(dbg, f"{label}NO ID: {' '.join(row_text.split())[:20]}", (row_left + 4, max(row_top + 14, anchor_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, ORANGE, 1)
      return False

    name = item.get("name", "unknown_item")
    lx, ly, lw, lh = local_match
    global_match = (row_left + lx, row_top + ly, lw, lh)
    entry = _build_item_entry(item, already_bought=False)
    _apply_shop_priority_overrides(entry, timeline_token=timeline_token, inventory_counts=inventory_counts)
    entry["price"] = _extract_price_from_match_region(screenshot, global_match, item.get("cost", 0))
    found_shop_items.append(entry)
    seen_row_hashes.add(row_hash)

    cx = TRACKBLAZER_SHOP_PANEL_LTRB[0] + global_match[0] + global_match[2] // 2
    cy = TRACKBLAZER_SHOP_PANEL_LTRB[1] + global_match[1] + global_match[3] // 2
    debug(f"[TRACKBLAZER] {label}Shop row '{name}' at ({cx},{cy}) price={entry['price']} state={row_state}")
    cv2.rectangle(dbg, (row_left, row_top), (row_right, row_bottom), GREEN, 2)
    cv2.putText(dbg, f"{label}{name[:28]}", (row_left + 4, max(row_top + 14, anchor_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREEN, 1)
    return False

  # Phase 1: full discovery pass.
  # Iteration 0 → scan ALL visible rows (initial view covers ~3 rows).
  # Iterations 1+ → scroll exactly 1 row height, then consider ONLY the newly revealed bottom row.
  discovery_started_at = perf_counter()
  for iteration in range(max_scrolls + 1):
    device_action.flush_screenshot_cache()
    screenshot = device_action.screenshot(region_ltrb=TRACKBLAZER_SHOP_PANEL_LTRB)
    panel_h, panel_w = screenshot.shape[:2]

    row_anchors = _find_shop_row_anchors(screenshot)
    dbg = screenshot.copy()

    # Partition into purchased / available
    available_anchors  = [r for r in row_anchors if r.get("state") != "purchased"]
    purchased_anchors  = [r for r in row_anchors if r.get("state") == "purchased"]

    if iteration == 0:
      # Initial scan: process every visible row top-to-bottom.
      if available_anchors:
        expected_new_anchor_y = max(
          r.get("y", 0) + r.get("h", 0) // 2 for r in available_anchors
        )
      for row in available_anchors:
        _process_row(row, screenshot, panel_h, panel_w, dbg, label="")
        if hit_purchased_tail:
          break
    else:
      # Only the bottom-most available anchor — the row freshly scrolled into view.
      if purchased_anchors:
        # When purchased is visible, we are at the tail. Do not process another
        # "new" available row in this frame; it is usually the previously seen last item.
        info("[TRACKBLAZER] Purchased anchor visible; skipping further new-row processing to avoid tail double-count.")
      elif available_anchors:
        bottom_row = max(available_anchors, key=lambda r: r.get("y", 0) + r.get("h", 0) // 2)
        observed_anchor_y = bottom_row.get("y", 0) + bottom_row.get("h", 0) // 2

        if expected_new_anchor_y is None:
          expected_new_anchor_y = observed_anchor_y

        drift_px = observed_anchor_y - expected_new_anchor_y
        if abs(drift_px) > TRACKBLAZER_SHOP_NEW_ROW_ANCHOR_TOLERANCE:
          proposed_adjust = int(drift_px * TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_GAIN)
          if proposed_adjust == 0:
            proposed_adjust = 1 if drift_px > 0 else -1
          pending_scroll_adjust = max(
            -TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_MAX,
            min(TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_MAX, proposed_adjust)
          )
          debug(
            "[TRACKBLAZER] New-row checkmark drift detected "
            f"(observed_y={observed_anchor_y}, expected_y={expected_new_anchor_y}, drift={drift_px}). "
            f"Scheduling next-swipe adjustment={pending_scroll_adjust}."
          )
        else:
          pending_scroll_adjust = 0

        # Draw all other visible rows as cyan (already processed, just in view)
        for row in available_anchors:
          if row is not bottom_row:
            ay = row.get("y", 0) + row.get("h", 0) // 2
            rt = max(0, ay - TRACKBLAZER_SHOP_ROW_HEIGHT // 2)
            rb = min(panel_h, rt + TRACKBLAZER_SHOP_ROW_HEIGHT)
            rr = min(panel_w, row.get("x", 0) + row.get("w", 0) + 14)
            rl = max(0, rr - TRACKBLAZER_SHOP_ROW_WIDTH)
            cv2.rectangle(dbg, (rl, rt), (rr, rb), CYAN, 1)
            cv2.putText(dbg, "prev", (rl + 4, max(rt + 14, ay)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CYAN, 1)
        _process_row(bottom_row, screenshot, panel_h, panel_w, dbg, label="new: ")

    # Mark purchased anchors in debug image
    for row in purchased_anchors:
      ay = row.get("y", 0) + row.get("h", 0) // 2
      rt = max(0, ay - TRACKBLAZER_SHOP_ROW_HEIGHT // 2)
      rb = min(panel_h, rt + TRACKBLAZER_SHOP_ROW_HEIGHT)
      rr = min(panel_w, row.get("x", 0) + row.get("w", 0) + 14)
      rl = max(0, rr - TRACKBLAZER_SHOP_ROW_WIDTH)
      cv2.rectangle(dbg, (rl, rt), (rr, rb), (180, 180, 180), 2)
      cv2.putText(dbg, "PURCHASED", (rl + 4, max(rt + 14, ay)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
      hit_purchased_tail = True

    debug_window(dbg, save_name="trackblazer_shop_discovery_rows")

    if hit_purchased_tail:
      info("[TRACKBLAZER] Purchased row reached; stopping shop discovery.")
      break

    if iteration >= max_scrolls:
      break

    if _scroll_trackblazer_shop_once(extra_scroll_pixels=pending_scroll_adjust):
      no_movement_scrolls += 1
      info("[TRACKBLAZER] Shop scroll reached end of list; stopping discovery to avoid tail double-counting.")
      break
    else:
      no_movement_scrolls = 0
    pending_scroll_adjust = 0

    if no_movement_scrolls >= 2:
      break

  _log_shop_timing("shop discovery total", discovery_started_at, f"items={len(found_shop_items)}")

  planning_started_at = perf_counter()
  shop_items = list(found_shop_items)
  shop_items.sort(key=lambda x: (x.get("price", 0), -x["priority"], x["name"]))

  raw_buy_candidates = [
    item for item in shop_items
    if item.get("priority", 0) >= 4 and item.get("price", 0) > 0
  ]
  raw_buy_candidates.sort(key=lambda item: (-item.get("priority", 0), item.get("price", 0), item.get("name", "")))

  current_counts = inventory_counts if isinstance(inventory_counts, dict) else {}
  planned_purchase_counts = Counter()
  skipped_at_inventory_cap = set()
  buy_candidates = []
  for item in raw_buy_candidates:
    name = item.get("name", "unknown_item")
    held_count = int(current_counts.get(name, 0) or 0)
    remaining_capacity = TRACKBLAZER_INVENTORY_QTY_MAX - held_count - planned_purchase_counts.get(name, 0)
    if remaining_capacity <= 0:
      skipped_at_inventory_cap.add(name)
      continue
    buy_candidates.append(item)
    planned_purchase_counts[name] += 1

  shopping_list = [
    {
      "name": item.get("name", "unknown_item"),
      "priority": item.get("priority", 0),
      "price": int(item.get("price", 0)),
    }
    for item in buy_candidates
  ]
  desired_checkout_total = sum(item["price"] for item in shopping_list)
  _log_shop_timing("shop candidate planning", planning_started_at, f"buy_candidates={len(buy_candidates)}, desired_total={desired_checkout_total}")

  if skipped_at_inventory_cap:
    info(
      "[TRACKBLAZER] Skipping item buys at inventory cap "
      f"({TRACKBLAZER_INVENTORY_QTY_MAX}): {sorted(skipped_at_inventory_cap)}"
    )

  if shopping_list:
    info("[TRACKBLAZER] Desired shopping list (priority>=4):")
    for idx, item in enumerate(shopping_list, start=1):
      info(f"[TRACKBLAZER]   {idx}. {item['name']} - {item['price']} coins (priority {item['priority']})")
    info(f"[TRACKBLAZER] Desired checkout total: {desired_checkout_total} coins")
  else:
    info("[TRACKBLAZER] Desired shopping list is empty.")

  # Exit after discovery, then re-open for purchase pass.
  back_out_started_at = perf_counter()
  _back_out_of_trackblazer_shop(threshold=threshold)
  _log_shop_timing("shop exit after discovery", back_out_started_at)

  if shop_coins is not None and shop_coins > 0:
    if not buy_candidates:
      _log_shop_timing("shop flow total", total_started_at, "no_buy_candidates")
      return shop_items, shop_coins, []

    # Build a constrained checkout plan in priority order before re-entering the shop.
    checkout_plan_started_at = perf_counter()
    planned_checkout = []
    planned_total = 0
    for candidate in buy_candidates:
      cost = int(candidate.get("price", 0))
      if cost <= 0:
        continue
      if planned_total + cost > shop_coins:
        continue
      planned_checkout.append(candidate)
      planned_total += cost

    info(
      "[TRACKBLAZER] Planned checkout (within coins): "
      f"{[item.get('name', 'unknown_item') for item in planned_checkout]} | total={planned_total}"
    )
    if planned_total > shop_coins:
      warning(
        f"[TRACKBLAZER] Planned checkout total {planned_total} exceeds available coins {shop_coins}."
      )
    else:
      info(
        f"[TRACKBLAZER] Checkout total validation passed: {planned_total} <= {shop_coins}."
      )
    _log_shop_timing("shop checkout planning", checkout_plan_started_at, f"planned_items={len(planned_checkout)}, planned_total={planned_total}")

    if not planned_checkout:
      _log_shop_timing("shop flow total", total_started_at, "planned_checkout_empty")
      return shop_items, shop_coins, []

    reopen_started_at = perf_counter()
    if not _reopen_trackblazer_shop_for_purchase(threshold=threshold):
      warning("[TRACKBLAZER] Could not re-open shop for purchase pass.")
      _log_shop_timing("shop reopen for purchase", reopen_started_at, "success=False")
      _log_shop_timing("shop flow total", total_started_at, "reopen_failed")
      return shop_items, shop_coins, []
    _log_shop_timing("shop reopen for purchase", reopen_started_at, "success=True")

    info(f"[TRACKBLAZER] Waiting {TRACKBLAZER_SHOP_REENTRY_LOAD_WAIT}s for shop to fully load before selection.")
    load_wait_started_at = perf_counter()
    sleep(TRACKBLAZER_SHOP_REENTRY_LOAD_WAIT)
    _log_shop_timing("shop reentry load wait", load_wait_started_at)

    remaining_coins = shop_coins
    selected_for_checkout = []
    # Use a counter so duplicate shop slots (same item stocked twice) are each purchased.
    pending_counts = Counter(item.get("name", "unknown_item") for item in planned_checkout)
    no_progress_scrolls = 0
    no_movement_scrolls = 0
    selection_scrolls = 0
    selection_expected_new_anchor_y = None
    selection_pending_scroll_adjust = 0

    # Phase 2: select all planned items for checkout, then confirm once.
    selection_started_at = perf_counter()

    def _run_selection_pass():
      selected_count = 0
      nonlocal remaining_coins
      for candidate in planned_checkout:
        name = candidate.get("name", "unknown_item")
        if pending_counts.get(name, 0) <= 0:
          continue

        cost = int(candidate.get("price", 0))
        if cost <= 0 or remaining_coins < cost:
          continue

        if _select_trackblazer_shop_item_for_checkout(candidate, threshold=threshold):
          remaining_coins -= cost
          pending_counts[name] -= 1
          selected_for_checkout.append(name)
          selected_count += 1

      return selected_count

    while True:
      selected_this_pass = _run_selection_pass()

      if selected_this_pass == 0:
        no_progress_scrolls += 1
      else:
        no_progress_scrolls = 0

      pending_selection = any(count > 0 for count in pending_counts.values())
      if not pending_selection:
        break

      device_action.flush_screenshot_cache()
      selection_panel_screen = device_action.screenshot(region_ltrb=TRACKBLAZER_SHOP_PANEL_LTRB)
      selection_row_anchors = _find_shop_row_anchors(selection_panel_screen)
      selection_available_anchors = [r for r in selection_row_anchors if r.get("state") != "purchased"]
      selection_purchased_anchors = [r for r in selection_row_anchors if r.get("state") == "purchased"]

      if selection_purchased_anchors:
        info("[TRACKBLAZER] Reached purchased row during selection pass; stopping exhaustive scroll.")
        break

      if selection_available_anchors:
        selection_bottom_row = max(
          selection_available_anchors,
          key=lambda r: r.get("y", 0) + r.get("h", 0) // 2,
        )
        selection_observed_anchor_y = selection_bottom_row.get("y", 0) + selection_bottom_row.get("h", 0) // 2

        if selection_expected_new_anchor_y is None:
          selection_expected_new_anchor_y = selection_observed_anchor_y

        selection_drift_px = selection_observed_anchor_y - selection_expected_new_anchor_y
        if abs(selection_drift_px) > TRACKBLAZER_SHOP_NEW_ROW_ANCHOR_TOLERANCE:
          selection_proposed_adjust = int(selection_drift_px * TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_GAIN)
          if selection_proposed_adjust == 0:
            selection_proposed_adjust = 1 if selection_drift_px > 0 else -1
          selection_pending_scroll_adjust = max(
            -TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_MAX,
            min(TRACKBLAZER_SHOP_NEXT_SWIPE_ADJUST_MAX, selection_proposed_adjust)
          )
          info(
            "[TRACKBLAZER] Selection-pass checkmark drift detected "
            f"(observed_y={selection_observed_anchor_y}, expected_y={selection_expected_new_anchor_y}, drift={selection_drift_px}). "
            f"Scheduling next-swipe adjustment={selection_pending_scroll_adjust}."
          )
        else:
          selection_pending_scroll_adjust = 0

      if selection_scrolls >= TRACKBLAZER_SHOP_SELECTION_EXHAUSTIVE_SCROLL_LIMIT:
        warning(
          "[TRACKBLAZER] Exhaustive selection scroll reached safety limit "
          f"({TRACKBLAZER_SHOP_SELECTION_EXHAUSTIVE_SCROLL_LIMIT})."
        )
        break

      selection_scrolls += 1

      if _scroll_trackblazer_shop_once(extra_scroll_pixels=selection_pending_scroll_adjust):
        no_movement_scrolls += 1
        info("[TRACKBLAZER] Selection pass reached shop list end; running final selection check on bottom rows.")

        # Final pass on the terminal viewport before checkout confirm.
        final_selected = _run_selection_pass()
        if final_selected > 0:
          info(f"[TRACKBLAZER] Final bottom-of-list selection added {final_selected} item(s).")

        break
      else:
        no_movement_scrolls = 0
      selection_pending_scroll_adjust = 0

      if no_movement_scrolls >= 2:
        break

    _log_shop_timing("shop selection total", selection_started_at, f"selected={len(selected_for_checkout)}")

    if selected_for_checkout:
      info(f"[TRACKBLAZER] Selected for checkout: {selected_for_checkout}")

      # Confirm all selected items with one checkout confirmation.
      checkout_started_at = perf_counter()
      confirmed = device_action.locate_and_click(
        TRACKBLAZER_CONFIRM_BTN_PATH,
        min_search_time=1.5,
        region_ltrb=constants.SCREEN_MIDDLE_BBOX,
      )
      if not confirmed:
        confirmed = device_action.locate_and_click(
          TRACKBLAZER_CONFIRM_BTN_PATH,
          min_search_time=1.5,
          region_ltrb=constants.SCREEN_BOTTOM_BBOX,
        )

      device_action.locate_and_click(
        TRACKBLAZER_OK_BTN_PATH,
        min_search_time=0.75,
        region_ltrb=constants.SCREEN_MIDDLE_BBOX,
      )

      _dismiss_trackblazer_post_purchase_popup()
      _log_shop_timing("shop checkout confirm+dismiss", checkout_started_at, f"confirmed={confirmed}")

      if confirmed:
        info(f"[TRACKBLAZER] Checkout confirmed. Estimated remaining coins: {remaining_coins}")
        purchased_items = list(selected_for_checkout)
      else:
        warning("[TRACKBLAZER] Confirm button not found during checkout.")
        purchased_items = []
    else:
      info("[TRACKBLAZER] No items selected for checkout in purchase pass.")
      purchased_items = []

    final_back_out_started_at = perf_counter()
    _back_out_of_trackblazer_shop(threshold=threshold)
    _log_shop_timing("shop exit after purchase", final_back_out_started_at)
    _log_shop_timing("shop flow total", total_started_at, f"purchased={len(purchased_items)}")
    return shop_items, shop_coins, purchased_items

  _log_shop_timing("shop flow total", total_started_at, "coins_unavailable_or_zero")
  return shop_items, shop_coins, []


def collect_trackblazer_owned_items_snapshot(item_catalog, threshold=0.75, max_scrolls=20):
  if not _open_trackblazer_inventory(threshold=threshold):
    return []

  item_quantities = {}
  detected_item_meta = {}
  item_quantity_samples = defaultdict(list)
  seen_item_rows = set()
  no_progress_scrolls = 0
  no_movement_scrolls = 0

  for _ in range(max_scrolls):
    list_screen = device_action.screenshot(region_ltrb=TRACKBLAZER_INVENTORY_PANEL_LTRB)
    list_h, list_w = list_screen.shape[:2]
    row_anchors = _find_inventory_row_anchors(list_screen)
    found_new_this_pass = False

    pass_matches = []

    for anchor_x, anchor_y, anchor_w, anchor_h in row_anchors:
      anchor_center_y = anchor_y + anchor_h // 2
      row_top = max(0, anchor_center_y - TRACKBLAZER_INVENTORY_ROW_HEIGHT // 2)
      row_bottom = min(list_h, row_top + TRACKBLAZER_INVENTORY_ROW_HEIGHT)
      row_right = min(list_w, anchor_x + anchor_w + 14)
      row_left = max(0, row_right - TRACKBLAZER_INVENTORY_ROW_WIDTH)
      if row_bottom - row_top < 60 or row_right - row_left < 220:
        continue

      row_crop = list_screen[row_top:row_bottom, row_left:row_right]
      if row_crop.size == 0:
        continue

      row_text = extract_text(Image.fromarray(row_crop)).lower()
      row_text_snippet = " ".join(row_text.split())[:80]
      row_text_candidates = _candidate_items_from_row_ocr(row_text, item_catalog)
      candidate_items = row_text_candidates if row_text_candidates else item_catalog

      def _try_match_items(item_pool):
        for item in item_pool:
          name = item.get("name", "unknown_item")
          image_path = item.get("image_path", "")
          match_threshold = item.get("match_threshold", threshold)

          if not image_path or not os.path.exists(image_path):
            continue

          matches = _find_item_matches(
            image_path,
            row_crop,
            match_threshold,
            scales=TRACKBLAZER_INVENTORY_TEMPLATE_SCALES,
            use_grayscale=False,
          )
          if not matches:
            matches = _find_item_matches(
              image_path,
              row_crop,
              max(0.72, match_threshold - 0.06),
              scales=(1.0,),
              use_grayscale=True,
            )
          if not matches:
            continue

          x, y, w, h = sorted(matches, key=lambda box: (box[1], box[0]))[0]
          global_match = (row_left + x, row_top + y, w, h)

          if not _is_shop_left_column_match(list_screen, global_match):
            continue

          # For OCR-suggested candidates, require hint check. Fallback-all pass skips this check.
          item_hint = _match_name_hint_from_item(item)
          if item_pool is row_text_candidates and item_hint and item_hint not in row_text:
            continue

          return item, global_match
        return None, None

      matched_item, global_match = _try_match_items(candidate_items)
      if matched_item is None and row_text_candidates:
        # OCR-suggested items failed image confirmation; fallback to full image pass.
        matched_item, global_match = _try_match_items(item_catalog)

      if args.debug is not None:
        chosen_name = matched_item.get("name", "none") if matched_item is not None else "none"
        debug(
          f"[TRACKBLAZER] Inventory row anchor y={anchor_center_y}, text='{row_text_snippet}', chosen_item='{chosen_name}'"
        )

      if matched_item is None:
        continue

      name = matched_item.get("name", "unknown_item")
      x, y, w, h = global_match

      row_bucket = (name, (y + h // 2) // TRACKBLAZER_INVENTORY_ROW_BUCKET_HEIGHT)
      quantity = _extract_quantity_from_use_item_anchor(list_screen, (anchor_x, anchor_y, anchor_w, anchor_h), item_name=name)
      if quantity <= 1 and not _is_inventory_quantity_row_reliable(list_screen, global_match):
        quantity = 0

      prev_quantity = item_quantities.get(name, 0)
      if row_bucket in seen_item_rows and quantity == prev_quantity:
        continue
      seen_item_rows.add(row_bucket)

      # Ignore duplicate item reads unless quantity changed.
      if name in item_quantities and quantity == prev_quantity:
        continue

      if name in item_quantities and quantity != prev_quantity:
        # High-quality recheck before accepting quantity changes.
        quantity_hq = _extract_quantity_from_use_item_anchor(
          list_screen,
          (anchor_x, anchor_y, anchor_w, anchor_h),
          item_name=name,
        )
        if _is_inventory_quantity_resolved(quantity_hq):
          quantity = quantity_hq

      detected_item_meta[name] = matched_item
      item_quantity_samples[name].append(quantity)
      valid_samples = [sample for sample in item_quantity_samples[name] if _is_inventory_quantity_resolved(sample)]
      if valid_samples:
        quantity_counts = Counter(valid_samples)
        best_qty = sorted(
          quantity_counts.keys(),
          key=lambda value: (quantity_counts[value], value),
          reverse=True,
        )[0]
      else:
        best_qty = 0

      item_quantities[name] = int(best_qty)
      center_x = TRACKBLAZER_INVENTORY_PANEL_LTRB[0] + x + w // 2
      center_y = TRACKBLAZER_INVENTORY_PANEL_LTRB[1] + y + h // 2
      pass_matches.append((center_y, name, center_x, quantity, item_quantities[name]))

      if prev_quantity == 0 or quantity != prev_quantity:
        found_new_this_pass = True

    for center_y, name, center_x, quantity_read, quantity_chosen in sorted(pass_matches, key=lambda row: row[0]):
      if quantity_read != quantity_chosen:
        debug(
          f"[TRACKBLAZER] Inventory match '{name}' at ({center_x}, {center_y}), "
          f"held_read={quantity_read}, held={quantity_chosen}"
        )
      else:
        debug(f"[TRACKBLAZER] Inventory match '{name}' at ({center_x}, {center_y}), held={quantity_chosen}")

    if found_new_this_pass:
      no_progress_scrolls = 0
    else:
      no_progress_scrolls += 1

    if no_progress_scrolls >= 3:
      info("[TRACKBLAZER] Stopping inventory scroll after 3 no-progress passes.")
      break

    if _scroll_trackblazer_inventory_once():
      no_movement_scrolls += 1
    else:
      no_movement_scrolls = 0

    if no_movement_scrolls >= 2:
      break

  _close_trackblazer_overlay(threshold=threshold, context_label="inventory")

  owned_items = []
  for name, quantity in item_quantities.items():
    if quantity <= 0:
      quantity = TRACKBLAZER_INVENTORY_QTY_MIN
    if quantity > TRACKBLAZER_INVENTORY_QTY_MAX:
      warning(f"[TRACKBLAZER] Clamping out-of-range quantity for '{name}': {quantity} -> {TRACKBLAZER_INVENTORY_QTY_MAX}")
      quantity = TRACKBLAZER_INVENTORY_QTY_MAX
    if not _is_inventory_quantity_resolved(quantity):
      warning(
        f"[TRACKBLAZER] Quantity for '{name}' is still high/uncertain after sampling: "
        f"samples={item_quantity_samples.get(name, [])}, using {quantity}"
      )
    owned_items.append(_build_item_entry(detected_item_meta[name], count=quantity))

  owned_items.sort(key=lambda x: (-x["count"], -x["priority"], x["name"]))
  if owned_items:
    inventory_summary = {item.get("name", "unknown_item"): int(item.get("count", 0)) for item in owned_items}
    debug(f"[TRACKBLAZER] Inventory quantities detected: {inventory_summary}")
  else:
    debug("[TRACKBLAZER] Inventory quantities detected: {}")
  return owned_items


class TrackblazerHandler(ScenarioHandler):
  """Default handler for Trackblazer scenario."""

  id = "trackblazer"
  display_name = "Trackblazer"
  aliases = ("trackblazer", "tb", "make a new track", "mant")

  def __init__(self):
    # Preload once so item checks do not repeatedly read from disk.
    preload_trackblazer_item_catalog()
    preload_trackblazer_qty_digit_templates()
    self.expected_inventory_counts = {}
    self.inventory_initialized = False
    self.last_turn_synced = None
    self.last_sync_key = None
    self.previous_turn_was_race_day = True

  def _inventory_counts_from_owned_items(self, owned_items):
    counts = {}
    for item in owned_items:
      name = item.get("name", "unknown_item")
      try:
        qty = int(item.get("count", 0))
      except (TypeError, ValueError):
        qty = 0
      if qty > 0:
        qty = min(max(qty, TRACKBLAZER_INVENTORY_QTY_MIN), TRACKBLAZER_INVENTORY_QTY_MAX)
        counts[name] = qty
    return counts

  def _format_inventory_counts_for_log(self, counts):
    if not counts:
      return {}
    return {name: counts[name] for name in sorted(counts.keys())}

  def _apply_purchases_to_expected_inventory(self, purchased_items):
    if not purchased_items:
      return
    for name in purchased_items:
      next_qty = self.expected_inventory_counts.get(name, 0) + 1
      self.expected_inventory_counts[name] = min(next_qty, TRACKBLAZER_INVENTORY_QTY_MAX)
    info(f"[TRACKBLAZER] Updated expected inventory after purchases: {purchased_items}")

  def _sync_trackblazer_items_for_turn(self, turn_token, check_shop_this_turn=False):
    item_catalog = preload_trackblazer_item_catalog()

    # Inventory OCR only runs once when baseline is uninitialized.
    if not self.inventory_initialized:
      owned_items = collect_trackblazer_owned_items_snapshot(item_catalog)
      actual_counts = self._inventory_counts_from_owned_items(owned_items)
      info(
        f"[TRACKBLAZER] Turn {turn_token} inventory quantities (initial snapshot): "
        f"{self._format_inventory_counts_for_log(actual_counts)}"
      )
      self.expected_inventory_counts = dict(actual_counts)
      self.inventory_initialized = True
      info("[TRACKBLAZER] Expected inventory initialized from first-turn snapshot.")
    else:
      info(
        f"[TRACKBLAZER] Turn {turn_token} inventory check skipped (already initialized). "
        f"Using tracked counts: {self._format_inventory_counts_for_log(self.expected_inventory_counts)}"
      )

    # Shop scan/purchase only on turns immediately after a race turn.
    if not check_shop_this_turn:
      info(f"[TRACKBLAZER] Turn {turn_token} is not post-race; skipping shop check.")
      return

    shop_items, shop_coins, purchased_items = collect_trackblazer_shop_snapshot(
      item_catalog,
      timeline_token=turn_token,
      inventory_counts=self.expected_inventory_counts,
    )
    self._apply_purchases_to_expected_inventory(purchased_items)

    shop_item_names = [item.get("name", "unknown_item") for item in shop_items]
    info(f"[TRACKBLAZER] Turn {turn_token} shop items: {shop_item_names}")
    info(f"[TRACKBLAZER] Turn {turn_token} shop coins: {shop_coins}")

  def stat_gain_region_configs(self) -> list[dict[str, object]]:
    return [
      {
        "region_xywh": TRACKBLAZER_STAT_GAINS_REGION,
        "scale_factor": 1.5,
        "secondary_stat_gains": False,
      },
      {
        "region_xywh": TRACKBLAZER_STAT_GAINS_2_REGION,
        "scale_factor": 1.5,
        "secondary_stat_gains": True,
      },
    ]

  def default_stat_gain_region_configs(self) -> list[dict[str, object]]:
    return [{"region_xywh": TRACKBLAZER_STAT_GAINS_REGION}]

  def race_day_button_templates(self) -> list[str]:
    return ["assets/buttons/race_day_btn.png", TRACKBLAZER_CLIMAX_RACE_BTN_PATH]

  def race_day_entry_action(self, options=None) -> bool:
    _ = options

    clicked_entry = False
    if os.path.exists(TRACKBLAZER_CLIMAX_RACE_BTN_PATH):
      clicked_entry = device_action.locate_and_click(
        TRACKBLAZER_CLIMAX_RACE_BTN_PATH,
        min_search_time=2,
        region_ltrb=constants.SCREEN_BOTTOM_BBOX,
      )

    if not clicked_entry:
      clicked_entry = device_action.locate_and_click(
        "assets/buttons/race_day_btn.png",
        min_search_time=2,
        region_ltrb=constants.SCREEN_BOTTOM_BBOX,
      )

    if not clicked_entry:
      return False

    sleep(0.5)
    device_action.locate_and_click("assets/buttons/ok_btn.png")
    sleep(0.5)

    # TS Climax race flow needs two race-button confirmations (same as URA/Unity behavior).
    for press_index in range(2):
      clicked_this_press = False
      for _ in range(5):
        if device_action.locate_and_click("assets/buttons/race_btn.png", min_search_time=2):
          clicked_this_press = True
          break
        if device_action.locate_and_click("assets/buttons/bluestacks/race_btn.png", min_search_time=2):
          clicked_this_press = True
          break

      if not clicked_this_press:
        warning(f"[TRACKBLAZER] Could not complete race confirmation click #{press_index + 1}.")
        return False

      sleep(0.3)

    sleep(0.5)
    return True

  def on_scenario_detected(self, context=None) -> bool:
    item_catalog = preload_trackblazer_item_catalog()

    # Do not run inventory/shop checks at startup; first per-turn sync will handle initialization.
    self.expected_inventory_counts = {}
    self.inventory_initialized = False
    self.last_turn_synced = None
    self.last_sync_key = None
    self.previous_turn_was_race_day = True

    info(f"[TRACKBLAZER] Catalog items loaded: {len(item_catalog)}")
    info("[TRACKBLAZER] Startup snapshot skipped. Inventory/shop checks will run at turn start.")
    return False

  def collect_main_state_patch(self, state: dict[str, object]) -> dict[str, object]:
    # Use timeline token (e.g. "Junior Year Early Apr") instead of turns-until-goal number.
    turn_token = str(state.get("year", "")).strip()
    if not turn_token:
      return {}

    is_climax_race_turn = False
    if os.path.exists(TRACKBLAZER_CLIMAX_RACE_BTN_PATH):
      is_climax_race_turn = bool(
        device_action.locate(
          TRACKBLAZER_CLIMAX_RACE_BTN_PATH,
          region_ltrb=constants.SCREEN_BOTTOM_BBOX,
        )
      )

    turn_marker = str(state.get("turn", "")).strip()
    sync_key = (turn_token, turn_marker)

    is_shop_reset_day = self.is_trackblazer_shop_reset_day(turn_token)
    check_shop_this_turn = (
      self.previous_turn_was_race_day
      or is_shop_reset_day
    )

    state_patch = {"is_climax_race_turn": is_climax_race_turn}
    if is_climax_race_turn:
      state_patch["turn"] = "Race Day"

    # Dedupe repeated snapshots only when there is no pending reason to check shop.
    if sync_key == self.last_sync_key and not check_shop_this_turn:
      return state_patch

    info(
      f"[TRACKBLAZER] Start-of-turn item sync for timeline token: {turn_token} "
      f"(prev_turn_race={self.previous_turn_was_race_day}, shop_reset_day={is_shop_reset_day}, check_shop_this_turn={check_shop_this_turn})"
    )
    self._sync_trackblazer_items_for_turn(turn_token, check_shop_this_turn=check_shop_this_turn)
    # This flag is set from action selection in the main loop when do_race is chosen.
    self.previous_turn_was_race_day = False
    self.last_turn_synced = turn_token
    self.last_sync_key = sync_key
    return state_patch

  def mark_race_selected_for_next_turn(self):
    self.previous_turn_was_race_day = True

  def is_trackblazer_key_item_usage_day(self, timeline_token) -> bool:
    token = str(timeline_token or "").strip()
    if not token:
      return False

    normalized_token = re.sub(r"\s+", " ", token.lower()).strip()
    # OCR can drop words from this banner; treat any obvious TS Climax underway token as key.
    if "climax" in normalized_token and ("underway" in normalized_token or "races" in normalized_token):
      return True

    key_days = {
      "Classic Year Early Jul",
      "Classic Year Late Jul",
      "Classic Year Early Aug",
      "Classic Year Late Aug",
      "Senior Year Early Jul",
      "Senior Year Late Jul",
      "Senior Year Early Aug",
      "Senior Year Late Aug",
      "TS Climax Races Underway"
    }
    if token in key_days:
      return True

    best_ratio = 0.0
    for key_day in key_days:
      normalized_key_day = re.sub(r"\s+", " ", key_day.lower()).strip()
      ratio = difflib.SequenceMatcher(None, normalized_token, normalized_key_day).ratio()
      if ratio > best_ratio:
        best_ratio = ratio

    # Allow moderate OCR noise while avoiding unrelated-day false positives.
    return best_ratio >= 0.82

  def trackblazer_shop_reset_days(self):
    """Return timeline tokens treated as Trackblazer shop reset days.

    Resets start at Junior Year Early Jul and repeat every 6 timeline steps.
    """
    start_token = "Junior Year Early Jul"
    if start_token not in constants.TIMELINE:
      return []

    start_index = constants.TIMELINE.index(start_token)
    return list(constants.TIMELINE[start_index::6]) # Offset by 6 to align with observed resets starting at Junior Year Early Oct

  def is_trackblazer_shop_reset_day(self, timeline_token) -> bool:
    token = str(timeline_token or "").strip()
    if not token:
      return False
    return token in self.trackblazer_shop_reset_days()

  def _mood_index(self, mood_name):
    mood = str(mood_name or "UNKNOWN").upper()
    try:
      return constants.MOOD_LIST.index(mood)
    except ValueError:
      return constants.MOOD_LIST.index("UNKNOWN")

  def _set_state_mood(self, state, mood_index):
    unknown_index = constants.MOOD_LIST.index("UNKNOWN")
    max_known_index = constants.MOOD_LIST.index("GREAT")
    mood_index = max(0, min(mood_index, max_known_index))
    if mood_index >= unknown_index:
      mood_index = max_known_index

    mood_name = constants.MOOD_LIST[mood_index]
    state["current_mood"] = mood_name
    state["mood_difference"] = mood_index - constants.MOOD_LIST.index(config.MINIMUM_MOOD)
    state["mood_difference_junior_year"] = mood_index - constants.MOOD_LIST.index(config.MINIMUM_MOOD_JUNIOR_YEAR)

  def _plan_energy_items_to_target(
    self,
    current_energy,
    inventory_counts,
    catalog_by_name,
    target_energy=50,
    allow_good_luck_charm=False,
    minimum_remaining_potential_energy=0,
  ):
    energy_item_names = ("Vita 20", "Vita 40", "Vita 65", "Royal Kale Juice")
    current_energy = int(current_energy or 0)
    if current_energy >= target_energy:
      return []

    needed_energy = target_energy - current_energy
    candidate_specs = []
    for name in energy_item_names:
      item = catalog_by_name.get(name)
      if not item:
        continue
      count = int(inventory_counts.get(name, 0) or 0)
      if count <= 0:
        continue
      restore = int(item.get("usage_effects", {}).get("energy_restore", 0) or 0)
      if restore <= 0:
        continue
      candidate_specs.append((name, restore, count))

    if allow_good_luck_charm:
      charm_count = int(inventory_counts.get("Good-Luck Charm", 0) or 0)
      if charm_count > 0:
        candidate_specs.append(("Good-Luck Charm", 50, charm_count))

    total_available_restore = sum(restore * count for _, restore, count in candidate_specs)

    best_plan = None

    def _search(index, chosen_names, total_restore, royal_kale_uses):
      nonlocal best_plan
      if total_restore >= needed_energy:
        remaining_restore = total_available_restore - total_restore
        if remaining_restore < int(minimum_remaining_potential_energy or 0):
          return
        overflow = total_restore - needed_energy
        score = (royal_kale_uses, overflow, len(chosen_names), total_restore)
        if best_plan is None or score < best_plan[0]:
          best_plan = (score, list(chosen_names))
        return

      if index >= len(candidate_specs):
        return

      name, restore, count = candidate_specs[index]
      for qty in range(count + 1):
        extra_names = [name] * qty
        _search(
          index + 1,
          chosen_names + extra_names,
          total_restore + (restore * qty),
          royal_kale_uses + (qty if name == "Royal Kale Juice" else 0),
        )

    _search(0, [], 0, 0)
    if best_plan is None:
      return []
    return best_plan[1]

  def _max_training_total_gain(self, training_state):
    if not isinstance(training_state, dict):
      return 0

    best_total_gain = 0
    for training_data in training_state.values():
      if not isinstance(training_data, dict):
        continue
      stat_gains = training_data.get("stat_gains", {})
      if not isinstance(stat_gains, dict):
        continue
      total_gain = 0
      for gain_value in stat_gains.values():
        try:
          total_gain += int(gain_value)
        except (TypeError, ValueError):
          continue
      if total_gain > best_total_gain:
        best_total_gain = total_gain

    return best_total_gain

  def _has_energy_item_training_trigger(self, training_state):
    """Energy items are allowed only on strong training opportunities.

    Trigger conditions:
    1) Any training has total stat gain > 45.
    2) Any training has >=4 supports at blue/green friendship levels and
       zero supports at other friendship levels.
    """
    if not isinstance(training_state, dict):
      return False

    if self._max_training_total_gain(training_state) > 45:
      return True

    for training_data in training_state.values():
      if not isinstance(training_data, dict):
        continue

      friendship_levels = training_data.get("friendship_levels", {})
      if not isinstance(friendship_levels, dict):
        continue

      blue_count = int(friendship_levels.get("blue", 0) or 0)
      green_count = int(friendship_levels.get("green", 0) or 0)
      gray_count = int(friendship_levels.get("gray", 0) or 0)
      yellow_count = int(friendship_levels.get("yellow", 0) or 0)
      max_count = int(friendship_levels.get("max", 0) or 0)

      if (blue_count + green_count) >= 4 and (gray_count + yellow_count + max_count) == 0:
        return True

    return False

  def _plan_non_key_day_megaphone(self, inventory_counts, min_remaining_after_use=2):
    megaphone_priority = [
      "Empowering Megaphone",
      "Motivating Megaphone",
      "Coaching Megaphone",
    ]
    for name in megaphone_priority:
      count = int(inventory_counts.get(name, 0) or 0)
      if (count - 1) >= int(min_remaining_after_use or 0):
        return [name]
    return []

  def _plan_mood_items_to_great(self, current_mood_name, inventory_counts):
    great_index = constants.MOOD_LIST.index("GREAT")
    current_index = self._mood_index(current_mood_name)
    if current_index >= great_index:
      return []

    available_counts = {
      "Berry Sweet Cupcake": int(inventory_counts.get("Berry Sweet Cupcake", 0) or 0),
      "Plain Cupcake": int(inventory_counts.get("Plain Cupcake", 0) or 0),
    }
    mood_bonus_map = {
      "Berry Sweet Cupcake": 2,
      "Plain Cupcake": 1,
    }

    planned_items = []
    while current_index < great_index:
      needed = great_index - current_index
      candidate_names = [
        name for name, count in available_counts.items()
        if count > 0 and mood_bonus_map[name] <= needed
      ]

      if candidate_names:
        chosen_name = max(candidate_names, key=lambda name: mood_bonus_map[name])
      else:
        overflow_candidates = [name for name, count in available_counts.items() if count > 0]
        if not overflow_candidates:
          break
        chosen_name = min(overflow_candidates, key=lambda name: mood_bonus_map[name])

      planned_items.append(chosen_name)
      available_counts[chosen_name] -= 1
      current_index = min(great_index, current_index + mood_bonus_map[chosen_name])

    return planned_items

  def _plan_condition_items(self, status_effect_names, inventory_counts):
    normalized_conditions = {str(name or "").strip() for name in status_effect_names or [] if str(name or "").strip()}
    if not normalized_conditions:
      return []

    available_counts = Counter({
      name: int(inventory_counts.get(name, 0) or 0)
      for name in (
        "Fluffy Pillow",
        "Pocket Planner",
        "Rich Hand Cream",
        "Smart Scale",
        "Aroma Diffuser",
        "Practice Drills DVD",
        "Miracle Cure",
      )
    })

    direct_item_by_condition = {
      "Night Owl": "Fluffy Pillow",
      "Slacker": "Pocket Planner",
      "Skin Outbreak": "Rich Hand Cream",
      "Slow Metabolism": "Smart Scale",
      "Migraine": "Aroma Diffuser",
      "Practice Poor": "Practice Drills DVD",
    }
    miracle_cure_allowed_conditions = {"Slacker", "Skin Outbreak", "Migraine"}

    planned_items = []
    unresolved_conditions = set(normalized_conditions)
    for condition_name, item_name in direct_item_by_condition.items():
      if condition_name not in unresolved_conditions:
        continue
      if available_counts.get(item_name, 0) <= 0:
        continue
      planned_items.append(item_name)
      available_counts[item_name] -= 1
      unresolved_conditions.discard(condition_name)

    if unresolved_conditions & miracle_cure_allowed_conditions and available_counts.get("Miracle Cure", 0) > 0:
      planned_items.append("Miracle Cure")

    return planned_items

  def _plan_key_day_megaphone(self, timeline_token, inventory_counts):
    if not self.is_trackblazer_key_item_usage_day(timeline_token):
      return []

    megaphone_priority = [
      "Empowering Megaphone",
      "Motivating Megaphone",
      "Coaching Megaphone",
    ]
    for name in megaphone_priority:
      if int(inventory_counts.get(name, 0) or 0) > 0:
        return [name]
    return []

  def _plan_ankle_weights_for_training(self, training_state, inventory_counts, timeline_token, allow_non_key_day=False):
    if (not allow_non_key_day) and (not self.is_trackblazer_key_item_usage_day(timeline_token)):
      return []
    if not isinstance(training_state, dict):
      return []

    ankle_weight_by_training = {
      "spd": "Speed Ankle Weights",
      "sta": "Stamina Ankle Weights",
      "pwr": "Power Ankle Weights",
      "guts": "Guts Ankle Weights",
    }
    best_item_name = None
    best_total_gain = 30

    for training_name, item_name in ankle_weight_by_training.items():
      if int(inventory_counts.get(item_name, 0) or 0) <= 0:
        continue
      training_data = training_state.get(training_name, {})
      stat_gains = training_data.get("stat_gains", {}) if isinstance(training_data, dict) else {}
      if not isinstance(stat_gains, dict):
        continue

      total_gain = 0
      for gain_value in stat_gains.values():
        try:
          total_gain += int(gain_value)
        except (TypeError, ValueError):
          continue

      if total_gain > best_total_gain:
        best_total_gain = total_gain
        best_item_name = item_name

    if best_item_name:
      return [best_item_name]
    return []

  def _is_scheduled_g1_race_day(self, race_schedule_state):
    timeline_token = str(race_schedule_state.get("timeline_token", "") or "")
    scheduled_races_today = race_schedule_state.get("scheduled_races_today", []) or []
    races_today = constants.RACES.get(timeline_token, []) if timeline_token else []

    grade_by_name = {}
    for race in races_today:
      race_name = race.get("name", "")
      if race_name:
        grade_by_name[race_name] = race.get("grade", "")

    for race in scheduled_races_today:
      if not isinstance(race, dict):
        continue
      race_grade = race.get("grade", "")
      if race_grade == "G1":
        return True
      race_name = race.get("name", "")
      if race_name and grade_by_name.get(race_name) == "G1":
        return True

    return False

  def _plan_cleat_hammers_for_scheduled_g1(self, race_schedule_state, inventory_counts):
    if not self._is_scheduled_g1_race_day(race_schedule_state):
      return []

    master_count = int(inventory_counts.get("Master Cleat Hammer", 0) or 0)
    if master_count > 3:
      return ["Master Cleat Hammer"]

    artisan_count = int(inventory_counts.get("Artisan Cleat Hammer", 0) or 0)
    if artisan_count > 0:
      return ["Artisan Cleat Hammer"]

    return []

  def _plan_cleat_hammers_for_climax_race_turn(self, race_schedule_state, inventory_counts):
    if not bool(race_schedule_state.get("is_climax_race_turn", False)):
      return []

    master_count = int(inventory_counts.get("Master Cleat Hammer", 0) or 0)
    if master_count > 0:
      return ["Master Cleat Hammer"]

    artisan_count = int(inventory_counts.get("Artisan Cleat Hammer", 0) or 0)
    if artisan_count > 0:
      return ["Artisan Cleat Hammer"]

    return []

  def _normalize_race_schedule_map(self, race_schedule_raw):
    if isinstance(race_schedule_raw, dict):
      return race_schedule_raw

    if not isinstance(race_schedule_raw, list):
      return {}

    schedule_map = {}
    for race in race_schedule_raw:
      if not isinstance(race, dict):
        continue
      year = str(race.get("year", "")).strip()
      date = str(race.get("date", "")).strip()
      if not year or not date:
        continue
      token = f"{year} {date}"
      if token not in schedule_map:
        schedule_map[token] = []
      schedule_map[token].append(race)

    return schedule_map

  def _is_senior_year_token(self, timeline_token):
    return str(timeline_token or "").startswith("Senior Year ")

  def _remaining_senior_year_training_turns(self, timeline_token, is_current_race_day, race_schedule_map):
    token = str(timeline_token or "").strip()
    if not token:
      return 0
    if token not in constants.TIMELINE:
      return 0

    try:
      senior_start = constants.TIMELINE.index("Senior Year Early Jan")
      senior_end = constants.TIMELINE.index("Senior Year Late Dec")
    except ValueError:
      return 0

    current_index = constants.TIMELINE.index(token)
    if current_index > senior_end:
      return 0

    start_index = max(current_index, senior_start)
    training_turns_remaining = 0
    for idx in range(start_index, senior_end + 1):
      turn_token = constants.TIMELINE[idx]
      if idx == current_index:
        is_race_turn = bool(is_current_race_day)
      else:
        is_race_turn = bool(race_schedule_map.get(turn_token, []))

      if not is_race_turn:
        training_turns_remaining += 1

    return training_turns_remaining

  def _count_training_turn_items(self, inventory_counts):
    training_item_names = (
      "Empowering Megaphone",
      "Motivating Megaphone",
      "Coaching Megaphone",
      "Speed Ankle Weights",
      "Stamina Ankle Weights",
      "Power Ankle Weights",
      "Guts Ankle Weights",
    )
    total = 0
    for name in training_item_names:
      total += int(inventory_counts.get(name, 0) or 0)
    return total

  def _plan_excess_energy_items(self, inventory_counts, catalog_by_name, max_reserve=100):
    energy_item_specs = [
      "Vita 20",
      "Vita 40",
      "Vita 65",
      "Royal Kale Juice",
    ]

    total_stored_energy = 0
    resolved_specs = []
    for name in energy_item_specs:
      item = catalog_by_name.get(name, {})
      restore = int(item.get("usage_effects", {}).get("energy_restore", 0) or 0)
      if restore <= 0:
        continue
      resolved_specs.append((name, restore))
      total_stored_energy += int(inventory_counts.get(name, 0) or 0) * restore

    excess = total_stored_energy - int(max_reserve or 0)
    if excess <= 0:
      return []

    available = sorted(
      [
        [name, restore, int(inventory_counts.get(name, 0) or 0)]
        for name, restore in resolved_specs
        if int(inventory_counts.get(name, 0) or 0) > 0
      ],
      key=lambda spec: spec[1],
    )

    planned = []
    for name, restore, count in available:
      while count > 0 and excess > 0:
        planned.append(name)
        count -= 1
        excess -= restore

      if excess <= 0:
        break

    return planned

  def _apply_item_effects_to_state(self, state, selected_item_names, catalog_by_name):
    if not state:
      return

    current_energy = int(state.get("energy_level", 0) or 0)
    max_energy = int(state.get("max_energy", current_energy) or current_energy)
    current_mood_index = self._mood_index(state.get("current_mood", "UNKNOWN"))
    great_index = constants.MOOD_LIST.index("GREAT")
    status_effect_names = [str(name) for name in state.get("status_effect_names", []) or []]

    for name in selected_item_names:
      item = catalog_by_name.get(name)
      if not item:
        continue
      usage_effects = item.get("usage_effects", {})

      energy_restore = int(usage_effects.get("energy_restore", 0) or 0)
      if name == "Good-Luck Charm":
        energy_restore = max(energy_restore, 50)
      if energy_restore > 0:
        current_energy = min(max_energy, current_energy + energy_restore)

      mood_bonus = int(usage_effects.get("mood_bonus", 0) or 0)
      if mood_bonus != 0:
        current_mood_index = max(0, min(great_index, current_mood_index + mood_bonus))

      if name == "Fluffy Pillow":
        status_effect_names = [condition for condition in status_effect_names if condition != "Night Owl"]
      elif name == "Pocket Planner":
        status_effect_names = [condition for condition in status_effect_names if condition != "Slacker"]
      elif name == "Rich Hand Cream":
        status_effect_names = [condition for condition in status_effect_names if condition != "Skin Outbreak"]
      elif name == "Smart Scale":
        status_effect_names = [condition for condition in status_effect_names if condition != "Slow Metabolism"]
      elif name == "Aroma Diffuser":
        status_effect_names = [condition for condition in status_effect_names if condition != "Migraine"]
      elif name == "Practice Drills DVD":
        status_effect_names = [condition for condition in status_effect_names if condition != "Practice Poor"]
      elif name == "Miracle Cure":
        status_effect_names = [
          condition for condition in status_effect_names
          if condition not in {"Slacker", "Skin Outbreak", "Migraine"}
        ]

    state["energy_level"] = current_energy
    state["status_effect_names"] = status_effect_names
    self._set_state_mood(state, current_mood_index)

  def plan_items_to_use_this_turn(self, training_state, race_schedule_state):
    """Scaffold for Trackblazer item-use planning.

    Args:
      training_state: Usually state["training_results"] from the current turn.
      race_schedule_state: Dict-like schedule context for the current timeline,
        including today's scheduled races and full schedule if needed.

    Returns:
      dict: {
        "item_names": list[str],
        "recheck_trainings": bool,
      }
    """
    item_catalog = preload_trackblazer_item_catalog()
    catalog_by_name = {item.get("name", ""): item for item in item_catalog}
    inventory_counts = self.expected_inventory_counts
    planned_item_names = []
    timeline_token = race_schedule_state.get("timeline_token", "")
    is_key_day = self.is_trackblazer_key_item_usage_day(timeline_token)
    is_race_day = bool(race_schedule_state.get("is_race_day", False))
    scheduled_races_today = race_schedule_state.get("scheduled_races_today", []) or []
    has_scheduled_race_today = bool(scheduled_races_today)
    is_climax_underway_window = ("climax" in str(timeline_token or "").lower()) and (
      "underway" in str(timeline_token or "").lower() or "races" in str(timeline_token or "").lower()
    )
    race_schedule_map = self._normalize_race_schedule_map(race_schedule_state.get("race_schedule", {}))
    remaining_senior_training_turns = self._remaining_senior_year_training_turns(
      timeline_token,
      is_race_day,
      race_schedule_map,
    )
    training_items_to_keep = remaining_senior_training_turns + 3
    training_item_count = self._count_training_turn_items(inventory_counts)
    excess_training_items = max(0, training_item_count - training_items_to_keep)

    megaphone_names = [
      "Empowering Megaphone",
      "Motivating Megaphone",
      "Coaching Megaphone",
    ]
    total_megaphones = sum(int(inventory_counts.get(name, 0) or 0) for name in megaphone_names)

    total_cleat_hammers = (
      int(inventory_counts.get("Master Cleat Hammer", 0) or 0)
      + int(inventory_counts.get("Artisan Cleat Hammer", 0) or 0)
    )

    if self._is_senior_year_token(timeline_token):
      debug(
        "[TRACKBLAZER] Senior-year item burn status: "
        f"remaining_training_turns={remaining_senior_training_turns}, "
        f"training_items={training_item_count}, keep_target={training_items_to_keep}, "
        f"excess_training_items={excess_training_items}, megaphones={total_megaphones}, "
        f"cleat_hammers={total_cleat_hammers}"
      )

    current_energy = int(race_schedule_state.get("energy_level", 0) or 0)
    max_energy = int(race_schedule_state.get("max_energy", current_energy) or current_energy)
    current_mood_name = race_schedule_state.get("current_mood", "UNKNOWN")
    status_effect_names = list(race_schedule_state.get("status_effect_names", []) or [])
    max_training_failure = 0
    if isinstance(training_state, dict):
      for training_data in training_state.values():
        if not isinstance(training_data, dict):
          continue
        try:
          failure = int(training_data.get("failure", 0) or 0)
        except (TypeError, ValueError):
          continue
        if failure > max_training_failure:
          max_training_failure = failure

    max_training_total_gain = self._max_training_total_gain(training_state)
    training_items_allowed = ((not is_race_day) and (not has_scheduled_race_today)) or is_climax_underway_window

    condition_item_names = self._plan_condition_items(status_effect_names, inventory_counts)
    planned_item_names.extend(condition_item_names)

    if (
      training_items_allowed
      and is_key_day
      and max_training_total_gain < 45
      and int(inventory_counts.get("Reset Whistle", 0) or 0) > 0
    ):
      debug(
        "[TRACKBLAZER] Planning Reset Whistle on weak key training turn: "
        f"timeline={timeline_token}, max_training_total_gain={max_training_total_gain}"
      )
      planned_item_names.append("Reset Whistle")
      return {
        "item_names": planned_item_names,
        "recheck_trainings": True,
      }

    # Emergency race-day recovery: if energy is empty, force an immediate MAX drink.
    if is_race_day and current_energy <= 0:
      emergency_name = "Energy Drink MAX"
      if int(inventory_counts.get(emergency_name, 0) or 0) > 0:
        planned_item_names.append(emergency_name)
        usage_effects = catalog_by_name.get(emergency_name, {}).get("usage_effects", {})
        energy_restore = int(usage_effects.get("energy_restore", 0) or 0)
        current_energy = min(
          max_energy,
          current_energy + max(0, energy_restore),
        )

    if self._is_senior_year_token(timeline_token):
      # Keep at most +100 energy in reserve by converting excess consumables now.
      excess_energy_item_names = self._plan_excess_energy_items(
        inventory_counts,
        catalog_by_name,
        max_reserve=100,
      )
      planned_item_names.extend(excess_energy_item_names)
      for name in excess_energy_item_names:
        usage_effects = catalog_by_name.get(name, {}).get("usage_effects", {})
        energy_restore = int(usage_effects.get("energy_restore", 0) or 0)
        current_energy = min(max_energy, current_energy + max(0, energy_restore))

    # Only use training-only items on actual training turns. Scheduled race dates are treated as race turns
    # here because the handler runs before the main race-selection logic.
    if training_items_allowed:
      if is_key_day:
        planned_item_names.extend(self._plan_key_day_megaphone(timeline_token, inventory_counts))
      elif max_training_total_gain > 42:
        planned_item_names.extend(self._plan_non_key_day_megaphone(inventory_counts, min_remaining_after_use=2))

      has_megaphone_planned = any(name in megaphone_names for name in planned_item_names)
      if total_megaphones > 3 and not has_megaphone_planned:
        planned_item_names.extend(self._plan_non_key_day_megaphone(inventory_counts, min_remaining_after_use=3))

      ankle_weight_item_names = self._plan_ankle_weights_for_training(
        training_state,
        inventory_counts,
        timeline_token,
        allow_non_key_day=(
          (not is_key_day and max_training_total_gain > 42)
          or (excess_training_items > 0)
        ),
      )
      if not is_key_day and max_training_total_gain > 42:
        filtered_ankle_items = []
        for name in ankle_weight_item_names:
          count = int(inventory_counts.get(name, 0) or 0)
          if (count - 1) >= 2:
            filtered_ankle_items.append(name)
        ankle_weight_item_names = filtered_ankle_items
      planned_item_names.extend(ankle_weight_item_names)

    cleat_hammer_item_names = self._plan_cleat_hammers_for_climax_race_turn(race_schedule_state, inventory_counts)
    if not cleat_hammer_item_names:
      cleat_hammer_item_names = self._plan_cleat_hammers_for_scheduled_g1(race_schedule_state, inventory_counts)
    if (
      not cleat_hammer_item_names
      and total_cleat_hammers > 3
      and self._is_scheduled_g1_race_day(race_schedule_state)
    ):
      master_count = int(inventory_counts.get("Master Cleat Hammer", 0) or 0)
      artisan_count = int(inventory_counts.get("Artisan Cleat Hammer", 0) or 0)
      if master_count > 0:
        cleat_hammer_item_names = ["Master Cleat Hammer"]
      elif artisan_count > 0:
        cleat_hammer_item_names = ["Artisan Cleat Hammer"]
    planned_item_names.extend(cleat_hammer_item_names)

    for item in item_catalog:
      name = item.get("name", "")
      if not name:
        continue
      if inventory_counts.get(name, 0) <= 0:
        continue

      normalized_name = name.lower()
      is_notepad = normalized_name.endswith("notepad")
      is_manual = normalized_name.endswith("manual")
      is_scroll = normalized_name.endswith("scroll")
      is_grilled_carrots = normalized_name == "grilled carrots"

      if is_notepad or is_manual or is_scroll or is_grilled_carrots:
        planned_item_names.append(name)

    potential_energy = current_energy
    potential_energy_item_names = ["Vita 20", "Vita 40", "Vita 65", "Royal Kale Juice"]
    if max_training_failure > 5:
      potential_energy_item_names.append("Good-Luck Charm")

    for name in potential_energy_item_names:
      item = catalog_by_name.get(name)
      if not item:
        continue
      count = int(inventory_counts.get(name, 0) or 0)
      restore = int(item.get("usage_effects", {}).get("energy_restore", 0) or 0)
      if name == "Good-Luck Charm":
        restore = max(restore, 50)
      potential_energy += count * restore

    energy_item_training_trigger = self._has_energy_item_training_trigger(training_state)
    should_plan_energy_items = current_energy < 45 and potential_energy >= 50 and energy_item_training_trigger
    if should_plan_energy_items:
      preferred_energy_stockpile = 100 if potential_energy >= 100 else 0
      energy_item_names = self._plan_energy_items_to_target(
        current_energy,
        inventory_counts,
        catalog_by_name,
        target_energy=50,
        allow_good_luck_charm=(max_training_failure > 5),
        minimum_remaining_potential_energy=preferred_energy_stockpile,
      )
      if not energy_item_names and preferred_energy_stockpile > 0:
        # If holding a +100 reserve is not feasible this turn, fall back to no-reserve planning.
        energy_item_names = self._plan_energy_items_to_target(
          current_energy,
          inventory_counts,
          catalog_by_name,
          target_energy=50,
          allow_good_luck_charm=(max_training_failure > 5),
          minimum_remaining_potential_energy=0,
        )
      planned_item_names.extend(energy_item_names)
      for name in energy_item_names:
        item = catalog_by_name.get(name, {})
        usage_effects = item.get("usage_effects", {})
        energy_restore = int(usage_effects.get("energy_restore", 0) or 0)
        if name == "Good-Luck Charm":
          energy_restore = max(energy_restore, 50)
        current_energy = min(
          max_energy,
          current_energy + energy_restore,
        )
        current_mood_name_index = self._mood_index(current_mood_name)
        mood_bonus = int(usage_effects.get("mood_bonus", 0) or 0)
        if mood_bonus != 0:
          current_mood_name_index = max(0, min(constants.MOOD_LIST.index("GREAT"), current_mood_name_index + mood_bonus))
          current_mood_name = constants.MOOD_LIST[current_mood_name_index]

    mood_item_names = self._plan_mood_items_to_great(current_mood_name, inventory_counts)
    planned_item_names.extend(mood_item_names)

    has_energy_item = any(
      (
        int(catalog_by_name.get(name, {}).get("usage_effects", {}).get("energy_restore", 0) or 0) > 0
        or name == "Good-Luck Charm"
      )
      for name in planned_item_names
    )
    recheck_trainings = has_energy_item or any(
      name in {"Reset Whistle", "Good-Luck Charm"}
      for name in planned_item_names
    )

    return {
      "item_names": planned_item_names,
      "recheck_trainings": recheck_trainings,
    }

  def decide_handler_action(self, state: dict[str, object]):
    training_state = state.get("training_results", {})
    timeline_token = str(state.get("year", ""))
    normalized_timeline_token = re.sub(r"\s+", " ", timeline_token.lower()).strip()
    is_climax_underway_window = (
      "climax" in normalized_timeline_token
      and ("underway" in normalized_timeline_token or "races" in normalized_timeline_token)
    )
    is_climax_race_turn = bool(state.get("is_climax_race_turn", False))
    is_race_day = bool(str(state.get("turn", "")).strip() == "Race Day")
    # TS Climax banner can represent either a training turn or race turn; use button detection as source of truth.
    if is_climax_underway_window:
      is_race_day = is_climax_race_turn

    race_schedule_map = self._normalize_race_schedule_map(config.RACE_SCHEDULE)
    race_schedule_state = {
      "timeline_token": timeline_token,
      "scheduled_races_today": race_schedule_map.get(timeline_token, []),
      "race_schedule": race_schedule_map,
      "is_climax_race_turn": is_climax_race_turn,
      "is_race_day": is_race_day,
      "energy_level": state.get("energy_level", 0),
      "max_energy": state.get("max_energy", 0),
      "current_mood": state.get("current_mood", "UNKNOWN"),
      "status_effect_names": list(state.get("status_effect_names", []) or []),
    }
    item_plan = self.plan_items_to_use_this_turn(training_state, race_schedule_state)
    planned_item_names = item_plan.get("item_names", []) if isinstance(item_plan, dict) else []
    recheck_trainings = bool(item_plan.get("recheck_trainings", False)) if isinstance(item_plan, dict) else False
    if not planned_item_names:
      debug("[TRACKBLAZER] decide_handler_action: planner returned no items for this turn.")
      return None

    item_catalog = preload_trackblazer_item_catalog()
    catalog_by_name = {item.get("name", ""): item for item in item_catalog}

    items_to_use = [
      catalog_by_name[name]
      for name in planned_item_names
      if name in catalog_by_name
    ]
    missing_item_names = [name for name in planned_item_names if name not in catalog_by_name]
    if missing_item_names:
      warning(f"[TRACKBLAZER] Planner returned unknown catalog items: {missing_item_names}")

    usable = [item for item in items_to_use if self.expected_inventory_counts.get(item.get("name", ""), 0) > 0]
    if not usable:
      debug(
        "[TRACKBLAZER] decide_handler_action: no usable items from planner list: "
        f"{[item.get('name', 'unknown_item') for item in items_to_use]}"
      )
      return None

    from core.actions import Action
    action = Action()
    action.func = "use_trackblazer_items"
    action._callable = lambda opts: self._execute_use_items(opts)
    action["items_to_use"] = list(usable)
    action["state"] = state
    action["recheck_trainings"] = recheck_trainings
    action["continue_turn_after_run"] = True
    return action

  def _execute_use_items(self, options):
    items = options.get("items_to_use", [])
    if not items:
      return False
    if not _open_trackblazer_inventory():
      warning("[TRACKBLAZER] _execute_use_items: could not open inventory.")
      return False
    item_catalog = preload_trackblazer_item_catalog()
    catalog_by_name = {item.get("name", ""): item for item in item_catalog}
    pending_items = Counter(item.get("name", "unknown_item") for item in items)
    selected_names = []

    max_inventory_scrolls = 20
    for scroll_index in range(max_inventory_scrolls + 1):
      if not pending_items:
        break

      visible_items = _collect_trackblazer_visible_inventory_items(item_catalog)
      visible_names = [entry.get("name", "unknown_item") for entry in visible_items]
      debug(f"[TRACKBLAZER] Inventory scan {scroll_index + 1} visible items: {visible_names}")

      selected_this_pass = []
      for entry in visible_items:
        name = entry.get("name", "unknown_item")
        remaining_uses = int(pending_items.get(name, 0) or 0)
        if remaining_uses <= 0:
          continue

        for _ in range(remaining_uses):
          device_action.click(entry["use_center"], text=f"[TRACKBLAZER] Selecting '{name}' for use")
          debug(f"[TRACKBLAZER] Queued '{name}' for item-use confirmation on scan {scroll_index + 1}.")
          selected_names.append(name)
          selected_this_pass.append(name)
          pending_items[name] -= 1
          if pending_items[name] <= 0:
            pending_items.pop(name, None)
          sleep(0.15)

      if not pending_items:
        break

      debug(
        f"[TRACKBLAZER] Pending item-use selections after scan {scroll_index + 1}: {sorted(pending_items.keys())}"
      )

      if scroll_index >= max_inventory_scrolls:
        warning(
          "[TRACKBLAZER] Inventory item-use search reached safety limit "
          f"({max_inventory_scrolls} scrolls)."
        )
        break

      if _scroll_trackblazer_inventory_once():
        debug("[TRACKBLAZER] Inventory did not move while searching additional item-use pages.")
        break

    used_any = False
    if selected_names:
      _confirm_trackblazer_item_use()
      self._apply_item_effects_to_state(options.get("state"), selected_names, catalog_by_name)
      for name in selected_names:
        new_count = max(0, self.expected_inventory_counts.get(name, 1) - 1)
        if new_count <= 0:
          self.expected_inventory_counts.pop(name, None)
          info(f"[TRACKBLAZER] Used '{name}'. Removed from expected inventory (remaining: 0).")
        else:
          self.expected_inventory_counts[name] = new_count
          info(f"[TRACKBLAZER] Used '{name}'. Estimated remaining: {new_count}")
      used_any = True
    elif pending_items:
      debug(f"[TRACKBLAZER] Could not locate pending item-use targets: {sorted(pending_items.keys())}")

    _close_trackblazer_overlay(context_label="inventory")
    return used_any

  def on_turn_read(self, turn_text: str) -> None:
    # Intentionally no-op: Trackblazer sync keys off (state["year"], state["turn"]).
    return

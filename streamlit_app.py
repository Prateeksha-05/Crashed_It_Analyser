import os
import cv2
import numpy as np
from ultralytics import YOLO
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Load YOLOv8 model
model = YOLO("yolov8n.pt")


def analyze_traffic_violations(detected_vehicles, total_count):
  violations = []

  if total_count >= 2:
    violations.append(
        "- Failure to maintain safe following distance (Rear-end collision"
        " pattern)"
    )
    violations.append("- Potential delayed reaction time / distracted driving")
    violations.append("- Intersection right-of-way violation or improper lane merging")
  elif total_count == 1:
    violations.append("- Single-vehicle loss of control")
    violations.append("- Driving too fast for road/weather conditions")
    violations.append("- Potential fixed-object or roadside hazard collision")
  else:
    violations.append("- Insufficient vehicle telemetry detected in frame")

  heavy_damages = [v for v in detected_vehicles if v["condition"] == "Heavy Damage"]
  moderate_damages = [
      v for v in detected_vehicles if v["condition"] == "Moderate Damage"
  ]

  if len(heavy_damages) > 0:
    violations.append(
        "- High-velocity impact indicator (Severe structural deformation)"
    )
    violations.append("- Excessive speeding or failure to yield right-of-way")
  elif len(moderate_damages) > 0:
    violations.append("- Moderate kinetic transfer (Failure to brake adequately)")
  else:
    violations.append("- Low-impact or minor contact friction indicator")

  return violations


def assess_damage_opencv(vehicle_crop):
  if vehicle_crop.size == 0:
    return {"condition": "Unknown", "density": 0.0}

  vehicle_crop = cv2.resize(vehicle_crop, (200, 200))
  gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

  h, w = gray.shape
  mask = np.zeros_like(gray)
  margin_h, margin_w = int(h * 0.15), int(w * 0.15)
  mask[margin_h : h - margin_h, margin_w : w - margin_w] = 255

  masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
  blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
  edges = cv2.Canny(blurred, 80, 200)

  active_pixels = np.sum(mask > 0)
  edge_density = np.sum(edges > 0) / active_pixels if active_pixels > 0 else 0

  car_body = gray[margin_h : h - margin_h, margin_w : w - margin_w]
  bh, bw = car_body.shape
  if bh > 0 and bw > 1:
    left_half = car_body[:, : bw // 2]
    right_half = cv2.flip(car_body[:, bw // 2 :], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
      left_half = left_half[:, :min_w]
      right_half = right_half[:, :min_w]
      diff = cv2.absdiff(left_half, right_half)
      asymmetry_score = np.mean(diff) / 255.0
    else:
      asymmetry_score = 0.0
  else:
    asymmetry_score = 0.0

  damage_score = (edge_density * 1.2) + (asymmetry_score * 0.6)

  if damage_score > 0.14:
    condition = "Heavy Damage"
  elif damage_score > 0.08:
    condition = "Moderate Damage"
  else:
    condition = "Minor / Intact"

  return {"condition": condition, "density": round(damage_score, 3)}


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
  # Notify user processing has started
  status_msg = await update.message.reply_text(
      "🔍 Analyzing accident scene with YOLOv8 & OpenCV..."
  )

  # Download image from Telegram
  photo_file = await update.message.photo[-1].get_file()
  input_path = "input_temp.jpg"
  output_path = "output_annotated.jpg"
  await photo_file.download_to_drive(input_path)

  # Process image with OpenCV and YOLOv8
  image_cv = cv2.imread(input_path)
  H, W, _ = image_cv.shape

  max_dimension = 800
  if max(H, W) > max_dimension:
    scale = max_dimension / float(max(H, W))
    image_cv = cv2.resize(image_cv, (int(W * scale), int(H * scale)))

  H, W, _ = image_cv.shape
  vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

  results = model(image_cv, conf=0.4, iou=0.45)
  detected_vehicles = []
  raw_detections = []

  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0])
      if cls_id in vehicle_classes:
        class_name = model.names[cls_id].capitalize()
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
          continue
        raw_detections.append(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": class_name}
        )

  raw_detections = sorted(raw_detections, key=lambda d: d["x1"])
  vehicle_count = 0
  summary_lines = []

  for det in raw_detections:
    vehicle_count += 1
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    class_name = det["class"]

    vehicle_crop = image_cv[y1:y2, x1:x2]
    damage_info = assess_damage_opencv(vehicle_crop)

    detected_vehicles.append({
        "id": vehicle_count,
        "type": class_name,
        "condition": damage_info["condition"],
        "density": damage_info["density"],
    })

    summary_lines.append(
        f"• Vehicle {vehicle_count} ({class_name}): `{damage_info['condition']}`"
        f" (Score: {damage_info['density']})"
    )

    # Draw bounding box and label
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        image_cv,
        f"V{vehicle_count}: {class_name}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

  violations = analyze_traffic_violations(detected_vehicles, len(detected_vehicles))
  cv2.imwrite(output_path, image_cv)

  # Format caption for Telegram
  caption = (
      "🚨 *CrashEdit Forensic Report*\n\n"
      f"*Total Vehicles Detected:* {len(detected_vehicles)}\n\n"
      "*🚙 Vehicle Breakdown:*\n"
      + ("\n".join(summary_lines) if summary_lines else "None detected")
      + "\n\n*⚖️ Potential Violations:*\n"
      + ("\n".join(violations) if violations else "None detected")
  )

  # Send back the annotated image and report
  await update.message.reply_photo(
      photo=open(output_path, "rb"), caption=caption, parse_mode="Markdown"
  )

  # Clean up temp files
  if os.path.exists(input_path):
    os.remove(input_path)
  if os.path.exists(output_path):
    os.remove(output_path)


def main():
  TOKEN = "8939155184:AAH2d5MM4xwqsWnIF5Yus6QdiiZj4UvRa54"  # Replace with your token from BotFather
  app = ApplicationBuilder().token(TOKEN).build()
  app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
  print("🤖 CrashEdit Telegram Bot is running and waiting for photos...")
  app.run_polling()


if __name__ == "__main__":
  main()

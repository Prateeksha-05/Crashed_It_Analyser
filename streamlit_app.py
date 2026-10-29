from io import BytesIO
import re
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="CrashedIt",
    page_icon="🚗",
    layout="centered",
)

# Custom CSS Injection for a professional dashboard look
st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #e3736b;
    }
    
    /* Optional: Make card/container backgrounds white for great contrast */
    div.stMarkdown, .stAlert {
        border-radius: 10px;
    }
    
    /* Header typography for high readability on the new background */
    h1, h2, h3, p, label {
        color: #1f2937 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""",
    unsafe_allow_html=True,
)
st.title("🚗 CrashedIt ")
st.write(
    "Upload an accident scene photo or paste a public Google Drive link"
    "or send you images to our telegram Chatbot :crashedit_bot!!"
)


# Load YOLOv8 model with Streamlit caching so it loads only once into RAM
@st.cache_resource
def load_model():
  return YOLO("yolov8n.pt")


model = load_model()


def analyze_traffic_violations(detected_vehicles, total_count):
   violations = []

  if total_count >= 2:
    violations.append("- Failure to maintain safe following distance (Rear-end collision pattern)")
    violations.append("- Potential delayed reaction time / distracted driving")
    violations.append("- Intersection right-of-way violation or improper lane merging")
  elif total_count == 1:
    violations.append("- Single-vehicle loss of control")
    violations.append("- Driving too fast for road/weather conditions")
    violations.append("- Potential fixed-object or roadside hazard collision")
  else:
    violations.append("- Insufficient vehicle telemetry detected in frame")

  heavy_damages = [v for v in detected_vehicles if v['condition'] == 'Heavy Damage']
  moderate_damages = [v for v in detected_vehicles if v['condition'] == 'Moderate Damage']

  if len(heavy_damages) > 0:
    violations.append("- High-velocity impact indicator (Severe structural deformation)")
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


# --- INPUT SELECTION: File Uploader OR Google Drive Link ---
input_method = st.radio(
    "Select Image Input Method:", ("Upload File", "Google Drive Public Link")
)

image_pil = None

if input_method == "Upload File":
  uploaded_file = st.file_uploader(
      "Choose an accident scene image...", type=["jpg", "jpeg", "png", "webp"]
  )
  if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
else:
  gdrive_url = st.text_input(
      "Paste Public Google Drive Image Link (Ensure permission is 'Anyone with"
      " the link'):",
      "",
  )
  if gdrive_url:
    file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", gdrive_url)
    if file_id_match:
      file_id = file_id_match.group(1)
      download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
      try:
        response = requests.get(download_url)
        if response.status_code == 200:
          image_pil = Image.open(BytesIO(response.content))
          st.success("Successfully fetched image from Google Drive!")
        else:
          st.error(
              "Failed to download. Make sure the file link permissions are"
              " public."
          )
      except Exception as e:
        st.error(f"Error fetching from Google Drive: {e}")
    else:
      st.warning(
          "Invalid Google Drive link format. Ensure it contains `/file/d/...`"
      )

# --- EXECUTE PROCESSING IF IMAGE IS AVAILABLE ---
if image_pil is not None:
  image_np = np.array(image_pil)
  image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  H, W, _ = image_cv.shape

  # Smart resizing to maintain performance
  max_dimension = 800
  if max(H, W) > max_dimension:
    scale = max_dimension / float(max(H, W))
    image_cv = cv2.resize(image_cv, (int(W * scale), int(H * scale)))

  H, W, _ = image_cv.shape
  vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

  with st.spinner("Analyzing accident scene ..."):
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

      # Draw bounding boxes and labels
      cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
      label = f"Vehicle {vehicle_count}: {class_name}"
      cv2.putText(
          image_cv,
          label,
          (x1, max(y1 - 10, 20)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )

    annotated_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    violations = analyze_traffic_violations(
        detected_vehicles, len(detected_vehicles)
    )

    # Display Results in Streamlit UI
    st.markdown("---")
    st.subheader("🔍 Forensic Analysis Report")

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
      st.image(
          annotated_rgb,
          caption="Annotated Accident Scene Breakdown",
          use_container_width=True,
      )

    with col2:
      st.metric(label="Total Vehicles Detected", value=len(detected_vehicles))
      st.markdown("### 🚙 Vehicle Breakdown")
      for v in detected_vehicles:
        st.write(f"- **V{v['id']} ({v['type']})**: `{v['condition']}`")

    st.markdown("### ⚖️ Potential Violations & Indicators")
    for viol in violations:
      st.write(viol)

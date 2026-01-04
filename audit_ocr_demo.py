import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# Gemini (Google GenAI SDK)
from google import genai  # pip: google-genai


# =========================
# Config
# =========================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "audit_nameplate_records.csv"

DEFAULT_MODEL = "gemini-2.5-flash"

AUDIT_TEMPLATES = {
    "Hotels": [
        "Lobby11",
        "Rooms",
        "Kitchen",
        "Pool",
        "Boiler system",
        "Electrical system",
        "Generator system",
        "Laundry",
    ],
    "Factories": [
        "Production Lines",
        "Workshop / Maintenance",
        "Warehouse",
        "Compressor system",
        "Boiler system",
        "Electrical system",
        "Generator Room",
        "Office Area",
        "Outdoor / Yard",
        "Chiller Plant",
        "Chilled Water Network",
        "Compressed Air Network",
        "Hot Water Network",
        "Steam Network",
        "Steam Generator",
    ],
    "Schools": [
        "Classrooms",
        "Computer Labs",
        "Science Labs",
        "Administration",
        "Library",
        "Cafeteria",
        "Sports Hall / Outdoor Area",
        "Electrical system",
        "Generator system",
    ],
    "Hospitals": [
        "ER",
        "ICU",
        "Wards",
        "Operating Theatres",
        "Labs",
        "Imaging Area",
        "Pharmacy",
        "Kitchen",
        "Laundry",
        "Boiler system",
        "Electrical system",
        "Generator system",
    ],
}

FIELDS = ["model", "serial", "voltage_v", "current_a", "power_kw", "frequency_hz"]


# =========================
# Helpers
# =========================
def get_gemini_client():
    key = None
    if "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"]
    if not key:
        key = st.session_state.get("GEMINI_API_KEY") or None

    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing in Streamlit Secrets.")

    return genai.Client(api_key=key)


def get_model_name():
    return st.secrets.get("GEMINI_MODEL", DEFAULT_MODEL)


def safe_open_image(uploaded_file) -> Image.Image:
    data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


def resize_max(img: Image.Image, max_side=1400) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w * scale), int(h * scale)))


def enhance_for_reading(img: Image.Image) -> Image.Image:
    img = resize_max(img, 1600)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    return img


def simple_edge_crop(img: Image.Image) -> Image.Image:
    """
    Lightweight "best effort" crop to focus on dense text area.
    If it fails, returns original.
    """
    try:
        small = resize_max(img, 900)
        gray = np.array(small.convert("L"), dtype=np.float32)

        # simple gradient magnitude (no cv2)
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        g = np.pad(gx, ((0, 0), (0, 1)), mode="constant") + np.pad(
            gy, ((0, 1), (0, 0)), mode="constant"
        )

        thr = np.percentile(g, 95)
        mask = g > thr

        if mask.sum() < 500:  # not enough edges
            return img

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        # add margin
        margin = 20
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(gray.shape[0] - 1, y1 + margin)
        x1 = min(gray.shape[1] - 1, x1 + margin)

        cropped_small = small.crop((x0, y0, x1, y1))

        # map crop back to original scale
        sw, sh = small.size
        ow, oh = img.size
        scale_x = ow / sw
        scale_y = oh / sh

        ox0 = int(x0 * scale_x)
        oy0 = int(y0 * scale_y)
        ox1 = int(x1 * scale_x)
        oy1 = int(y1 * scale_y)

        # final crop (with safety)
        ox0 = max(0, min(ox0, ow - 1))
        oy0 = max(0, min(oy0, oh - 1))
        ox1 = max(1, min(ox1, ow))
        oy1 = max(1, min(oy1, oh))

        if ox1 - ox0 < 50 or oy1 - oy0 < 50:
            return img

        return img.crop((ox0, oy0, ox1, oy1))
    except Exception:
        return img


def extract_json_anyhow(text: str):
    # remove ```json fences
    t = text.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```$", "", t).strip()

    # best: direct json
    try:
        return json.loads(t)
    except Exception:
        pass

    # fallback: take first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def regex_fill_from_plate_text(fields: dict) -> dict:
    txt = (fields.get("plate_text") or "").replace("\n", " ")

    def find(pattern):
        m = re.search(pattern, txt, flags=re.IGNORECASE)
        return m.group(1) if m else None

    # Voltage
    if not fields.get("voltage_v"):
        v = find(r"(\d{2,4})\s*V\b")
        if v:
            fields["voltage_v"] = v

    # Frequency
    if not fields.get("frequency_hz"):
        hz = find(r"(\d{2,3})\s*Hz\b")
        if hz:
            fields["frequency_hz"] = hz

    # Current
    if not fields.get("current_a"):
        a = find(r"(\d+(?:\.\d+)?)\s*A\b")
        if a:
            fields["current_a"] = a

    # Power
    if not fields.get("power_kw"):
        kw = find(r"(\d+(?:\.\d+)?)\s*kW\b")
        if kw:
            fields["power_kw"] = kw
        else:
            w = find(r"(\d+(?:\.\d+)?)\s*W\b")
            if w:
                try:
                    fields["power_kw"] = str(round(float(w) / 1000.0, 4))
                except Exception:
                    pass

    return fields


def gemini_extract_fields(pil_img: Image.Image):
    client = get_gemini_client()
    model = get_model_name()

    # Make 2 variants: cropped + enhanced (helps when plate is small)
    cropped = simple_edge_crop(pil_img)
    enhanced = enhance_for_reading(cropped)

    prompt = (
        "You are reading an equipment NAMEPLATE. Extract these fields if visible:\n"
        "- model\n- serial\n- voltage (V)\n- current (A)\n- power (kW)\n- frequency (Hz)\n\n"
        "Rules:\n"
        "1) If power is in W, convert to kW (W/1000).\n"
        "2) If a value is not visible, use null.\n"
        "3) Also return a short 'plate_text' (max ~10 lines) of what you can read.\n"
        "Return ONLY valid JSON."
    )

    # JSON Schema (Structured Output)
    schema = {
        "type": "object",
        "properties": {
            "model": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "serial": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "voltage_v": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "current_a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "power_kw": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "frequency_hz": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "plate_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "model",
            "serial",
            "voltage_v",
            "current_a",
            "power_kw",
            "frequency_hz",
            "plate_text",
        ],
        "additionalProperties": False,
    }

    # Gemini performs better with image first in single-image prompts
    # and we force JSON using response_mime_type + schema. :contentReference[oaicite:1]{index=1}
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[enhanced, prompt],
            config={
                "temperature": 0,
                "max_output_tokens": 256,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        raw = resp.text or ""
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}")

    data = extract_json_anyhow(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"Gemini returned no JSON. Raw: {raw[:400]}")

    # Ensure keys exist
    for k in ["model", "serial", "voltage_v", "current_a", "power_kw", "frequency_hz", "plate_text"]:
        data.setdefault(k, None)

    # Fallback fill from plate_text
    data = regex_fill_from_plate_text(data)

    return raw, data


def load_records() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "AuditType",
                "Facility",
                "Place",
                "PlaceIndex",
                "Model",
                "Serial",
                "Voltage_V",
                "Current_A",
                "Power_kW",
                "Frequency_Hz",
                "Notes",
            ]
        )
    return pd.read_csv(CSV_PATH)


def append_record(row: dict):
    df = load_records()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Audit Nameplate OCR (Gemini)", layout="wide")
st.title("Audit Nameplate Capture (Gemini)")

with st.sidebar:
    st.header("System Status")
    try:
        _ = get_gemini_client()
        st.success("AI Vision: ENABLED ✅")
    except Exception as e:
        st.error(f"AI Vision: DISABLED\n\n{e}")

    st.write(f"Model: `{get_model_name()}`")
    if "GEMINI_API_KEY" in st.secrets:
        st.write(f"Key length: {len(st.secrets['GEMINI_API_KEY'])}")

st.divider()

audit_type = st.selectbox("Audit Type", list(AUDIT_TEMPLATES.keys()))
facility_name = st.text_input("Facility name (required)", placeholder="e.g., First Hotel")

st.subheader("Audit Components (editable list)")
st.caption("One place per line (edit if needed).")
default_places = "\n".join(AUDIT_TEMPLATES[audit_type])
places_text = st.text_area(" ", value=default_places, height=160, label_visibility="collapsed")
places = [p.strip() for p in places_text.splitlines() if p.strip()]

if not places:
    st.warning("Please add at least one place (one per line).")

col1, col2 = st.columns([2, 1])
with col1:
    place = st.selectbox("Place (from your list)", places) if places else None
with col2:
    count = st.number_input("How many instances of this place?", min_value=1, max_value=20, value=1)

custom_area = st.text_input(
    "Custom area name (optional) – e.g., Main Kitchen, West Lobby",
    placeholder="Optional",
)

# Combine as ONE string like you requested (Place + Custom)
place_full = place
if place and custom_area.strip():
    place_full = f"{place} - {custom_area.strip()}"

if facility_name and place_full:
    st.info(f"You will capture nameplates for: **{facility_name} → {place_full} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place_full} #{i}", expanded=(i == 1)):

            camera_photo = st.camera_input(
                f"Take photo for {place_full} #{i}",
                key=f"cam_{audit_type}_{place_full}_{i}",
            )

            uploaded = st.file_uploader(
                f"Or upload existing image for {place_full} #{i}",
                type=["png", "jpg", "jpeg", "webp"],  # keep mobile-safe
                key=f"upl_{audit_type}_{place_full}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                img = safe_open_image(image_file)
                st.image(img, caption="Uploaded image", use_container_width=True)

                run_key = f"run_{audit_type}_{place_full}_{i}"
                if st.button("Analyze image (AI)", key=run_key):
                    with st.spinner("Analyzing image with Gemini..."):
                        raw, out = gemini_extract_fields(img)

                    st.subheader("Extracted fields (edit if needed)")
                    c1, c2, c3, c4, c5, c6 = st.columns(6)

                    with c1:
                        model_val = st.text_input("Model", value=out.get("model") or "", key=f"model_{run_key}")
                    with c2:
                        serial_val = st.text_input("Serial", value=out.get("serial") or "", key=f"serial_{run_key}")
                    with c3:
                        voltage_val = st.text_input("Voltage (V)", value=out.get("voltage_v") or "", key=f"voltage_{run_key}")
                    with c4:
                        current_val = st.text_input("Current (A)", value=out.get("current_a") or "", key=f"current_{run_key}")
                    with c5:
                        power_val = st.text_input("Power (kW)", value=out.get("power_kw") or "", key=f"power_{run_key}")
                    with c6:
                        freq_val = st.text_input("Frequency (Hz)", value=out.get("frequency_hz") or "", key=f"freq_{run_key}")

                    notes = st.text_area("Notes (optional)", value="", key=f"notes_{run_key}")

                    with st.expander("Raw OCR / AI output (debug)"):
                        st.code(raw)

                    if st.button(f"Save record for {place_full} #{i}", key=f"save_{run_key}"):
                        row = {
                            "Timestamp": datetime.now().isoformat(timespec="seconds"),
                            "AuditType": audit_type,
                            "Facility": facility_name,
                            "Place": place_full,
                            "PlaceIndex": i,
                            "Model": model_val or None,
                            "Serial": serial_val or None,
                            "Voltage_V": voltage_val or None,
                            "Current_A": current_val or None,
                            "Power_kW": power_val or None,
                            "Frequency_Hz": freq_val or None,
                            "Notes": notes or None,
                        }
                        append_record(row)
                        st.success(f"Saved ✅  ({CSV_PATH})")

st.divider()
st.subheader("Current CSV preview")

df_all = load_records()
if facility_name:
    df_show = df_all[df_all["Facility"] == facility_name].copy()
else:
    df_show = df_all.copy()

st.caption(f"Saved at: `{CSV_PATH}`")
st.dataframe(df_show, use_container_width=True)

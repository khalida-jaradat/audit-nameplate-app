import os
import re
import io
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# Optional OCR libs (won't break if Tesseract missing)
import cv2
import numpy as np
import pytesseract

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

from openai import OpenAI


# =========================================================
# 0) CONFIG
# =========================================================

AUDIT_TEMPLATES = {
    "Hotels": [
        "Lobby",
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

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "audit_nameplate_records.csv")
THUMBS_DIR = os.path.join(OUTPUT_DIR, "_thumbs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

DEFAULT_TESS_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Choose a vision-capable model
OPENAI_VISION_MODEL = "gpt-4o-mini"


# =========================================================
# 1) HELPERS
# =========================================================

def safe_name(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    return s.replace(" ", "_") if s else "AUDIT"


def ensure_tesseract():
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    if os.name == "nt" and os.path.exists(DEFAULT_TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD


def append_record(row: dict):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, index=False)


def create_thumbnail(img_path: str) -> str | None:
    try:
        if not img_path or not os.path.exists(img_path):
            return None
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        thumb_path = os.path.join(THUMBS_DIR, f"{name}_thumb.png")

        pil = Image.open(img_path).convert("RGB")
        pil.thumbnail((260, 160))
        pil.save(thumb_path, format="PNG")
        return thumb_path
    except Exception:
        return None


def extract_json_object(text: str) -> dict:
    """
    Robust JSON extraction:
    - finds the first {...} block and tries json.loads
    """
    if not text:
        return {}
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object-like substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = text[start:end + 1]
        try:
            return json.loads(sub)
        except Exception:
            return {}

    return {}


# =========================================================
# 2) OCR + AI VISION
# =========================================================

def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return th


def classic_ocr_text(pil_img: Image.Image) -> str:
    try:
        ensure_tesseract()
        proc = preprocess_for_ocr(pil_img)
        return pytesseract.image_to_string(proc, config="--oem 3 --psm 6", lang="eng") or ""
    except Exception:
        return ""


def get_openai_client() -> OpenAI | None:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)  # Streamlit Cloud secrets
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def ai_extract_fields_openai(pil_img: Image.Image) -> tuple[str, dict, str]:
    """
    Returns: (raw_text, fields_dict, error_msg)
    fields_dict keys: Model, Serial, Voltage_V, Current_A, Power_kW, Frequency_Hz
    """
    client = get_openai_client()
    if client is None:
        return "", {
            "Model": "", "Serial": "", "Voltage_V": "",
            "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }, "OPENAI_API_KEY is missing in Streamlit Secrets."

    try:
        # encode image as base64
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are an electrical audit assistant. Read this single equipment nameplate image "
            "and extract ONLY these fields as JSON:\n"
            "{\n"
            '  "model": string|null,\n'
            '  "serial": string|null,\n'
            '  "voltage_v": number|null,\n'
            '  "current_a": number|null,\n'
            '  "power_kw": number|null,\n'
            '  "frequency_hz": number|null\n'
            "}\n\n"
            "Rules:\n"
            "- Return JSON ONLY (no markdown, no explanation).\n"
            "- Keep serial exactly as written.\n"
            "- If voltage is a range like 220-240V, return the mid value (e.g., 230).\n"
            "- If power is in W, convert to kW (e.g., 2400 W -> 2.4).\n"
            "- If a field is not present, use null.\n"
        )

        resp = client.responses.create(
            model=OPENAI_VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                    ],
                }
            ],
            max_output_tokens=180,
        )

        raw_text = getattr(resp, "output_text", "") or ""
        data = extract_json_object(raw_text)

        fields = {
            "Model": (data.get("model") if isinstance(data, dict) else "") or "",
            "Serial": (data.get("serial") if isinstance(data, dict) else "") or "",
            "Voltage_V": "" if not isinstance(data, dict) else ("" if data.get("voltage_v") is None else str(data.get("voltage_v"))),
            "Current_A": "" if not isinstance(data, dict) else ("" if data.get("current_a") is None else str(data.get("current_a"))),
            "Power_kW": "" if not isinstance(data, dict) else ("" if data.get("power_kw") is None else str(data.get("power_kw"))),
            "Frequency_Hz": "" if not isinstance(data, dict) else ("" if data.get("frequency_hz") is None else str(data.get("frequency_hz"))),
        }

        return raw_text, fields, ""

    except Exception as e:
        return "", {
            "Model": "", "Serial": "", "Voltage_V": "",
            "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }, f"{e}"


def analyze_nameplate(pil_img: Image.Image) -> tuple[str, dict, str]:
    """
    AI first. OCR is only for debugging (optional).
    Returns: (raw_combined, fields, error_msg)
    """
    ai_raw, ai_fields, err = ai_extract_fields_openai(pil_img)

    classic = classic_ocr_text(pil_img)  # may be empty on Streamlit Cloud

    raw_combined = ""
    if ai_raw:
        raw_combined += ai_raw
    if classic:
        raw_combined += "\n\n---\n\n" + classic

    return raw_combined, ai_fields, err


# =========================================================
# 3) EXCEL EXPORT
# =========================================================

def build_excel_report(df: pd.DataFrame, out_path: str):
    cols = [
        "Record ID",
        "Place Name",
        "PlaceIndex",
        "Model",
        "Serial",
        "Voltage (V)",
        "Current (A)",
        "Power (kW)",
        "Frequency (Hz)",
        "Capture DateTime",
        "Notes",
        "Nameplate Image",
    ]

    header_fill = PatternFill("solid", fgColor="1F2937")
    header_font = Font(color="FFFFFF", bold=True)
    center = Alignment(vertical="center", wrap_text=True)

    def set_col_widths(ws):
        widths = {
            1: 10, 2: 22, 3: 10, 4: 20, 5: 22,
            6: 12, 7: 12, 8: 12, 9: 14,
            10: 22, 11: 22, 12: 30,
        }
        for col_idx, w in widths.items():
            ws.column_dimensions[get_column_letter(col_idx)].width = w

    def add_headers(ws):
        ws.append(cols)
        for c in range(1, len(cols) + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

    wb = Workbook()
    wb.remove(wb.active)

    if df.empty:
        wb.save(out_path)
        return

    places = sorted(df["Place"].dropna().unique().tolist())
    rec_counter = 1

    for place in places:
        ws = wb.create_sheet(title=str(place)[:31])
        add_headers(ws)
        set_col_widths(ws)
        ws.freeze_panes = "A2"

        df_place = df[df["Place"] == place].copy()

        for _, row in df_place.iterrows():
            record_id = f"{rec_counter:03d}"
            rec_counter += 1

            ws.append(
                [
                    record_id,
                    row.get("Place", ""),
                    row.get("PlaceIndex", ""),
                    row.get("Model", ""),
                    row.get("Serial", ""),
                    row.get("Voltage_V", ""),
                    row.get("Current_A", ""),
                    row.get("Power_kW", ""),
                    row.get("Frequency_Hz", ""),
                    row.get("Timestamp", ""),
                    row.get("Notes", ""),
                    "",
                ]
            )

            r = ws.max_row
            ws.row_dimensions[r].height = 120

            img_path = str(row.get("ImagePath", "")).strip()
            img_path = os.path.normpath(img_path)

            thumb_path = create_thumbnail(img_path)
            if thumb_path and os.path.exists(thumb_path):
                try:
                    xl_img = XLImage(thumb_path)
                    anchor_cell = f"{get_column_letter(len(cols))}{r}"
                    xl_img.anchor = anchor_cell
                    ws.add_image(xl_img)
                except Exception:
                    pass

            for c in range(1, len(cols)):
                ws.cell(row=r, column=c).alignment = center

    wb.save(out_path)


# =========================================================
# 4) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate (AI Vision)", layout="wide")
st.title("Audit Nameplate - AI Vision (OpenAI)")

# ---- Status box
with st.sidebar:
    st.subheader("System Status")
    key_ok = False
    try:
        key_ok = bool(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        key_ok = bool(os.environ.get("OPENAI_API_KEY", ""))

    st.write(f"Model: `{OPENAI_VISION_MODEL}`")
    st.write(f"AI Vision: {'ENABLED ✅' if key_ok else 'DISABLED ❌'}")
    if key_ok:
        k = (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else os.environ.get("OPENAI_API_KEY", ""))
        st.write(f"Key length: {len(str(k))}")

st.write(
    "Select audit type → facility → place → capture/upload nameplate → AI extracts fields → save → export Excel."
)

# ---- Audit type + facility
col1, col2 = st.columns(2)
with col1:
    audit_type = st.selectbox("Audit Type", list(AUDIT_TEMPLATES.keys()))
with col2:
    facility_name = st.text_input("Facility Name", placeholder="e.g., Demo Factory")

default_places = AUDIT_TEMPLATES.get(audit_type, [])

st.divider()
st.subheader("Audit Components (editable list)")
places_text = st.text_area(
    "One place per line",
    value="\n".join(default_places),
    height=180,
)
places = [p.strip() for p in places_text.splitlines() if p.strip()]

if not places:
    st.warning("Please add at least one place (one per line).")

# ---- Place + optional custom area + count
st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place (from your list)", places) if places else None

custom_area = st.text_input(
    "Custom area name (optional) — e.g., Main Kitchen, West Lobby",
    value="",
)

count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1,
)

# Combine place label (this is what will be saved + shown in CSV/Excel)
custom_area_clean = (custom_area or "").strip()
place_label = f"{place} - {custom_area_clean}" if (place and custom_area_clean) else place

if facility_name and place_label:
    st.info(f"You will capture nameplates for: **{facility_name} → {place_label} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place_label:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place_label} #{i}", expanded=(i == 1)):
            # Works great on phone
            camera_photo = st.camera_input(
                f"Take photo for {place_label} #{i}",
                key=f"cam_{audit_type}_{place_label}_{i}",
            )

            uploaded = st.file_uploader(
                f"Or upload existing image for {place_label} #{i}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upl_{audit_type}_{place_label}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                pil_img = Image.open(image_file)
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                with st.spinner("Analyzing image (AI Vision)..."):
                    raw, fields, err = analyze_nameplate(pil_img)

                if err:
                    st.warning(f"AI error: {err}")

                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model = st.text_input("Model", value=fields.get("Model", ""), key=f"model_{place_label}_{i}")
                with c2:
                    serial = st.text_input("Serial", value=fields.get("Serial", ""), key=f"serial_{place_label}_{i}")
                with c3:
                    voltage = st.text_input("Voltage (V)", value=fields.get("Voltage_V", ""), key=f"volt_{place_label}_{i}")
                with c4:
                    current = st.text_input("Current (A)", value=fields.get("Current_A", ""), key=f"curr_{place_label}_{i}")
                with c5:
                    power = st.text_input("Power (kW)", value=fields.get("Power_kW", ""), key=f"pwr_{place_label}_{i}")
                with c6:
                    freq = st.text_input("Frequency (Hz)", value=fields.get("Frequency_Hz", ""), key=f"hz_{place_label}_{i}")

                notes = st.text_input("Notes (optional)", value="", key=f"notes_{place_label}_{i}")

                with st.expander("Raw OCR / AI output (debug)"):
                    st.code(raw[:3000] if raw else "")

                if st.button(f"Save record for {place_label} #{i}", key=f"save_{place_label}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place_label)

                    dir_path = os.path.join(OUTPUT_DIR, safe_fac, f"{safe_place}_{i}")
                    os.makedirs(dir_path, exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(dir_path, f"nameplate_{ts}.png")
                    pil_img.convert("RGB").save(img_path)
                    img_path = os.path.abspath(img_path)

                    row = {
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                        "AuditType": audit_type,
                        "Facility": facility_name,
                        "Place": place_label,          # ✅ place + custom area as one field
                        "PlaceIndex": i,
                        "Model": model,
                        "Serial": serial,
                        "Voltage_V": voltage,
                        "Current_A": current,
                        "Power_kW": power,
                        "Frequency_Hz": freq,
                        "Notes": notes,
                        "ImagePath": img_path,
                        "RawOCR": raw,
                    }
                    append_record(row)

                    json_path = os.path.join(dir_path, f"record_{ts}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)

                    st.success("Saved! Record added to CSV + image saved.")


# =========================================================
# 5) CURRENT CSV PREVIEW
# =========================================================

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)

    # ✅ show only current facility if typed
    if facility_name:
        df_show = df_all[df_all["Facility"] == facility_name].copy()
    else:
        df_show = df_all.copy()

    st.dataframe(df_show.tail(50), use_container_width=True)
    st.caption(f"Saved at: {CSV_PATH}")
else:
    st.write("No records yet.")


# =========================================================
# 6) EXPORT EXCEL
# =========================================================

st.divider()
st.subheader("Export Excel (embedded images by Place)")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)

    if facility_name:
        df_use = df_all[df_all["Facility"] == facility_name].copy()
    else:
        df_use = df_all.copy()

    if not df_use.empty:
        safe_fac = safe_name(facility_name or "AUDIT")
        xlsx_path = os.path.join(
            OUTPUT_DIR,
            f"AUDIT_{safe_fac}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        )

        if st.button("Generate Excel with embedded images"):
            for col in ["Notes", "RawOCR", "Current_A"]:
                if col not in df_use.columns:
                    df_use[col] = ""

            build_excel_report(df_use, xlsx_path)
            st.success("Excel generated successfully!")

            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "Download Excel Report",
                    data=f,
                    file_name=os.path.basename(xlsx_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.info("No records for this facility yet. Save at least one nameplate record.")
else:
    st.info("No CSV records yet. Save at least one nameplate record first.")

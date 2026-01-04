import os
import re
import io
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

import requests

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

# =========================
# 0) CONFIG
# =========================

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


# =========================
# 1) HELPERS
# =========================

def safe_name(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    return s.replace(" ", "_") if s else "AUDIT"


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

        pil = Image.open(img_path)
        pil = pil.convert("RGB")
        pil.thumbnail((260, 160))
        pil.save(thumb_path, format="PNG")
        return thumb_path
    except Exception:
        return None


def image_to_jpeg_b64(pil_img: Image.Image) -> str:
    # Fix rotation from phone EXIF
    pil_img = ImageOps.exif_transpose(pil_img)

    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json_from_text(raw: str) -> dict | None:
    """
    Gemini ممكن يرجع:
    - JSON صافي
    - JSON داخل ```json ... ```
    - نص وفيه JSON
    هون بنستخرج أول JSON object مضبوط.
    """
    if not raw:
        return None

    s = raw.strip()

    # remove code fences
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # direct try
    try:
        return json.loads(s)
    except Exception:
        pass

    # find first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0).strip()

    # try again
    try:
        return json.loads(candidate)
    except Exception:
        return None


def normalize_fields(data: dict) -> dict:
    """
    تحويل القيم لنصوص جاهزة للواجهة
    """
    def as_str(x):
        if x is None:
            return ""
        return str(x).strip()

    fields = {
        "Model": as_str(data.get("model")),
        "Serial": as_str(data.get("serial")),
        "Voltage_V": as_str(data.get("voltage_v")),
        "Current_A": as_str(data.get("current_a")),
        "Power_kW": as_str(data.get("power_kw")),
        "Frequency_Hz": as_str(data.get("frequency_hz")),
    }
    return fields


# =========================
# 2) GEMINI VISION (Google AI Studio)
# =========================

def get_gemini_key_and_model() -> tuple[str | None, str]:
    api_key = None
    model = "gemini-2.5-flash"

    # Streamlit secrets first
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        model = st.secrets.get("GEMINI_MODEL", model)
    except Exception:
        api_key = None

    # env fallback
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", model)

    return api_key, model


def gemini_extract_fields(pil_img: Image.Image) -> tuple[str, dict]:
    api_key, model = get_gemini_key_and_model()
    if not api_key:
        return "Gemini API key missing. Add GEMINI_API_KEY in Streamlit Secrets.", {
            "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }

    b64 = image_to_jpeg_b64(pil_img)

    prompt = (
        "You are reading a single electrical equipment nameplate.\n"
        "Extract ONLY these fields and return STRICT JSON only (no markdown, no code fences):\n"
        "{\n"
        '  "model": string or null,\n'
        '  "serial": string or null,\n'
        '  "voltage_v": number or null,   (if range like 220-240V return midpoint 230)\n'
        '  "current_a": number or null,\n'
        '  "power_kw": number or null,    (if power is in W convert to kW)\n'
        '  "frequency_hz": number or null\n'
        "}\n"
        "If a field is not present, set it to null.\n"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 220,
            "responseMimeType": "application/json",
            "stopSequences": ["```"],
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            return f"Gemini HTTP {r.status_code}: {r.text}", {
                "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
            }

        resp = r.json()

        # text extraction
        raw_text = ""
        candidates = resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            raw_text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)])

        data = extract_json_from_text(raw_text)
        if not data:
            # fallback: sometimes responseMimeType ignored; try without it once
            payload2 = payload.copy()
            payload2["generationConfig"] = {
                "temperature": 0,
                "maxOutputTokens": 220,
                "stopSequences": ["```"],
            }
            r2 = requests.post(url, json=payload2, timeout=60)
            if r2.status_code != 200:
                return f"Gemini returned no JSON. Raw: {raw_text}\n\nRetryHTTP {r2.status_code}: {r2.text}", {
                    "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
                }

            resp2 = r2.json()
            raw_text2 = ""
            candidates2 = resp2.get("candidates", [])
            if candidates2:
                parts2 = candidates2[0].get("content", {}).get("parts", [])
                raw_text2 = "\n".join([p.get("text", "") for p in parts2 if isinstance(p, dict)])

            data2 = extract_json_from_text(raw_text2)
            if not data2:
                return f"Gemini returned no JSON. Raw:\n{raw_text2}", {
                    "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
                }

            return raw_text2, normalize_fields(data2)

        return raw_text, normalize_fields(data)

    except Exception as e:
        return f"Gemini exception: {e}", {
            "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }


def analyze_nameplate(pil_img: Image.Image) -> tuple[str, dict]:
    # AI only (Gemini). No OCR dependency for phone.
    raw, fields = gemini_extract_fields(pil_img)
    return raw, fields


# =========================
# 3) EXCEL EXPORT
# =========================

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
            1: 10, 2: 22, 3: 10, 4: 20, 5: 22, 6: 12, 7: 12,
            8: 12, 9: 14, 10: 22, 11: 22, 12: 30,
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

    # group by BASE Place (Place column)
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

            place_label = row.get("PlaceLabel", place)  # show combined label in Excel
            place_index = row.get("PlaceIndex", "")
            model = row.get("Model", "")
            serial = row.get("Serial", "")
            voltage = row.get("Voltage_V", "")
            current = row.get("Current_A", "")
            power = row.get("Power_kW", "")
            freq = row.get("Frequency_Hz", "")
            ts = row.get("Timestamp", "")
            notes = row.get("Notes", "")
            img_path = row.get("ImagePath", "")

            ws.append(
                [
                    record_id,
                    place_label,
                    place_index,
                    model,
                    serial,
                    voltage,
                    current,
                    power,
                    freq,
                    ts,
                    notes,
                    "",
                ]
            )

            r = ws.max_row
            ws.row_dimensions[r].height = 120

            img_path = str(img_path).strip()
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


# =========================
# 4) STREAMLIT UI
# =========================

st.set_page_config(page_title="Audit Nameplate OCR - Demo (Gemini)", layout="wide")
st.title("Audit Nameplate – AI Extraction (Gemini)")

api_key, model_name = get_gemini_key_and_model()

with st.sidebar:
    st.subheader("System Status")
    st.write(f"Model: `{model_name}`")
    if api_key:
        st.success("AI Vision: ENABLED ✅")
        st.caption(f"Key length: {len(api_key)}")
    else:
        st.error("AI Vision: DISABLED ❌")
        st.caption("Add GEMINI_API_KEY in Streamlit Secrets.")

# اختيار نوع الأوديت والمنشأة
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
    st.warning("Please add at least one place to continue.")

st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place", places) if places else None

# Custom area name (optional) — (you asked to keep old flow, but store next to Place in Excel)
custom_area = st.text_input(
    "Custom area name (optional) – e.g., Main Kitchen, West Lobby",
    value="",
)

count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1,
)

place_label = place
if place and custom_area.strip():
    place_label = f"{place} - {custom_area.strip()}"

if facility_name and place:
    st.info(f"You will capture nameplates for: **{facility_name} → {place_label} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place_label} #{i}", expanded=(i == 1)):
            # camera for mobile + uploader fallback
            camera_photo = st.camera_input(
                f"Take photo for {place_label} #{i}",
                key=f"cam_{audit_type}_{place}_{i}",
            )
            uploaded = st.file_uploader(
                f"Or upload existing image for {place_label} #{i}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upl_{audit_type}_{place}_{i}",
            )

            image_file = camera_photo if camera_photo is not None else uploaded

            if image_file:
                try:
                    pil_img = Image.open(image_file)
                    pil_img = ImageOps.exif_transpose(pil_img)
                except Exception as e:
                    st.error(f"Could not open image. Try JPG/PNG. Error: {e}")
                    continue

                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                with st.spinner("Analyzing image with Gemini..."):
                    raw, fields = analyze_nameplate(pil_img)

                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model_v = st.text_input("Model", value=fields.get("Model") or "", key=f"model_{place}_{i}")
                with c2:
                    serial_v = st.text_input("Serial", value=fields.get("Serial") or "", key=f"serial_{place}_{i}")
                with c3:
                    voltage_v = st.text_input("Voltage (V)", value=fields.get("Voltage_V") or "", key=f"volt_{place}_{i}")
                with c4:
                    current_v = st.text_input("Current (A)", value=fields.get("Current_A") or "", key=f"curr_{place}_{i}")
                with c5:
                    power_v = st.text_input("Power (kW)", value=fields.get("Power_kW") or "", key=f"pwr_{place}_{i}")
                with c6:
                    freq_v = st.text_input("Frequency (Hz)", value=fields.get("Frequency_Hz") or "", key=f"hz_{place}_{i}")

                notes = st.text_input("Notes (optional)", value="", key=f"notes_{place}_{i}")

                with st.expander("Raw OCR / AI output (debug)"):
                    st.code(raw[:6000] if raw else "")

                if st.button(f"Save record for {place_label} #{i}", key=f"save_{place}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)

                    dir_path = os.path.join(OUTPUT_DIR, safe_fac, f"{safe_place}_{i}")
                    os.makedirs(dir_path, exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(dir_path, f"nameplate_{ts}.png")
                    pil_img.save(img_path)
                    img_path = os.path.abspath(img_path)

                    row = {
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                        "AuditType": audit_type,
                        "Facility": facility_name,
                        "Place": place,                 # base place for grouping
                        "PlaceLabel": place_label,      # combined label shown in Excel
                        "PlaceIndex": i,
                        "Model": model_v,
                        "Serial": serial_v,
                        "Voltage_V": voltage_v,
                        "Current_A": current_v,
                        "Power_kW": power_v,
                        "Frequency_Hz": freq_v,
                        "Notes": notes,
                        "ImagePath": img_path,
                        "RawOCR": raw,
                    }
                    append_record(row)

                    json_path = os.path.join(dir_path, f"record_{ts}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)

                    st.success("Saved! Record added to CSV with local image.")

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)
    # show only current facility if provided
    if facility_name:
        df_show = df_all[df_all["Facility"] == facility_name].copy()
    else:
        df_show = df_all.copy()

    st.dataframe(df_show.tail(50), use_container_width=True)
    st.caption(f"Saved at: {CSV_PATH}")
else:
    st.write("No records yet.")

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
            # ensure columns exist
            for col in ["Notes", "RawOCR", "Current_A", "PlaceLabel"]:
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

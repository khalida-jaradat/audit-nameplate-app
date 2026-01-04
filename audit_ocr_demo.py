import os
import re
import io
import json
import base64
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

import requests

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

# =========================================================
# 0) CONFIG (آخر أسماء ثابتة حسب تعديلك)
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

# =========================================================
# 1) HELPERS
# =========================================================

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

        pil = Image.open(img_path).convert("RGB")
        pil.thumbnail((260, 160))
        pil.save(thumb_path, format="PNG")
        return thumb_path
    except Exception:
        return None


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
            1: 10,  2: 22,  3: 10,  4: 22,  5: 22,
            6: 12,  7: 12,  8: 12,  9: 14,  10: 22,
            11: 22, 12: 30,
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
# 2) GEMINI VISION (Google AI Studio Key)
# =========================================================

def get_gemini_settings():
    api_key = None
    model = "gemini-2.5-flash"

    # Streamlit secrets أولاً
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        model = st.secrets.get("GEMINI_MODEL", model)
    except Exception:
        pass

    # Env fallback
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    return api_key, model


def _strip_to_json(text: str) -> dict | None:
    if not text:
        return None

    t = text.strip()

    # شيل code fences
    t = re.sub(r"^```(?:json)?", "", t.strip(), flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t.strip()).strip()

    # حاول parse مباشر
    try:
        return json.loads(t)
    except Exception:
        pass

    # حاول قص من أول { لآخر }
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = t[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None

    return None


def gemini_extract_fields(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[str, dict]:
    api_key, model = get_gemini_settings()

    if not api_key:
        return "GEMINI_API_KEY is missing in Streamlit Secrets.", {
            "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are extracting electrical nameplate fields from ONE image.\n"
        "Return ONLY a valid JSON object with EXACT keys:\n"
        "model, serial, voltage_v, current_a, power_kw, frequency_hz\n\n"
        "Rules:\n"
        "- If a value is missing, return empty string.\n"
        "- voltage_v/current_a/power_kw/frequency_hz must be numbers as text (e.g. \"230\", \"2.0\", \"0.4\", \"50\").\n"
        "- If voltage is a range like 220-240V, return \"230\".\n"
        "- If power is in W, convert to kW (e.g. 400W -> \"0.4\").\n"
        "Return JSON only. No markdown, no explanations."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 220,
        },
    }

    # محاولة أولى
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return f"Gemini HTTP {r.status_code}: {r.text}", {
                "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
            }

        data = r.json()
        raw_text = ""
        try:
            raw_text = data["candidates"][0]["content"]["parts"][0].get("text", "")
        except Exception:
            raw_text = str(data)

        obj = _strip_to_json(raw_text)

        # لو ما نجح، نعمل retry prompt أقسى
        if obj is None:
            payload["contents"][0]["parts"][0]["text"] = (
                "Return STRICT JSON ONLY. No code fences. No extra text.\n"
                "Keys: model, serial, voltage_v, current_a, power_kw, frequency_hz\n"
                "All values must be strings. Missing -> \"\".\n"
            )
            r2 = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if r2.status_code != 200:
                return f"Gemini HTTP {r2.status_code}: {r2.text}", {
                    "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
                }
            data2 = r2.json()
            raw_text2 = ""
            try:
                raw_text2 = data2["candidates"][0]["content"]["parts"][0].get("text", "")
            except Exception:
                raw_text2 = str(data2)

            obj = _strip_to_json(raw_text2)
            if obj is None:
                return f"Gemini returned no JSON. Raw: {raw_text2[:800]}", {
                    "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
                }
            raw_text = raw_text2  # للـ debug

        fields = {
            "Model": str(obj.get("model", "") or ""),
            "Serial": str(obj.get("serial", "") or ""),
            "Voltage_V": str(obj.get("voltage_v", "") or ""),
            "Current_A": str(obj.get("current_a", "") or ""),
            "Power_kW": str(obj.get("power_kw", "") or ""),
            "Frequency_Hz": str(obj.get("frequency_hz", "") or ""),
        }
        return raw_text, fields

    except Exception as e:
        return f"Gemini vision error: {e}", {
            "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }


def detect_mime(uploaded_file) -> str:
    # Streamlit uploader sometimes provides type
    try:
        if getattr(uploaded_file, "type", None):
            return uploaded_file.type
    except Exception:
        pass
    return "image/jpeg"


# =========================================================
# 3) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate - Gemini", layout="wide")
st.title("Audit Nameplate Capture (Gemini Vision)")

api_key, model_name = get_gemini_settings()

with st.sidebar:
    st.subheader("System Status")
    st.write(f"Model: `{model_name}`")
    st.write(f"AI Vision: {'ENABLED ✅' if api_key else 'DISABLED ❌'}")
    st.write(f"Key length: {len(api_key) if api_key else 0}")

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
    st.warning("Please add at least one place (one per line).")

st.divider()
st.subheader("Choose a place")

place = st.selectbox("Place", places) if places else None

# (رجّعنا القديم) عدد التكرارات موجود
count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1,
)

# Custom area name (اختياري) — وبنخليه ينضاف جنب Place عند الحفظ/الإكسل
custom_area = st.text_input(
    "Custom area name (optional) – e.g., Main Kitchen, West Lobby",
    value="",
)

place_label = place
if place and custom_area.strip():
    place_label = f"{place} - {custom_area.strip()}"

if facility_name and place_label:
    st.info(f"You will capture nameplates for: **{facility_name} → {place_label} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place_label} #{i}", expanded=(i == 1)):

            # خيار الكاميرا (يفيد للموبايل)
            camera_photo = st.camera_input(
                f"Take photo for {place_label} #{i}",
                key=f"cam_{audit_type}_{place}_{i}",
            )

            # أو رفع صورة
            uploaded = st.file_uploader(
                f"Or upload existing image for {place_label} #{i}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upl_{audit_type}_{place}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                img_bytes = image_file.getvalue()
                mime = detect_mime(image_file)

                # افتحها بـ PIL (لو فشل لأي سبب نعرض رسالة)
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    st.error(f"Could not open image on server. Try JPG/PNG. Error: {e}")
                    st.stop()

                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                # مهم جدًا: لا نستدعي الـ AI كل rerun
                img_hash = hashlib.sha256(img_bytes).hexdigest()[:12]
                state_key = f"analysis_{audit_type}_{place_label}_{i}_{img_hash}"

                if state_key not in st.session_state:
                    if st.button("Analyze image (AI)", key=f"an_{audit_type}_{place_label}_{i}_{img_hash}"):
                        with st.spinner("Analyzing image with Gemini..."):
                            raw, fields = gemini_extract_fields(img_bytes, mime_type=mime)
                        st.session_state[state_key] = {"raw": raw, "fields": fields}
                else:
                    raw = st.session_state[state_key]["raw"]
                    fields = st.session_state[state_key]["fields"]

                # لو لسه ما حللنا، ما نعرض حقول فاضية بس
                if state_key not in st.session_state:
                    st.info("Click **Analyze image (AI)** to extract fields.")
                    continue

                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)

                with c1:
                    model_val = st.text_input("Model", value=fields.get("Model", ""), key=f"model_{place_label}_{i}")
                with c2:
                    serial_val = st.text_input("Serial", value=fields.get("Serial", ""), key=f"serial_{place_label}_{i}")
                with c3:
                    voltage_val = st.text_input("Voltage (V)", value=fields.get("Voltage_V", ""), key=f"volt_{place_label}_{i}")
                with c4:
                    current_val = st.text_input("Current (A)", value=fields.get("Current_A", ""), key=f"curr_{place_label}_{i}")
                with c5:
                    power_val = st.text_input("Power (kW)", value=fields.get("Power_kW", ""), key=f"pwr_{place_label}_{i}")
                with c6:
                    freq_val = st.text_input("Frequency (Hz)", value=fields.get("Frequency_Hz", ""), key=f"hz_{place_label}_{i}")

                notes_val = st.text_input("Notes (optional)", value="", key=f"notes_{place_label}_{i}")

                with st.expander("Raw OCR / AI output (debug)"):
                    st.code(raw[:3000] if raw else "")

                if st.button(f"Save record for {place_label} #{i}", key=f"save_{place_label}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place_label)

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
                        "Place": place_label,  # <-- هون صارت جنبها custom area
                        "PlaceIndex": i,
                        "Model": model_val,
                        "Serial": serial_val,
                        "Voltage_V": voltage_val,
                        "Current_A": current_val,
                        "Power_kW": power_val,
                        "Frequency_Hz": freq_val,
                        "Notes": notes_val,
                        "ImagePath": img_path,
                        "RawOCR": raw,
                    }
                    append_record(row)

                    json_path = os.path.join(dir_path, f"record_{ts}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)

                    st.success("Saved! Record added to CSV with local image.")

# =========================================================
# 4) CURRENT CSV PREVIEW (فلترة حسب Facility الحالية)
# =========================================================

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)

    if facility_name:
        df_show = df_all[df_all["Facility"] == facility_name].copy()
    else:
        df_show = df_all.copy()

    st.dataframe(df_show.tail(50), use_container_width=True)
    st.caption(f"Saved at: {CSV_PATH}")
else:
    st.write("No records yet.")

# =========================================================
# 5) EXPORT EXCEL
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

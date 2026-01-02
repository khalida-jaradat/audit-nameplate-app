import os
import re
import io
import json
import base64
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

import cv2
import numpy as np

# pytesseract اختياري (لو موجود + Tesseract متثبت)
try:
    import pytesseract
except Exception:
    pytesseract = None

import requests

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill


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


# =========================================================
# 1) HELPERS
# =========================================================

def safe_name(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    return s.replace(" ", "_") if s else "AUDIT"


def get_gemini_key_and_model():
    key = None
    model = None

    # 1) env
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    model = os.environ.get("GEMINI_MODEL")

    # 2) streamlit secrets
    try:
        if not key:
            key = st.secrets.get("GEMINI_API_KEY", None)
        if not model:
            model = st.secrets.get("GEMINI_MODEL", None)
    except Exception:
        pass

    if not model:
        model = "gemini-2.5-flash"

    return key, model


def append_record(row: dict):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(CSV_PATH, index=False)


def create_thumbnail(img_path: str):
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


def try_enable_heic():
    """
    لو بدك HEIC/HEIF يشتغل: pip install pillow-heif
    """
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


HEIC_ENABLED = try_enable_heic()


def load_pil_image(file_obj) -> Image.Image:
    """
    تحميل صورة من camera_input أو file_uploader بشكل robust
    """
    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
    else:
        data = file_obj.read()

    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img


def resize_for_ai(pil_img: Image.Image, max_side: int = 1600) -> Image.Image:
    """
    تصغير الصورة لتقليل الاستهلاك وتحسين السرعة، بدون ما نخرب القراءة
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)


# =========================================================
# 2) OCR (اختياري) + Gemini Vision
# =========================================================

def ensure_tesseract():
    if pytesseract is None:
        return
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    if os.name == "nt" and os.path.exists(DEFAULT_TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD


def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return th


def classic_ocr_text(pil_img: Image.Image) -> str:
    if pytesseract is None:
        return ""
    try:
        ensure_tesseract()
        proc = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(proc, config="--oem 3 --psm 6", lang="eng")
        return text or ""
    except Exception:
        return ""


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def normalize_fields(data: dict) -> dict:
    """
    تنظيف وتحويل القيم للعرض
    """
    model = (data.get("model") or data.get("Model") or "").strip()
    serial = (data.get("serial") or data.get("Serial") or "").strip()

    voltage = data.get("voltage_v") if "voltage_v" in data else data.get("Voltage_V")
    current = data.get("current_a") if "current_a" in data else data.get("Current_A")
    power = data.get("power_kw") if "power_kw" in data else data.get("Power_kW")
    freq = data.get("frequency_hz") if "frequency_hz" in data else data.get("Frequency_Hz")

    v = parse_number(voltage)
    a = parse_number(current)
    kw = parse_number(power)
    hz = parse_number(freq)

    # لو الجهد جاينا كنص "220-240" نخليها 230
    if isinstance(voltage, str) and "-" in voltage:
        parts = re.findall(r"\d+(\.\d+)?", voltage)
        if len(parts) >= 2:
            try:
                v1 = float(parts[0]); v2 = float(parts[1])
                v = (v1 + v2) / 2.0
            except Exception:
                pass

    # لو الاستطاعة جاية بالواط داخل نص
    if kw is None and isinstance(power, str):
        m = re.search(r"(\d+(\.\d+)?)\s*W", power, re.IGNORECASE)
        if m:
            try:
                kw = float(m.group(1)) / 1000.0
            except Exception:
                pass

    return {
        "Model": model,
        "Serial": serial,
        "Voltage_V": "" if v is None else str(int(v)) if abs(v - int(v)) < 1e-6 else str(round(v, 2)),
        "Current_A": "" if a is None else str(round(a, 3)),
        "Power_kW": "" if kw is None else str(round(kw, 3)),
        "Frequency_Hz": "" if hz is None else str(int(hz)) if abs(hz - int(hz)) < 1e-6 else str(round(hz, 2)),
    }


def gemini_extract_fields(pil_img: Image.Image, max_output_tokens: int = 180) -> tuple[str, dict, str]:
    """
    Gemini Vision عبر REST
    returns: (raw_text, fields, err)
    """
    api_key, model = get_gemini_key_and_model()
    if not api_key:
        return "", {
            "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
        }, "No GEMINI_API_KEY found in secrets/env."

    img = resize_for_ai(pil_img, max_side=1600)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are an electrical audit expert. Read ONE equipment nameplate image and extract ONLY these fields.\n"
        "Return STRICT JSON only (no markdown, no extra text):\n"
        "{\n"
        '  "model": string | null,\n'
        '  "serial": string | null,\n'
        '  "voltage_v": number | null,\n'
        '  "current_a": number | null,\n'
        '  "power_kw": number | null,\n'
        '  "frequency_hz": number | null\n'
        "}\n"
        "Rules:\n"
        "- If voltage is a range like 220-240V, return 230.\n"
        "- If power is in W, convert to kW.\n"
        "- If a field is missing, return null.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "model": {"type": ["string", "null"]},
            "serial": {"type": ["string", "null"]},
            "voltage_v": {"type": ["number", "null"]},
            "current_a": {"type": ["number", "null"]},
            "power_kw": {"type": ["number", "null"]},
            "frequency_hz": {"type": ["number", "null"]},
        },
        "required": ["model", "serial", "voltage_v", "current_a", "power_kw", "frequency_hz"],
    }

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64
                    },
                    "mediaResolution": {"level": "media_resolution_high"},
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }

    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    def call(version: str):
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent"
        return requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    # retry بسيط ل 429
    for attempt in range(3):
        resp = call("v1beta")
        if resp.status_code == 404:
            resp = call("v1alpha")

        if resp.status_code == 429:
            try:
                time.sleep(2 + attempt)
            except Exception:
                pass
            continue

        if resp.status_code != 200:
            return "", {
                "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
            }, f"Gemini HTTP {resp.status_code}: {resp.text[:800]}"

        try:
            rj = resp.json()
            parts = rj["candidates"][0]["content"]["parts"]
            raw_text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)])

            # raw_text المفروض JSON
            try:
                data = json.loads(raw_text)
            except Exception:
                # حاول نقتص JSON من داخل النص
                s = raw_text
                i1 = s.find("{"); i2 = s.rfind("}")
                if i1 != -1 and i2 != -1 and i2 > i1:
                    data = json.loads(s[i1:i2+1])
                else:
                    data = {}

            fields = normalize_fields(data if isinstance(data, dict) else {})
            return raw_text, fields, ""
        except Exception as e:
            return "", {
                "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
            }, f"Parse error: {e}"

    return "", {
        "Model": "", "Serial": "", "Voltage_V": "", "Current_A": "", "Power_kW": "", "Frequency_Hz": ""
    }, "Rate limited (429). Try again."


def analyze_nameplate(pil_img: Image.Image, max_output_tokens: int = 180) -> tuple[str, dict, str]:
    """
    Unified:
    - Gemini Vision أولاً
    - OCR اختياري كـ fallback
    """
    ai_raw, ai_fields, ai_err = gemini_extract_fields(pil_img, max_output_tokens=max_output_tokens)

    classic = ""
    if not any(ai_fields.values()):
        classic = classic_ocr_text(pil_img)

    raw_combined = ""
    if classic:
        raw_combined += classic + "\n\n---\n\n"
    if ai_raw:
        raw_combined += ai_raw

    return raw_combined, ai_fields, ai_err


# =========================================================
# 3) EXCEL EXPORT (مع الصور)
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
        widths = {1: 10, 2: 22, 3: 10, 4: 22, 5: 22, 6: 12, 7: 12, 8: 12, 9: 14, 10: 22, 11: 22, 12: 30}
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

            ws.append([
                record_id,
                place,
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
            ])

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

st.set_page_config(page_title="Audit Nameplate OCR - Demo", layout="wide")
st.title("Audit Nameplate OCR - Quick Demo")

api_key, model_name = get_gemini_key_and_model()

with st.sidebar:
    st.header("System Status")
    st.write(f"Model: `{model_name}`")
    st.success("AI Vision: ENABLED ✅" if api_key else "AI Vision: DISABLED ❌")
    st.write(f"Key length: {len(api_key) if api_key else 0}")
    st.caption(f"HEIC enabled: {HEIC_ENABLED}")

    max_tokens = st.slider("AI max output tokens", 80, 260, 180, 10)
    st.caption("Lower tokens = أقل تكلفة/استهلاك + أقل أخطاء quotas.")


st.write(
    "Select audit type → facility → area → capture nameplate → AI extracts values → save → export Excel."
)

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

# Custom area name (اختياري) — وبنخليه ينحفظ جنب الـ Place ككلمة وحدة
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

place_full = None
if place:
    ca = custom_area.strip()
    place_full = f"{place} - {ca}" if ca else place

if facility_name and place_full:
    st.info(f"You will capture nameplates for: **{facility_name} → {place_full} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place_full:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place_full} #{i}", expanded=(i == 1)):

            camera_photo = st.camera_input(
                f"Take photo for {place_full} #{i}",
                key=f"cam_{audit_type}_{place_full}_{i}",
            )

            uploaded = st.file_uploader(
                f"Or upload existing image for {place_full} #{i}",
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                key=f"upl_{audit_type}_{place_full}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                try:
                    pil_img = load_pil_image(image_file)
                except Exception as e:
                    st.error(f"Cannot open this image on server. Try JPG/PNG. Error: {e}")
                    st.stop()

                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                with st.spinner("Analyzing image (AI)..."):
                    raw, fields, ai_err = analyze_nameplate(pil_img, max_output_tokens=int(max_tokens))

                if ai_err and not any(fields.values()):
                    st.warning(ai_err)

                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model_val = st.text_input("Model", value=fields.get("Model") or "", key=f"model_{place_full}_{i}")
                with c2:
                    serial_val = st.text_input("Serial", value=fields.get("Serial") or "", key=f"serial_{place_full}_{i}")
                with c3:
                    voltage_val = st.text_input("Voltage (V)", value=fields.get("Voltage_V") or "", key=f"volt_{place_full}_{i}")
                with c4:
                    current_val = st.text_input("Current (A)", value=fields.get("Current_A") or "", key=f"curr_{place_full}_{i}")
                with c5:
                    power_val = st.text_input("Power (kW)", value=fields.get("Power_kW") or "", key=f"pwr_{place_full}_{i}")
                with c6:
                    freq_val = st.text_input("Frequency (Hz)", value=fields.get("Frequency_Hz") or "", key=f"hz_{place_full}_{i}")

                notes = st.text_input("Notes (optional)", value="", key=f"notes_{place_full}_{i}")

                with st.expander("Raw OCR / AI JSON"):
                    st.code(raw[:3000] if raw else "")

                if st.button(f"Save record for {place_full} #{i}", key=f"save_{place_full}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place_full)

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
                        "Place": place_full,     # ✅ هون الدمج
                        "PlaceIndex": i,
                        "Model": model_val,
                        "Serial": serial_val,
                        "Voltage_V": voltage_val,
                        "Current_A": current_val,
                        "Power_kW": power_val,
                        "Frequency_Hz": freq_val,
                        "Notes": notes,
                        "ImagePath": img_path,
                        "RawOCR": raw,
                    }

                    append_record(row)

                    json_path = os.path.join(dir_path, f"record_{ts}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)

                    st.success("Saved! Record added to CSV with local image.")


# =========================================================
# 5) CURRENT CSV PREVIEW
# =========================================================

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)

    # ✅ عرض فقط سجلات المنشأة الحالية لو مكتوب اسمها
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
        xlsx_path = os.path.join(OUTPUT_DIR, f"AUDIT_{safe_fac}_{datetime.now().strftime('%Y%m%d')}.xlsx")

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

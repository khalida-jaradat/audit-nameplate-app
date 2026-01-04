import os
import re
import io
import json
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

import cv2
import numpy as np
import pytesseract

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

# Gemini (Google)
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# =========================================================
# 0) CONFIG
# =========================================================

AUDIT_TEMPLATES = {
    "Hotelsس": [
        "Lobbyسس",
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


def normalize_pil(pil_img: Image.Image) -> Image.Image:
    """حل مشاكل دوران صور الهاتف + توحيد اللون + تصغير حجم الإرسال للـ API"""
    pil_img = ImageOps.exif_transpose(pil_img)  # مهم جدًا للموبايل
    pil_img = pil_img.convert("RGB")

    # تصغير (يحسن السرعة ويقلل مشاكل)
    max_side = 1600
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))
    return pil_img


def pil_to_jpeg_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# =========================================================
# 2) CLASSIC OCR (اختياري)
# =========================================================

def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return th


def classic_ocr_text(pil_img: Image.Image) -> str:
    try:
        ensure_tesseract()
        proc = preprocess_for_ocr(pil_img)
        return pytesseract.image_to_string(proc, config="--oem 3 --psm 6", lang="eng") or ""
    except Exception:
        return ""


# =========================================================
# 3) GEMINI VISION (الأساسي)
# =========================================================

def get_gemini_api_key() -> str | None:
    # 1) env
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    # 2) streamlit secrets
    try:
        k = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if k:
            return k
    except Exception:
        pass
    return None


def extract_json_from_text(raw: str) -> str | None:
    if not raw:
        return None

    s = raw.strip()

    # لو جاي ضمن ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    # لو في كلام قبل/بعد JSON — نستخرج أول كائن JSON متوازن
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    return s[start:end + 1]


def gemini_extract_fields(pil_img: Image.Image) -> tuple[str, dict]:
    empty = {
        "Model": "",
        "Serial": "",
        "Voltage_V": "",
        "Current_A": "",
        "Power_kW": "",
        "Frequency_Hz": "",
    }

    if genai is None or types is None:
        return "Gemini SDK not installed (google-genai).", empty

    api_key = get_gemini_api_key()
    if not api_key:
        return "No GEMINI_API_KEY found in secrets/env.", empty

    client = genai.Client(api_key=api_key)

    schema = types.Schema(
        type=types.Type.OBJECT,
        required=["model", "serial", "voltage_v", "current_a", "power_kw", "frequency_hz"],
        properties={
            "model": types.Schema(type=types.Type.STRING, nullable=True),
            "serial": types.Schema(type=types.Type.STRING, nullable=True),
            "voltage_v": types.Schema(type=types.Type.NUMBER, nullable=True),
            "current_a": types.Schema(type=types.Type.NUMBER, nullable=True),
            "power_kw": types.Schema(type=types.Type.NUMBER, nullable=True),
            "frequency_hz": types.Schema(type=types.Type.NUMBER, nullable=True),
        },
    )

    prompt = (
        "You are an electrical audit expert. Extract ONLY these fields from the nameplate image.\n"
        "Return JSON only (no markdown, no explanation).\n"
        "- model (string)\n"
        "- serial (string)\n"
        "- voltage_v (number)\n"
        "- current_a (number)\n"
        "- power_kw (number)  (convert W to kW)\n"
        "- frequency_hz (number)\n"
        "If a field is missing, return null.\n"
    )

    img_bytes = pil_to_jpeg_bytes(pil_img)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    try:
        # محاولة 1
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        raw = (resp.text or "").strip()

        js = extract_json_from_text(raw) or raw
        data = json.loads(js)

        fields = {
            "Model": (data.get("model") or "") if isinstance(data, dict) else "",
            "Serial": (data.get("serial") or "") if isinstance(data, dict) else "",
            "Voltage_V": "" if (data.get("voltage_v") is None) else str(data.get("voltage_v")),
            "Current_A": "" if (data.get("current_a") is None) else str(data.get("current_a")),
            "Power_kW": "" if (data.get("power_kw") is None) else str(data.get("power_kw")),
            "Frequency_Hz": "" if (data.get("frequency_hz") is None) else str(data.get("frequency_hz")),
        }
        return js if js else raw, fields

    except Exception as e:
        msg = str(e)

        # 429 / rate limit: نرجّع رسالة واضحة بدل ما ينهار
        if "429" in msg or "Quota" in msg or "rate" in msg.lower():
            return f"Gemini rate/quota limit. Details: {msg}", empty

        # إذا ردّ كان مقطوع/غير JSON — محاولة ثانية بس إصلاح صيغة الرد
        try:
            resp2 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Fix the following into VALID JSON that matches the schema. Output JSON only:\n\n"
                    + (msg[:1500]),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            raw2 = (resp2.text or "").strip()
            js2 = extract_json_from_text(raw2) or raw2
            data2 = json.loads(js2)

            fields2 = {
                "Model": (data2.get("model") or "") if isinstance(data2, dict) else "",
                "Serial": (data2.get("serial") or "") if isinstance(data2, dict) else "",
                "Voltage_V": "" if (data2.get("voltage_v") is None) else str(data2.get("voltage_v")),
                "Current_A": "" if (data2.get("current_a") is None) else str(data2.get("current_a")),
                "Power_kW": "" if (data2.get("power_kw") is None) else str(data2.get("power_kw")),
                "Frequency_Hz": "" if (data2.get("frequency_hz") is None) else str(data2.get("frequency_hz")),
            }
            return js2 if js2 else raw2, fields2
        except Exception:
            return f"Gemini vision error: {msg}", empty


def analyze_nameplate(pil_img: Image.Image) -> tuple[str, dict]:
    # classic (اختياري)
    classic = classic_ocr_text(pil_img)

    # gemini (الأساسي)
    gem_raw, gem_fields = gemini_extract_fields(pil_img)

    raw_combined = ""
    if classic:
        raw_combined += classic + "\n\n---\n\n"
    if gem_raw:
        raw_combined += gem_raw

    return raw_combined.strip(), gem_fields


# =========================================================
# 4) EXCEL EXPORT (Embedded Images)
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
            1: 10, 2: 18, 3: 10,
            4: 20, 5: 22, 6: 12,
            7: 12, 8: 12, 9: 14,
            10: 22, 11: 22, 12: 30
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

            ws.append([
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
            ])

            r = ws.max_row
            ws.row_dimensions[r].height = 120

            img_path = str(row.get("ImagePath", "")).strip()
            img_path = os.path.normpath(img_path)

            thumb_path = create_thumbnail(img_path)
            if thumb_path and os.path.exists(thumb_path):
                try:
                    xl_img = XLImage(thumb_path)
                    xl_img.anchor = f"{get_column_letter(len(cols))}{r}"
                    ws.add_image(xl_img)
                except Exception:
                    pass

            for c in range(1, len(cols)):
                ws.cell(row=r, column=c).alignment = center

    wb.save(out_path)


# =========================================================
# 5) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate AI - Demo", layout="wide")
st.title("Audit Nameplate AI - Quick Demo")

st.write(
    "Flow: select audit type → facility → place → take/upload photo → AI extracts values → save → export Excel."
)

col1, col2 = st.columns(2)
with col1:
    audit_type = st.selectbox("Audit Type", list(AUDIT_TEMPLATES.keys()))
with col2:
    facility_name = st.text_input("Facility Name", placeholder="e.g., Demo Factory")

default_places = AUDIT_TEMPLATES.get(audit_type, [])

st.divider()
st.subheader("Audit Components (editable list)")
places_text = st.text_area("One place per line", value="\n".join(default_places), height=180)
places = [p.strip() for p in places_text.splitlines() if p.strip()]
if not places:
    st.warning("Please add at least one place to continue.")

st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place", places) if places else None
count = st.number_input("How many instances of this place?", min_value=1, max_value=20, value=1)

if facility_name and place:
    st.info(f"You will capture nameplates for: **{facility_name} → {place} (1..{int(count)})**")

st.divider()
st.subheader("Capture nameplates")

if facility_name and place:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place} #{i}", expanded=(i == 1)):

            camera_photo = st.camera_input(
                f"Take photo for {place} #{i}",
                key=f"cam_{audit_type}_{place}_{i}",
            )

            uploaded = st.file_uploader(
                f"Or upload existing image for {place} #{i}",
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                key=f"upl_{audit_type}_{place}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                try:
                    pil_img = Image.open(image_file)
                except Exception:
                    st.error("Cannot open this image type. If it's HEIC/HEIF, add pillow-heif in requirements.")
                    continue

                pil_img = normalize_pil(pil_img)
                st.image(pil_img, caption="Captured / Uploaded image", use_container_width=True)

                with st.spinner("Analyzing image (Gemini + optional classic OCR)..."):
                    raw, fields = analyze_nameplate(pil_img)

                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model = st.text_input("Model", value=fields.get("Model") or "", key=f"model_{place}_{i}")
                with c2:
                    serial = st.text_input("Serial", value=fields.get("Serial") or "", key=f"serial_{place}_{i}")
                with c3:
                    voltage = st.text_input("Voltage (V)", value=fields.get("Voltage_V") or "", key=f"volt_{place}_{i}")
                with c4:
                    current = st.text_input("Current (A)", value=fields.get("Current_A") or "", key=f"curr_{place}_{i}")
                with c5:
                    power = st.text_input("Power (kW)", value=fields.get("Power_kW") or "", key=f"pwr_{place}_{i}")
                with c6:
                    freq = st.text_input("Frequency (Hz)", value=fields.get("Frequency_Hz") or "", key=f"hz_{place}_{i}")

                notes = st.text_input("Notes (optional)", value="", key=f"notes_{place}_{i}")

                with st.expander("Raw OCR / Gemini JSON / Errors"):
                    st.code(raw[:6000] if raw else "")

                if st.button(f"Save record for {place} #{i}", key=f"save_{place}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)

                    dir_path = os.path.join(OUTPUT_DIR, safe_fac, f"{safe_place}_{i}")
                    os.makedirs(dir_path, exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(dir_path, f"nameplate_{ts}.jpg")
                    pil_img.save(img_path, format="JPEG", quality=90)
                    img_path = os.path.abspath(img_path)

                    row = {
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                        "AuditType": audit_type,
                        "Facility": facility_name,
                        "Place": place,
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

                    st.success("Saved! Record added to CSV with local image.")


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
        st.info("No records for this facility yet. Save at least one record first.")
else:
    st.info("No CSV records yet. Save at least one record first.")


import os
import re
import json
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

import cv2
import numpy as np
import pytesseract

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

import io
import base64

from openai import OpenAI
client = OpenAI()
# نقرأ الـ API key من Streamlit secrets ونجهّز الكلاينت
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# =========================================================
# 0) CONFIG
# =========================================================

AUDIT_TEMPLATES = {
    "Hotels": [
        "Lobby",
        "Rooms",
        "Kitchen",
        "Pool",
        "Boiler Room",
        "Electrical Room",
        "Generator Room",
        "Laundry",
    ],
    "Factories": [
        "Production Line",
        "Workshop / Maintenance",
        "Warehouse",
        "Compressor Room",
        "Boiler Room",
        "Electrical Room",
        "Generator Room",
        "Office Area",
        "Outdoor / Yard",
    ],
    "Schools": [
        "Classrooms",
        "Computer Labs",
        "Science Labs",
        "Administration",
        "Library",
        "Cafeteria",
        "Sports Hall / Outdoor Area",
        "Electrical Room",
        "Generator Room",
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
        "Boiler Room",
        "Electrical Room",
        "Generator Room",
    ],
}

OUTPUT_DIR = "outputs"
THUMBS_DIR = os.path.join(OUTPUT_DIR, "_thumbs")
TABLES_DIR_NAME = "_tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

# Common default install path on Windows
DEFAULT_TESS_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# 1) HELPERS
# =========================================================

def safe_name(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    return s.replace(" ", "_") if s else "AUDIT"


def ensure_tesseract():
    """
    Make pytesseract find the local Tesseract binary.
    Priority:
      1) env var TESSERACT_CMD
      2) default Windows path if exists
      3) else rely on PATH
    """
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    if os.name == "nt" and os.path.exists(DEFAULT_TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD


def get_facility_dir(facility: str) -> str:
    safe_fac = safe_name(facility)
    d = os.path.join(OUTPUT_DIR, safe_fac)
    os.makedirs(d, exist_ok=True)
    return d


def get_tables_dir(facility: str) -> str:
    d = os.path.join(get_facility_dir(facility), TABLES_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def get_place_csv_path(facility: str, place: str) -> str:
    safe_place = safe_name(place)
    return os.path.join(get_tables_dir(facility), f"{safe_place}.csv")


def append_record_to_csv(row: dict, csv_path: str):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def load_facility_records(facility: str) -> pd.DataFrame:
    """
    Load all place CSVs under:
      outputs/<Facility>/_tables/*.csv
    """
    try:
        tables_dir = get_tables_dir(facility)
        if not os.path.exists(tables_dir):
            return pd.DataFrame()

        dfs = []
        for fn in os.listdir(tables_dir):
            if fn.lower().endswith(".csv"):
                p = os.path.join(tables_dir, fn)
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True, sort=False)
    except Exception:
        return pd.DataFrame()


def create_thumbnail(img_path: str) -> str | None:
    """
    Create a small thumbnail for embedding in Excel.
    Uses hashed filename to avoid collisions.
    """
    try:
        if not img_path:
            return None

        img_path = str(img_path).strip()
        img_path = os.path.normpath(img_path)

    except Exception:
        return None

    try:
        if not os.path.exists(img_path):
            return None

        key = hashlib.md5(img_path.encode("utf-8")).hexdigest()[:12]
        thumb_path = os.path.join(THUMBS_DIR, f"{key}_thumb.png")

        pil = Image.open(img_path)
        pil = pil.convert("RGB")
        pil.thumbnail((260, 160))
        pil.save(thumb_path, format="PNG")
        return thumb_path
    except Exception:
        return None


# =========================================================
# 2) OCR PIPELINE (محسَّن)
# =========================================================

def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    """
    تحسين الصورة قبل الـOCR:
    - تكبير
    - Grayscale
    - إزالة نويز
    - تحسين كونتراست (CLAHE)
    - Threshold (OTSU)
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # تكبير الصورة حسب حجمها الأصلي
    h, w = img.shape[:2]
    scale = 1.8 if max(h, w) < 1400 else 1.2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # إزالة نويز
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # تحسين الكونتراست
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold أوتوماتيكي
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # لو الصورة داكنة جداً، نعكس
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    return th





def call_ai_vision(pil_img: Image.Image) -> dict:
    """
    تستخدم نموذج رؤية من OpenAI لقراءة لوحة بيانات الجهاز (nameplate)
    وترجع نفس الحقول اللي عندنا في المشروع.
    """
    try:
        # نحول الصورة لـ bytes ثم Base64
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "You are helping with an energy audit. "
            "You are given a photo of an equipment nameplate. "
            "Read the text carefully and extract these fields if present:\n"
            "- model (string)\n"
            "- serial (string)\n"
            "- voltage_v (number, volts)\n"
            "- current_a (number, amps)\n"
            "- power_kw (number, kW)\n"
            "- frequency_hz (number, Hz)\n"
            "If a field is missing, use null. "
            "Return ONLY valid JSON with exactly these keys."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured technical data from equipment nameplate photos.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                },
            ],
        )

        content = completion.choices[0].message.content
        data = json.loads(content)

        # نطابق أسماء المفاتيح مع نظامنا
        return {
            "Model": data.get("model"),
            "Serial": data.get("serial"),
            "Voltage_V": (
                str(data.get("voltage_v"))
                if data.get("voltage_v") is not None
                else None
            ),
            "Current_A": (
                str(data.get("current_a"))
                if data.get("current_a") is not None
                else None
            ),
            "Power_kW": (
                str(data.get("power_kw"))
                if data.get("power_kw") is not None
                else None
            ),
            "Frequency_Hz": (
                str(data.get("frequency_hz"))
                if data.get("frequency_hz") is not None
                else None
            ),
        }

    except Exception as e:
        # لو صار خطأ (ما في API key, مشكلة نت, ...الخ) نرجّع قاموس فاضي
        print("AI vision error:", e)
        return {}

# =========================================================
# 2) OCR + AI VISION PIPELINE
# =========================================================

def classic_ocr_text(pil_img: Image.Image) -> str:
    """محاولة سريعة مع Tesseract (لو مش موجود، نرجع نص فاضي بدون ما نكسر التطبيق)."""
    try:
        ensure_tesseract()
        proc = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(
            proc,
            config="--oem 3 --psm 6",
            lang="eng"
        )
        return text or ""
    except pytesseract.pytesseract.TesseractNotFoundError:
        # على Streamlit Cloud رح نوصل لهون – عادي
        return ""
    except Exception:
        return ""


def ai_extract_fields(pil_img: Image.Image) -> tuple[str, dict]:
    """
    نستخدم نموذج OpenAI Vision عشان يقرأ النيمبليت
    ويرجع JSON فيه القيم اللي بدنا ياها.
    """
    try:
        # نحول الصورة لـ base64
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """
أنت خبير تدقيق كهربائي. عندك صورة Nameplate واحدة لجهاز.
اقرأ كل المعلومات المهمة واستخرج القيم التالية فقط:

- model  : كود الموديل / type
- serial : الرقم التسلسلي كما هو مكتوب
- voltage_v : جهد التغذية بالفولت (رقم واحد، لو مكتوب 220-240V خذ 230 مثلاً)
- current_a : التيار بالأمبير
- power_kw  : الاستطاعة بالكيلوواط (لو مكتوب W حوّلها إلى kW بثلاث منازل عشرية)
- frequency_hz : التردد بالهرتز (50 أو 60 غالبًا)

ارجع **JSON فقط** بدون أي شرح إضافي، مثلاً:

{
  "model": "TCM80C6PIZ(EX)",
  "serial": "509501000",
  "voltage_v": 230,
  "current_a": 2.0,
  "power_kw": 0.4,
  "frequency_hz": 50
}

لو حقل غير موجود على النيمبليت، رجّعه بالقيمة null.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    },
                ],
            }],
            response_format={"type": "json_object"},
            max_output_tokens=300,
        )

        json_text = response.output_text
        data = json.loads(json_text)

        fields = {
            "Model": data.get("model"),
            "Serial": data.get("serial"),
            "Voltage_V": str(data.get("voltage_v") or "") or "",
            "Current_A": str(data.get("current_a") or "") or "",
            "Power_kW": str(data.get("power_kw") or "") or "",
            "Frequency_Hz": str(data.get("frequency_hz") or "") or "",
        }
        return json_text, fields

    except Exception as e:
        print("AI vision error:", e)
        empty = {
            "Model": "",
            "Serial": "",
            "Voltage_V": "",
            "Current_A": "",
            "Power_kW": "",
            "Frequency_Hz": "",
        }
        return "", empty


def analyze_nameplate(pil_img: Image.Image) -> tuple[str, dict]:
    """
    دالة موحّدة: تحاول Tesseract (لو موجود) + AI Vision
    وترجع:
      raw_text  -> نخزنه في RawOCR
      fields    -> نعبّي فيه الحقول على الشاشة + بالملف
    """
    classic = classic_ocr_text(pil_img)  # ممكن يكون فاضي على الكلاود
    ai_json, ai_fields = ai_extract_fields(pil_img)

    # فقط عشان الديبَغ: نخزن كل الإشيين في RawOCR
    raw_combined = ""
    if classic:
        raw_combined += classic + "\n\n---\n\n"
    if ai_json:
        raw_combined += ai_json

    return raw_combined, ai_fields

# =========================================================
# 3) EXCEL REPORT (Embedded Images)
# =========================================================

def build_excel_report(df: pd.DataFrame, out_path: str):
    """
    Create Excel with:
      - Sheet per Place
      - PlaceIndex column
      - Embedded thumbnail image in last column
    Expected df columns:
      Timestamp, AuditType, Facility, Place, PlaceIndex,
      Model, Serial, Voltage_V, Current_A, Power_kW, Frequency_Hz, ImagePath, Notes, RawOCR
    """

    cols = [
        "Record ID", "Place Name", "PlaceIndex",
        "Model", "Serial",
        "Voltage (V)", "Current (A)", "Power (kW)", "Frequency (Hz)",
        "Capture DateTime", "Notes", "Nameplate Image"
    ]

    header_fill = PatternFill("solid", fgColor="1F2937")
    header_font = Font(color="FFFFFF", bold=True)
    center = Alignment(vertical="center", wrap_text=True)

    def set_col_widths(ws):
        widths = {
            1: 10,  # Record ID
            2: 18,  # Place Name
            3: 10,  # PlaceIndex
            4: 16,  # Model
            5: 18,  # Serial
            6: 12,  # Voltage
            7: 12,  # Current
            8: 12,  # Power
            9: 14,  # Frequency
            10: 20, # DateTime
            11: 18, # Notes
            12: 28, # Image
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

            place_index = row.get("PlaceIndex", "")
            model = row.get("Model", "")
            serial = row.get("Serial", "")
            voltage = row.get("Voltage_V", "")
            current = row.get("Current_A", "")
            power = row.get("Power_kW", "")
            freq = row.get("Frequency_Hz", "")
            ts = row.get("Timestamp", "")
            notes = row.get("Notes", "") or ""
            img_path = row.get("ImagePath", "")

            ws.append([
                record_id, place, place_index,
                model, serial,
                voltage, current, power, freq,
                ts, notes, ""
            ])

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


# =========================================================
# 4) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate OCR - Demo", layout="wide")
st.title("Audit Nameplate OCR - Quick Demo")

st.write(
    "Proof-of-concept for your audit flow: "
    "select audit type → facility → area → upload nameplate → OCR (assistive) → auto log → export Excel."
)

col1, col2 = st.columns(2)
with col1:
    audit_type = st.selectbox("Audit Type", list(AUDIT_TEMPLATES.keys()))
with col2:
    facility_name = st.text_input("Facility Name", placeholder="e.g., Demo Hotel")

default_places = AUDIT_TEMPLATES.get(audit_type, [])

st.divider()
st.subheader("Audit Components (editable list)")
places_text = st.text_area("One place per line", value="\n".join(default_places), height=160)
places = [p.strip() for p in places_text.splitlines() if p.strip()]

if not places:
    st.warning("Please add at least one place to continue.")

st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place", places) if places else None
count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1
)

if facility_name and place:
    st.info(f"You will capture nameplates for: **{facility_name} → {place} (1..{int(count)})**")

st.divider()

if facility_name and place:
    for i in range(1, int(count) + 1):
        with st.expander(f"{place} #{i}", expanded=(i == 1)):
            uploaded = st.file_uploader(
                f"Upload Nameplate image for {place} #{i}",
                type=["png", "jpg", "jpeg"],
                key=f"upl_{audit_type}_{place}_{i}",
            )

            if uploaded:
                # 1) عرض الصورة
                pil_img = Image.open(uploaded)
                st.image(
                    pil_img,
                    caption="Uploaded image",
                    use_container_width=True,
                )

                # 2) تحليل الصورة (AI Vision + OCR الكلاسيكي)
                with st.spinner("تحليل الصورة واستخراج القيم..."):
                    raw, fields = analyze_nameplate(pil_img)

                # 3) الحقول القابلة للتعديل من المستخدم
                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model = st.text_input(
                        "Model",
                        value=fields.get("Model") or "",
                        key=f"model_{place}_{i}",
                    )
                with c2:
                    serial = st.text_input(
                        "Serial",
                        value=fields.get("Serial") or "",
                        key=f"serial_{place}_{i}",
                    )
                with c3:
                    voltage = st.text_input(
                        "Voltage (V)",
                        value=fields.get("Voltage_V") or "",
                        key=f"volt_{place}_{i}",
                    )
                with c4:
                    current = st.text_input(
                        "Current (A)",
                        value=fields.get("Current_A") or "",
                        key=f"curr_{place}_{i}",
                    )
                with c5:
                    power = st.text_input(
                        "Power (kW)",
                        value=fields.get("Power_kW") or "",
                        key=f"pwr_{place}_{i}",
                    )
                with c6:
                    freq = st.text_input(
                        "Frequency (Hz)",
                        value=fields.get("Frequency_Hz") or "",
                        key=f"hz_{place}_{i}",
                    )

                notes = st.text_input(
                    "Notes (optional)",
                    value="",
                    key=f"notes_{place}_{i}",
                )

                # 4) نص الـ OCR / AI JSON الخام
                with st.expander("Raw OCR / AI JSON"):
                    st.code(raw[:3000] if raw else "")

                # 5) زر الحفظ
                if st.button(
                    f"Save record for {place} #{i}",
                    key=f"save_{place}_{i}",
                ):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)

                    # مسار حفظ الصورة و JSON
                    dir_path = os.path.join(
                        OUTPUT_DIR,
                        safe_fac,
                        f"{safe_place}_{i}",
                    )
                    os.makedirs(dir_path, exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(dir_path, f"nameplate_{ts}.png")
                    pil_img.save(img_path)
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
st.subheader("Current Place Table")

if facility_name and place:
    place_csv = get_place_csv_path(facility_name, place)

    if os.path.exists(place_csv):
        df_place = pd.read_csv(place_csv)
        st.dataframe(df_place.tail(50), use_container_width=True)
        st.caption(f"Saved at: {place_csv}")
    else:
        st.write("No records yet for this place.")
else:
    st.write("Select facility and place first to see the place table.")

# =========================================================
# 5) EXPORT EXCEL (Embedded Images)
# =========================================================

st.divider()
st.subheader("Export Excel (embedded images by Place)")

if facility_name:
    df_use = load_facility_records(facility_name)

    if not df_use.empty:
        safe_fac = safe_name(facility_name)
        xlsx_path = os.path.join(OUTPUT_DIR, f"AUDIT_{safe_fac}_{datetime.now().strftime('%Y%m%d')}.xlsx")

        if st.button("Generate Excel with embedded images"):
            # Ensure expected columns exist
            for col in ["Notes", "RawOCR"]:
                if col not in df_use.columns:
                    df_use[col] = ""

            build_excel_report(df_use, xlsx_path)
            st.success("Excel generated successfully!")

            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "Download Excel Report",
                    data=f,
                    file_name=os.path.basename(xlsx_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("No records for this facility yet. Save at least one nameplate record.")
else:
    st.info("Enter a Facility Name first.")

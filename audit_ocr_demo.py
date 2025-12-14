import os
import re
import io
import json
import base64
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

from openai import OpenAI

# =========================================================
# 0) OpenAI CLIENT (من الـ secrets)
# =========================================================

# توقّعنا إنك مبرمجة OPENAI_API_KEY في Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================================================
# 1) CONFIG
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
        "Chiller Plant",
        "Chilled Water Network",
        "Compressed Air Network",
        "Hot Water Network",
        "Steam Network & Steam Generator",
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
CSV_PATH = os.path.join(OUTPUT_DIR, "audit_nameplate_records.csv")
THUMBS_DIR = os.path.join(OUTPUT_DIR, "_thumbs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

# useful لو شغّلتيه محليًا على ويندوز
DEFAULT_TESS_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# 2) HELPERS
# =========================================================

def safe_name(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    return s.replace(" ", "_") if s else "AUDIT"


def ensure_tesseract():
    """تهيئة مسار Tesseract لو متوفر (محليًا على ويندوز)."""
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    if os.name == "nt" and os.path.exists(DEFAULT_TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD


def append_record(row: dict):
    """إضافة صف جديد إلى ملف CSV (أو إنشاء الملف إذا غير موجود)."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(CSV_PATH, index=False)


def create_thumbnail(img_path: str) -> str | None:
    """إنشاء ثَمبنيل صغير للصورة لاستخدامه في ملف Excel."""
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


# =========================================================
# 3) OCR + AI VISION
# =========================================================

def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    """معالجة أولية للصورة لو استعملنا Tesseract."""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # تكبير بسيط
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return th


def classic_ocr_text(pil_img: Image.Image) -> str:
    """Tesseract OCR – لو غير متوفر يرجّع نص فاضي بدون ما يكسر التطبيق."""
    try:
        ensure_tesseract()
        proc = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(
            proc,
            config="--oem 3 --psm 6",
            lang="eng",
        )
        return text or ""
    except pytesseract.pytesseract.TesseractNotFoundError:
        return ""
    except Exception:
        return ""


def ai_extract_fields(pil_img: Image.Image) -> tuple[str, dict]:
    """
    استخدام OpenAI Vision لاستخراج الحقول من صورة الـ Nameplate.
    يرجّع:
      - json_text: سترنغ JSON خام للعرض في RawOCR
      - fields: dict فيه Model / Serial / Voltage_V / Current_A / Power_kW / Frequency_Hz
    """
    empty = {
        "Model": "",
        "Serial": "",
        "Voltage_V": "",
        "Current_A": "",
        "Power_kW": "",
        "Frequency_Hz": "",
    }

    try:
        # تحويل الصورة لـ base64
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        prompt = """
أنت خبير تدقيق كهربائي. عندك صورة Nameplate واحدة لجهاز.
اقرأ كل المعلومات المهمة واستخرج القيم التالية فقط:

- model       : كود الموديل / type
- serial      : الرقم التسلسلي كما هو مكتوب
- voltage_v   : جهد التغذية بالفولت (رقم واحد؛ لو مكتوب 220-240V خذ قيمة وسطية مثل 230)
- current_a   : التيار بالأمبير
- power_kw    : الاستطاعة بالكيلوواط (لو مكتوب W حوّلها إلى kW بثلاث منازل عشرية)
- frequency_hz: التردد بالهرتز (50 أو 60 غالبًا)

أرجع JSON فقط بدون أي شرح إضافي، مثلاً:

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
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_output_tokens=300,
        )

        # نص JSON الناتج
        json_text = response.output_text

        data = json.loads(json_text)

        fields = {
            "Model": data.get("model") or "",
            "Serial": data.get("serial") or "",
            "Voltage_V": str(data.get("voltage_v") or "") or "",
            "Current_A": str(data.get("current_a") or "") or "",
            "Power_kW": str(data.get("power_kw") or "") or "",
            "Frequency_Hz": str(data.get("frequency_hz") or "") or "",
        }
        return json_text, fields

    except Exception as e:
        # نعرض الخطأ في واجهة Streamlit ونرجّع قيم فاضية
        st.error(f"AI vision error: {e}")
        return "", empty


def analyze_nameplate(pil_img: Image.Image) -> tuple[str, dict]:
    """
    دالة موحّدة:
      - تحاول Tesseract (لو موجود محليًا)
      - تستعمل AI Vision
      - ترجع نص مشترك + الحقول المستخرجة
    """
    classic = classic_ocr_text(pil_img)
    ai_json, ai_fields = ai_extract_fields(pil_img)

    raw_combined = ""
    if classic:
        raw_combined += classic + "\n\n---\n\n"
    if ai_json:
        raw_combined += ai_json

    return raw_combined, ai_fields


# =========================================================
# 4) EXCEL EXPORT (مع الصور)
# =========================================================

def build_excel_report(df: pd.DataFrame, out_path: str):
    """
    إنشاء ملف Excel:
      - شيت لكل Place
      - أعمدة أساسية + Current
      - صورة مصغّرة في آخر عمود
    """

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
            1: 10,   # ID
            2: 18,   # Place
            3: 10,   # index
            4: 20,   # Model
            5: 22,   # Serial
            6: 12,   # V
            7: 12,   # A
            8: 12,   # kW
            9: 14,   # Hz
            10: 22,  # timestamp
            11: 22,  # notes
            12: 30,  # image
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
            notes = row.get("Notes", "")
            img_path = row.get("ImagePath", "")

            ws.append(
                [
                    record_id,
                    place,
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


# =========================================================
# 5) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate OCR - Demo", layout="wide")
st.title("Audit Nameplate OCR - Quick Demo")

st.write(
    "Proof-of-concept for your audit flow: "
    "select audit type → facility → area → upload nameplate → OCR / AI assist → auto log → export Excel."
)

# اختيار نوع الأوديت والمنشأة
col1, col2 = st.columns(2)
with col1:
    audit_type = st.selectbox("Audit Type", list(AUDIT_TEMPLATES.keys()))
with col2:
    facility_name = st.text_input("Facility Name", placeholder="e.g., Demo Factory")

default_places = AUDIT_TEMPLATES.get(audit_type, [])

# قائمة الأماكن القابلة للتعديل
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

# اختيار المكان وعدد التكرارات
st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place", places) if places else None
count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1,
)

if facility_name and place:
    st.info(
        f"You will capture nameplates for: "
        f"**{facility_name} → {place} (1..{int(count)})**"
    )

st.divider()
st.subheader("Capture nameplates")

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

                # 2) تحليل الصورة (OCR + AI)
                with st.spinner("تحليل الصورة واستخراج القيم..."):
                    raw, fields = analyze_nameplate(pil_img)

                # 3) الحقول القابلة للتعديل
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

                # 4) نصّ الـ OCR / JSON
                with st.expander("Raw OCR / AI JSON"):
                    st.code(raw[:3000] if raw else "")

                # 5) زر الحفظ
                if st.button(
                    f"Save record for {place} #{i}",
                    key=f"save_{place}_{i}",
                ):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)

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

# =========================================================
# 6) CURRENT CSV PREVIEW
# =========================================================

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)
    st.dataframe(df_all.tail(50), use_container_width=True)
    st.caption(f"Saved at: {CSV_PATH}")
else:
    st.write("No records yet.")

# =========================================================
# 7) EXPORT EXCEL
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
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )
    else:
        st.info("No records for this facility yet. Save at least one nameplate record.")
else:
    st.info("No CSV records yet. Save at least one nameplate record first.")

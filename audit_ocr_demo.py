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


def run_ocr(pil_img: Image.Image) -> str:
    ensure_tesseract()
    proc = preprocess_for_ocr(pil_img)

    # نحدّد نوع النص (بلوك) ونقيّد الحروف شوي
    config = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./:-%"
    )

    text = pytesseract.image_to_string(proc, config=config, lang="eng")
    return text or ""


def extract_fields(raw_text: str) -> dict:
    """
    استخراج Model / Serial / Voltage / Current / Power / Frequency
    مع مراعاة معظم الطرق الشائعة لكتابتها عالميًا (إنجليزي + لغات أخرى + عربي).
    """
    import re

    raw_text = raw_text or ""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    text = " ".join(lines)

    # ==== كلمات مفتاحية لتمييز الحقول (labels) ====

    # Serial number: Serial No, S/N, Serien-Nr, N° de série, رقم تسلسلي, ...
    SERIAL_KEYWORDS = [
        "serial", "serial no", "serial number", "ser no", "ser. no","S/N","S.NO"
        "s/n", "s-n", "sn",
        "serien-nr",             # ألماني
        "n° de série", "no de serie",  # فرنسي / إسباني
        "nr seryjny",            # بولندي
        "رقم تسلسلي", "الرقم التسلسلي",
    ]

    # Model / type / part: Model, Model Code, Type No, Cat No, Part No, P/N, Article No, Ref, ...
    MODEL_KEYWORDS = [
        "model code", "model no", "model number", "model",
        "type no", "type",
        "cat no", "catalog no", "cat. no",
        "part no", "p/n",
        "art no", "article no",
        "ref", "reference",
        "modèle",      # فرنسي
        "tipo",        # إيطالي / إسباني
        "رقم الموديل", "موديل",
    ]

    # Voltage: Voltage, Un, Ue, Volt, V~, VAC, VDC, الجهد, ...
    VOLTAGE_KEYWORDS = [
        "voltage", "rated voltage", "un", "ue", "u=", "u~",
        "volt",
        "tension", "spannung", "tensión", "tensione",
        "الجهد", "فولت",
    ]

    # Current: Current, FLA, FLC, RLA, amp, amps, ampere, التيار, ...
    CURRENT_KEYWORDS = [
        "current", "rated current", "flc", "fla", "rla",
        "amp", "amps", "ampere", "amperes",
        "intensidad", "corrente",
        "التيار", "شدة التيار",
    ]

    # Power: Power, kW, W, VA, kVA, HP, ...
    POWER_KEYWORDS = [
        "power", "rated power", "input power", "output power",
        "kw", "w", "va", "kva", "hp",
    ]

    # Frequency: Frequency, Hz, F=, التردد, ...
    FREQ_KEYWORDS = [
        "frequency", "freq", "hz", "f=",
        "التردد",
    ]

    # ===== Helpers =====

    def format_num(val):
        """تخلي الرقم بشكل لطيف (يكون string، بدون أصفار زيادة)."""
        try:
            v = float(str(val).replace(",", "."))
        except Exception:
            return None
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return str(round(v, 3))

    def pick_code_from_line(ln: str):
        """
        يختار كود شبِه بالموديل/السيريال:
        - أحرف + أرقام
        - طول >= 4
        """
        tokens = re.findall(r"[A-Z0-9][A-Z0-9\-_\/\.]{3,}", ln, flags=re.IGNORECASE)
        codes = [t for t in tokens if re.search(r"\d", t)]
        return codes[0] if codes else None

    # ===== Serial =====
    serial = None
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in SERIAL_KEYWORDS):
            cand = pick_code_from_line(ln)
            if cand:
                serial = cand
                break

    # ===== Model =====
    model = None
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in MODEL_KEYWORDS):
            cand = pick_code_from_line(ln)
            if cand:
                model = cand
                break

    # ===== Voltage (V) =====
    voltage = None

    # 1) أنماط مباشرة فيها وحدة V / VAC / VDC / kV
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:v|vac|vdc|kv)\b", text, flags=re.IGNORECASE)
    if m:
        voltage = format_num(m.group(1))

    # 2) لو ما لقينا، نحاول أسطر فيها كلمات جهد (Un, Voltage, الجهد...)
    if voltage is None:
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in VOLTAGE_KEYWORDS):
                m2 = re.search(r"(\d+(?:\.\d+)?)", ln)
                if m2:
                    voltage = format_num(m2.group(1))
                    break

    # ===== Current (A) =====
    current = None

    # 1) رقم مع A/Amp/Amps/mA
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:a|amp|amps|ma)\b", text, flags=re.IGNORECASE)
    if m:
        current = format_num(m.group(1))

    # 2) لو ما لقينا، ندور في أسطر اللي فيها كلمات Current/FLA/التيار...
    if current is None:
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in CURRENT_KEYWORDS):
                m2 = re.search(r"(\d+(?:\.\d+)?)", ln)
                if m2:
                    current = format_num(m2.group(1))
                    break

    # ===== Frequency (Hz) =====
    freq = None

    m = re.search(r"(\d+(?:\.\d+)?)\s*hz\b", text, flags=re.IGNORECASE)
    if m:
        freq = format_num(m.group(1))

    if freq is None:
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in FREQ_KEYWORDS):
                m2 = re.search(r"(\d+(?:\.\d+)?)", ln)
                if m2:
                    freq = format_num(m2.group(1))
                    break

    # ===== Power (kW / W) =====
    power_kw = None

    # kW مباشرة
    m = re.search(r"(\d+(?:\.\d+)?)\s*kw\b", text, flags=re.IGNORECASE)
    if m:
        power_kw = format_num(m.group(1))
    else:
        # نحاول W ونحوّل لـ kW
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*w\b", text, flags=re.IGNORECASE)
        if m2:
            try:
                wv = float(m2.group(1).replace(",", "."))
                power_kw = format_num(wv / 1000.0)
            except Exception:
                power_kw = m2.group(1)

    return {
        "Model": model,
        "Serial": serial,
        "Voltage_V": voltage,
        "Current_A": current,
        "Power_kW": power_kw,
        "Frequency_Hz": freq,
    }

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
                key=f"upl_{audit_type}_{place}_{i}"
            )

            if uploaded:
                # 1) عرض الصورة
                pil_img = Image.open(uploaded)
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                raw = ""
                fields_ocr = {}
                fields_ai = {}

                # 2) OCR الكلاسيكي (Tesseract + Regex)
                try:
                    with st.spinner("Running classic OCR (Tesseract)..."):
                        raw = run_ocr(pil_img)
                    fields_ocr = extract_fields(raw)
                except pytesseract.pytesseract.TesseractNotFoundError:
                    st.warning(
                        "Tesseract not found. Classic OCR is disabled. "
                        "Install it or set TESSERACT_CMD if you want it."
                    )
                    fields_ocr = {}

                # 3) AI Vision (OpenAI) – اختياري إذا في OPENAI_API_KEY
                if os.environ.get("OPENAI_API_KEY"):
                    with st.spinner("Running AI vision (OpenAI)..."):
                        fields_ai = call_ai_vision(pil_img)
                else:
                    st.info(
                        "AI vision is disabled (no OPENAI_API_KEY set). "
                        "Only classic OCR is used."
                    )
                    fields_ai = {}

                # 4) دمج نتائج AI + OCR
                fields = {}
                fields["Model"] = fields_ai.get("Model") or fields_ocr.get("Model")
                fields["Serial"] = fields_ai.get("Serial") or fields_ocr.get("Serial")
                fields["Voltage_V"] = (
                    fields_ai.get("Voltage_V") or fields_ocr.get("Voltage_V")
                )
                fields["Current_A"] = (
                    fields_ai.get("Current_A") or fields_ocr.get("Current_A")
                )
                fields["Power_kW"] = (
                    fields_ai.get("Power_kW") or fields_ocr.get("Power_kW")
                )
                fields["Frequency_Hz"] = (
                    fields_ai.get("Frequency_Hz") or fields_ocr.get("Frequency_Hz")
                )

                # 5) الحقول القابلة للتعديل من المستخدم
                st.markdown("### Extracted fields (edit if needed)")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    model = st.text_input(
                        "Model",
                        value=fields.get("Model") or "",
                        key=f"model_{place}_{i}"
                    )
                with c2:
                    serial = st.text_input(
                        "Serial",
                        value=fields.get("Serial") or "",
                        key=f"serial_{place}_{i}"
                    )
                with c3:
                    voltage = st.text_input(
                        "Voltage (V)",
                        value=fields.get("Voltage_V") or "",
                        key=f"volt_{place}_{i}"
                    )
                with c4:
                    current = st.text_input(
                        "Current (A)",
                        value=fields.get("Current_A") or "",
                        key=f"curr_{place}_{i}"
                    )
                with c5:
                    power = st.text_input(
                        "Power (kW)",
                        value=fields.get("Power_kW") or "",
                        key=f"pwr_{place}_{i}"
                    )
                with c6:
                    freq = st.text_input(
                        "Frequency (Hz)",
                        value=fields.get("Frequency_Hz") or "",
                        key=f"hz_{place}_{i}"
                    )

                notes = st.text_input(
                    "Notes (optional)",
                    value="",
                    key=f"notes_{place}_{i}"
                )

                # 6) نص الـ OCR الخام للفلترة اليدوية لو حبيتي
                with st.expander("Raw OCR Text"):
                    st.code(raw[:3000] if raw else "")

                # 7) زر الحفظ
                if st.button(f"Save record for {place} #{i}", key=f"save_{place}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)

                    # مسار حفظ الصورة و JSON
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

# =========================================================
# 4.5) CURRENT PLACE TABLE PREVIEW
# =========================================================

st.divider()
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

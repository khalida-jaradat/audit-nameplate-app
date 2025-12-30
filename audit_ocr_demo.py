import os
import re
import io
import json
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Tuple, Dict

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

# (اختياري) دعم HEIC/HEIF لو أضفتي pillow-heif في requirements.txt
HEIF_ENABLED = False
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False


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
        "Steam Network ",
        " Steam Generator",
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

# common default on Windows (لو شغّلتي محليًا)
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
    يضبط مسار tesseract:
    - لو TESSERACT_CMD موجود
    - لو ويندوز والمسار الافتراضي موجود
    - لو لينكس (Streamlit Cloud) يلقطه من PATH
    """
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    if os.name == "nt" and os.path.exists(DEFAULT_TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD
        return

    tpath = shutil.which("tesseract")
    if tpath:
        pytesseract.pytesseract.tesseract_cmd = tpath


def append_record(row: dict):
    """إضافة صف جديد إلى ملف CSV (أو إنشاء الملف إذا غير موجود)."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(CSV_PATH, index=False)


def create_thumbnail(img_path: str) -> Optional[str]:
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
# 2) OCR + FIELD EXTRACTION
# =========================================================

def _score_text(t: str) -> int:
    # كل ما زاد النص المنطقي (حروف/أرقام) اعتبرناه أفضل
    t = t or ""
    return sum(ch.isalnum() for ch in t)


def _preprocess_variants(pil_img: Image.Image):
    """
    يرجّع عدة نسخ معالجة للصورة لتحسين OCR
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # تكبير
    img = cv2.resize(img, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    den = cv2.bilateralFilter(gray, 9, 75, 75)

    # threshold 1
    th1 = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # threshold inverted
    th2 = cv2.bitwise_not(th1)

    # otsu
    _, th3 = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [gray, den, th1, th2, th3]


def _ocr_once(img_arr: np.ndarray, psm: int, lang: str) -> str:
    config = f"--oem 3 --psm {psm}"
    try:
        txt = pytesseract.image_to_string(img_arr, config=config, lang=lang)
        return txt or ""
    except Exception:
        return ""


def classic_ocr_text(pil_img: Image.Image) -> str:
    """
    OCR على Tesseract:
    - يجرب دوران 0/90/180/270
    - يجرب أكثر من preprocessing + أكثر من psm
    """
    ensure_tesseract()

    # جرّبي 4 زوايا
    angles = [0, 90, 180, 270]
    best_text = ""
    best_score = -1

    for ang in angles:
        rotated = pil_img.rotate(ang, expand=True) if ang else pil_img
        variants = _preprocess_variants(rotated)

        # جرّبي إنجليزي فقط (الأكثر شيوعًا) + fallback eng+ara
        for lang in ["eng", "eng+ara"]:
            for v in variants:
                # psm 6 و 11 غالبًا الأفضل للنيم بليت
                for psm in [6, 11, 4]:
                    txt = _ocr_once(v, psm=psm, lang=lang)
                    sc = _score_text(txt)
                    if sc > best_score:
                        best_score = sc
                        best_text = txt

    return best_text.strip()


def _norm_text(t: str) -> str:
    t = t or ""
    t = t.replace("—", "-").replace("–", "-").replace("−", "-")
    t = t.replace("Ｏ", "0")
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _parse_voltage(raw: str) -> str:
    raw = raw or ""
    raw = raw.replace("VAC", "V").replace("V~", "V").replace("VDC", "V").replace("DC", "DC")

    # range: 220-240V أو 220 - 240 V
    m = re.search(r"(\d{2,4})\s*-\s*(\d{2,4})\s*V\b", raw, re.IGNORECASE)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        mid = (a + b) / 2.0
        # لو الكل int رجعيه int
        return str(int(mid)) if abs(mid - int(mid)) < 1e-6 else f"{mid:.1f}"

    # single: 19V / 380V
    m = re.search(r"\b(\d{2,4}(?:\.\d+)?)\s*V\b", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    return ""


def _parse_current(raw: str) -> str:
    raw = raw or ""

    # mA
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*mA\b", raw, re.IGNORECASE)
    if m:
        ma = float(m.group(1))
        a = ma / 1000.0
        return f"{a:.3f}".rstrip("0").rstrip(".")

    # A
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*A\b", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    return ""


def _parse_frequency(raw: str) -> str:
    raw = raw or ""

    # 50/60 Hz
    m = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*Hz\b", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    m = re.search(r"\b(\d{2,3})\s*Hz\b", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    return ""


def _parse_power_kw(raw: str) -> str:
    raw = raw or ""

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*kW\b", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    # W -> kW
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*W\b", raw, re.IGNORECASE)
    if m:
        w = float(m.group(1))
        kw = w / 1000.0
        return f"{kw:.3f}".rstrip("0").rstrip(".")

    return ""


def extract_fields_from_ocr(raw_text: str) -> Dict[str, str]:
    """
    يستخرج:
    Model, Serial, Voltage_V, Current_A, Power_kW, Frequency_Hz
    """
    t = _norm_text(raw_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    joined = " \n".join(lines)

    # --- Serial ---
    serial = ""
    serial_patterns = [
        r"\bS\/N\b\s*[:#]?\s*([A-Z0-9\-\/]{4,})",
        r"\bSN\b\s*[:#]?\s*([A-Z0-9\-\/]{4,})",
        r"\bSERIAL\b(?:\s*NO\.?)?\s*[:#]?\s*([A-Z0-9\-\/]{4,})",
    ]
    for pat in serial_patterns:
        m = re.search(pat, joined, re.IGNORECASE)
        if m:
            serial = m.group(1).strip()
            break

    # --- Model ---
    model = ""
    model_patterns = [
        r"\bMODEL\b(?:\s*CODE)?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_\/]{3,})",
        r"\bTYPE\b(?:\s*NO\.?)?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_\/]{3,})",
    ]
    for pat in model_patterns:
        m = re.search(pat, joined, re.IGNORECASE)
        if m:
            model = m.group(1).strip()
            break

    # fallback: لو ما لقينا Model نجيب أطول كود شكله موديل
    if not model:
        candidates = []
        for ln in lines:
            # كود فيه أرقام وحروف
            m = re.findall(r"\b[A-Z0-9][A-Z0-9\-_\/]{4,}\b", ln.upper())
            for c in m:
                # استبعد كلمات عامة
                if c in {"SAMSUNG", "MADE", "CHINA", "POWER", "SUPPLY"}:
                    continue
                candidates.append(c)
        if candidates:
            model = sorted(candidates, key=len, reverse=True)[0]

    # --- Voltage / Current / Power / Frequency ---
    voltage = _parse_voltage(joined)
    current = _parse_current(joined)
    power = _parse_power_kw(joined)
    freq = _parse_frequency(joined)

    return {
        "Model": model or "",
        "Serial": serial or "",
        "Voltage_V": voltage or "",
        "Current_A": current or "",
        "Power_kW": power or "",
        "Frequency_Hz": freq or "",
    }


def analyze_nameplate(pil_img: Image.Image) -> Tuple[str, Dict[str, str]]:
    """
    موحّد:
      - OCR
      - استخراج حقول
    """
    raw = classic_ocr_text(pil_img)
    fields = extract_fields_from_ocr(raw)
    return raw, fields


# =========================================================
# 3) EXCEL EXPORT (مع الصور)
# =========================================================

def build_excel_report(df: pd.DataFrame, out_path: str):
    """
    إنشاء ملف Excel:
      - شيت لكل Place (الأساسي من القائمة)
      - عمود Place Name يعرض (Place + CustomArea) في نفس الخانة
      - صورة مصغّرة في آخر عمود
    """
    cols = [
        "Record ID",
        "Place Name",          # combined label
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
            1: 10,
            2: 26,  # Place Name combined
            3: 10,
            4: 22,
            5: 22,
            6: 12,
            7: 12,
            8: 12,
            9: 14,
            10: 22,
            11: 22,
            12: 30,
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

    # sheets grouped by base place
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

            place_label = row.get("PlaceLabel", "") or row.get("Place", "")
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


# =========================================================
# 4) STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Audit Nameplate OCR - Demo", layout="wide")
st.title("Audit Nameplate OCR - Quick Demo")

# Sidebar: OCR Status
st.sidebar.markdown("## System Status")
ensure_tesseract()
tp = shutil.which("tesseract")
if tp:
    try:
        ver = subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0]
        st.sidebar.success(f"OCR: ENABLED ✅\n\n{ver}")
    except Exception:
        st.sidebar.success("OCR: ENABLED ✅")
else:
    st.sidebar.error("OCR: DISABLED ❌ (tesseract not found)")

st.sidebar.write("HEIC support:", "✅" if HEIF_ENABLED else "❌")

st.write(
    "Workflow: select audit type → facility → place → upload/camera nameplate → OCR → edit if needed → save → export Excel."
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
    "One place per line (edit if needed)",
    value="\n".join(default_places),
    height=180,
)
places = [p.strip() for p in places_text.splitlines() if p.strip()]

if not places:
    st.warning("Please add at least one place (one per line).")

# اختيار المكان وعدد التكرارات + اسم منطقة اختياري (ينزل بالأكسل جنب المكان)
st.divider()
st.subheader("Choose a place to capture nameplates")

place = st.selectbox("Place (from your list)", places) if places else None

count = st.number_input(
    "How many instances of this place?",
    min_value=1,
    max_value=20,
    value=1,
)

custom_area = st.text_input(
    "Custom area name (optional) – e.g., Main Kitchen, West Lobby",
    value="",
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

            # تصوير مباشر (مفيد للموبايل)
            camera_photo = st.camera_input(
                f"Take photo for {place_label} #{i}",
                key=f"cam_{audit_type}_{place}_{i}",
            )

            # أو رفع صورة
            uploaded = st.file_uploader(
                f"Or upload existing image for {place_label} #{i}",
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                key=f"upl_{audit_type}_{place}_{i}",
            )

            image_file = camera_photo or uploaded

            if image_file:
                try:
                    pil_img = Image.open(image_file)
                except Exception as e:
                    st.error(f"Can't open this image. If it's HEIC, add pillow-heif. Error: {e}")
                    continue

                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                with st.spinner("Running OCR and extracting fields..."):
                    raw, fields = analyze_nameplate(pil_img)

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

                with st.expander("Raw OCR Text"):
                    st.code(raw[:5000] if raw else "")

                if st.button(f"Save record for {place_label} #{i}", key=f"save_{place}_{i}"):
                    safe_fac = safe_name(facility_name)
                    safe_place = safe_name(place)  # base place folder
                    safe_label = safe_name(place_label)

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
                        "Place": place,                 # base place (للشيت)
                        "PlaceLabel": place_label,       # للعرض (Place + CustomArea)
                        "CustomArea": custom_area.strip(),
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

                    st.success("Saved! Record added to CSV.")

# =========================================================
# 5) CURRENT CSV PREVIEW (فلترة على Facility الحالية)
# =========================================================

st.divider()
st.subheader("Current CSV preview")

if os.path.exists(CSV_PATH):
    df_all = pd.read_csv(CSV_PATH)

    # لو في ملفات قديمة ما فيها PlaceLabel / CustomArea
    if "PlaceLabel" not in df_all.columns:
        df_all["PlaceLabel"] = df_all["Place"].astype(str)
    if "CustomArea" not in df_all.columns:
        df_all["CustomArea"] = ""

    if facility_name:
        df_view = df_all[df_all["Facility"] == facility_name].copy()
    else:
        df_view = df_all.copy()

    # اعرض أهم الأعمدة
    show_cols = [
        "Timestamp", "AuditType", "Facility", "Place", "PlaceLabel", "PlaceIndex",
        "Model", "Serial", "Voltage_V", "Current_A", "Power_kW", "Frequency_Hz"
    ]
    show_cols = [c for c in show_cols if c in df_view.columns]

    st.dataframe(df_view[show_cols].tail(50), use_container_width=True)
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

    if "PlaceLabel" not in df_all.columns:
        df_all["PlaceLabel"] = df_all["Place"].astype(str)
    if "CustomArea" not in df_all.columns:
        df_all["CustomArea"] = ""

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
            for col in ["Notes", "RawOCR", "Current_A", "PlaceLabel", "CustomArea"]:
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

# Data Protocol — มาตรฐานข้อมูล Multi-View ระหว่างเครื่อง CPU และ GPU

เอกสารนี้กำหนด **สัญญาแลกเปลี่ยนข้อมูล (Data Contract)** ระหว่าง `[เครื่อง CPU - คนที่ 1]` และ `[เครื่อง GPU - คนที่ 2]` เพื่อให้โค้ดทั้งสองฝั่งอ่าน–เขียนไฟล์ชุดภาพหลายมุม (Multi-View) ได้ตรงกัน 100%

---

## 1. แนวทางการถ่ายภาพหลายมุม (Multi-View Capture Guidelines)

เพื่อให้โมเดล Multi-View 3D (เช่น `InstantMesh`) และขั้นตอน Classical DIP ทำงานได้อย่างเสถียร แนะนำมาตรฐานการถ่ายภาพดังนี้:

1. **จำนวนมุมมองต่อวัตถุ (Views per Object):**
   - **โหมดมาตรฐาน (Sparse Multi-View):** ถ่าย **6 มุม** (ห่างกันมุมละ 60°: `0°, 60°, 120°, 180°, 240°, 300°`) หรือ **8 มุม** (ห่างกันมุมละ 45°: `0°, 45°, ..., 315°`)
   - **มุมเงย (Elevation):** ถ่ายระดับสายตาหรือก้มเล็กน้อยประมาณ 15°–20° ให้สม่ำเสมอทุกมุม เพื่อให้เห็นทั้งด้านข้างและด้านบนบางส่วน
2. **การล็อกค่ากล้อง (ถ้าทำได้):**
   - สำหรับชุด `normal/`: ล็อก Focus และ Exposure เพื่อลดความเพี้ยนระหว่างมุม
   - สำหรับชุดทดสอบจริง (`low_light/`, `backlight/`, `reflective/`, `similar_bg/`): ปล่อย Auto ตามสภาพแวดล้อมจริงเพื่อทดสอบประสิทธิภาพ Pre-processing ในสัปดาห์ที่ 8 และ 14
3. **การจัดองค์ประกอบภาพ:**
   - วางวัตถุให้อยู่กลางเฟรม ครอบคลุมพื้นที่ประมาณ 50%–70% ของภาพ ไม่ให้ขอบวัตถุโดนตัดขาด

---

## 2. กติกาการตั้งชื่อและโครงสร้างโฟลเดอร์ `data/`

เนื่องจาก 1 วัตถุมีหลายภาพ **ต้องจัดเก็บแยกเป็นโฟลเดอร์ต่อวัตถุ (`<object_id>/`)** ภายใต้สภาพแสง (`<condition>/`) เสมอ

### 2.1 รายชื่อสภาพแสงมาตรฐาน (`<condition>`)
ใช้ชื่อโฟลเดอร์ตัวพิมพ์เล็ก 5 เงื่อนไขตรงตามเกณฑ์สัปดาห์ที่ 14:
- `normal` — แสงสว่างสม่ำเสมอ พื้นหลังเรียบตัดกับสีวัตถุ
- `low_light` — แสงน้อย / ถ่ายในที่มืด (มี Sensor Noise สูง)
- `backlight` — ย้อนแสง (วัตถุมืด ด้านหลังสว่างจ้า)
- `reflective` — วัตถุผิวเงา/โลหะ/พลาสติกสะท้อนแสง
- `similar_bg` — วัตถุสีใกล้เคียงกับพื้นหลัง

### 2.2 รูปแบบชื่อวัตถุ (`<object_id>`) และชื่อไฟล์แต่ละมุม (`view_XXX`)
- `<object_id>`: ตัวพิมพ์เล็ก ไม่มีเว้นวรรค เช่น `ceramic_mug_01`, `sneaker_01`, `toy_robot_02`
- ชื่อไฟล์มุมภาพ: ใช้เลข 3 หลักเรียงตามลำดับมุมรอบตัว (ทวนเข็มนาฬิกาหรือตามเข็มนาฬิกาให้เหมือนกันทั้งชุด) เช่น:
  - `view_000.jpg` (มุมด้านหน้าหลัก 0° — ใช้เป็นมุมอ้างอิงสำหรับ Single-View Baseline ด้วย)
  - `view_001.jpg` (มุมถัดไป)
  - ...
  - `view_005.jpg` (หรือ `view_007.jpg` กรณี 8 มุม)

### 2.3 ตัวอย่างโครงสร้างจริงในทุกขั้นตอนของ Pipeline

```text
data/
├── 01_raw/
│   └── backlight/
│       └── ceramic_mug_01/
│           ├── view_000.jpg          # ภาพต้นฉบับจากกล้อง (JPG/PNG)
│           ├── view_001.jpg
│           ├── ...
│           └── view_005.jpg
│
├── 02_processed/                     # [สร้างโดย CPU] Lossless PNG เสมอ!
│   └── backlight/
│       └── ceramic_mug_01/
│           ├── view_000.png          # RGB 8-bit, 1024x1024, แก้ EXIF แล้ว
│           ├── ...
│           └── view_005.png
│
├── 03_ground_truth/
│   └── masks/
│       └── backlight/
│           └── ceramic_mug_01/
│               ├── view_000.png      # Grayscale 1-Channel (0=พื้นหลัง, 255=วัตถุ)
│               └── ...
│
├── 04_masks/
│   ├── classical/                    # [สร้างโดย CPU] Binary Mask (0 / 255)
│   │   └── backlight/ceramic_mug_01/view_000.png
│   └── deep_learning/                # [สร้างโดย GPU] Binary Mask (0 / 255)
│       └── backlight/ceramic_mug_01/view_000.png
│
├── 05_segmented/                     # ภาพ 4-Channel RGBA (512x512, พื้นหลังโปร่งใส)
│   ├── from_raw/                     # ตัดพื้นหลังตรงจาก 01_raw (ใช้เทียบสัปดาห์ 8)
│   │   └── backlight/ceramic_mug_01/view_000.png
│   ├── classical/                    # จาก 02_processed + Mask ฝั่ง CPU
│   │   └── backlight/ceramic_mug_01/view_000.png
│   └── deep_learning/                # จาก 02_processed + Mask ฝั่ง GPU
│       └── backlight/ceramic_mug_01/view_000.png
│
├── 06_models_3d/                     # [สร้างโดย GPU] 1 โฟลเดอร์วัตถุ -> 1 โมเดล 3D
│   ├── from_raw/
│   │   └── backlight/ceramic_mug_01.glb
│   ├── from_classical/
│   │   └── backlight/ceramic_mug_01.glb
│   └── from_dl/
│       └── backlight/ceramic_mug_01.glb
│
└── 07_metadata/                      # บันทึกเวลาทำงาน (Latency) และผลวัด
    └── backlight_ceramic_mug_01_run.json
```

---

## 3. ข้อกำหนดทางเทคนิคของไฟล์ภาพ (Technical Format Contract)

| โฟลเดอร์ | นามสกุล | ช่องสี (Channels) | ขนาดมาตรฐาน | เหตุผลทางทฤษฎี DIP |
| :--- | :--- | :--- | :--- | :--- |
| `02_processed/` | `.png` (บังคับ) | 3-Channel RGB (8-bit) | `1024 × 1024` | ป้องกัน JPEG DCT Blocking/Ringing Artifacts ที่ทำลายขอบวัตถุก่อนตัดพื้นหลัง |
| `03_ground_truth/` & `04_masks/` | `.png` (บังคับ) | 1-Channel Grayscale (`0` หรือ `255` เท่านั้น) | เท่ากับ `02_processed` | คำนวณ Intersection over Union ($\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$) ได้ตรงพิกเซล |
| `05_segmented/` | `.png` (บังคับ) | 4-Channel **RGBA** (8-bit) | `512 × 512` (จัดกลาง ~80%) | โมเดล 3D เช่น InstantMesh ต้องการ Alpha channel ที่คมชัดและขนาดสม่ำเสมอทุกมุม |
| `06_models_3d/` | `.glb` (+ `.obj` เสริม) | Mesh + PBR Texture | Scale = 1.0m, Pivot ฐานล่าง | `.glb` ฝัง Texture ในไฟล์เดียว โหลดเข้า Unity (`GLTFast`) ได้ทันที |

---

## 4. รูปแบบไฟล์ Metadata & Latency Log (`data/07_metadata/`)

ทุกครั้งที่รัน Pipeline (ทั้งฝั่ง CPU และ GPU) ให้บันทึกไฟล์ JSON เพื่อนำไปสร้างตารางและกราฟในสัปดาห์ที่ 8, 12 และ 14:

```json
{
  "object_id": "ceramic_mug_01",
  "condition": "backlight",
  "num_views": 6,
  "cpu_preprocess": {
    "machine": "CPU",
    "config": "configs/preprocessing.yaml",
    "total_time_sec": 1.42,
    "avg_time_per_view_sec": 0.237,
    "sharpness_scores": [145.2, 162.8, 138.4, 151.0, 149.7, 155.3],
    "flagged_blurry_views": []
  },
  "segmentation": {
    "classical_cpu_time_sec": 3.85,
    "dl_gpu_time_sec": 1.12,
    "mean_iou_classical": 0.784,
    "mean_iou_dl": 0.956
  },
  "reconstruction_3d": {
    "machine": "GPU",
    "model": "instantmesh",
    "mode": "multi_view",
    "input_branch": "deep_learning",
    "inference_time_sec": 18.6,
    "vertices": 14250,
    "faces": 28400,
    "is_watertight": true
  }
}
```

---

## 5. วิธีซิงก์โฟลเดอร์ `data/` ข้าม 2 เครื่อง

1. สร้างโฟลเดอร์แชร์บน **Google Drive / OneDrive** ชื่อ `Itemize_Data` ที่มีโครงสร้างย่อย `01_raw` ถึง `07_metadata` เหมือนในโปรเจกต์
2. **ห้ามแก้ไฟล์ในโฟลเดอร์ของอีกเครื่องทับกัน:**
   - `[เครื่อง CPU - คนที่ 1]` เขียนเฉพาะ `02_processed/`, `04_masks/classical/`, `05_segmented/classical/` และอ่าน `04_masks/deep_learning/`, `06_models_3d/` มาคำนวณคะแนน
   - `[เครื่อง GPU - คนที่ 2]` อ่าน `01_raw/` กับ `02_processed/` และเขียนเฉพาะ `04_masks/deep_learning/`, `05_segmented/from_raw/`, `05_segmented/deep_learning/`, `06_models_3d/`
3. หากต้องการความสะดวกในสัปดาห์ที่ 12–14 สามารถเปิด `src/gpu/api_bridge.py` (FastAPI) เพื่อส่งชุดภาพจากเครื่อง CPU ไปยังเครื่อง GPU ผ่าน Wi-Fi วงเดียวกันโดยตรงได้

# Itemize — Multi-View Photo-to-3D Reconstruction & Game Integration

**งานรายวิชารอบนี้:** ถ่ายวัตถุหลายมุม → สร้างไฟล์ `.glb` ด้วย COLMAP → นำไปแก้รูปทรงใน Blender ดู[คู่มือเริ่มต้นบน Windows](docs/photogrammetry-quickstart.md) สำหรับแผนเดโมสองเดือนเศษ

รายละเอียดด้านล่างเป็นแผนตั้งต้นสำหรับเส้นทาง InstantMesh และ Unity

> **โปรเจกต์รายวิชา Digital Image Processing & Computer Vision (ระยะเวลา 15 สัปดาห์ | ทำคู่ 2 คน)**  
> **เป้าหมายหลัก:** ถ่ายภาพวัตถุจริง **หลายมุมมอง (Multi-View)** → ปรับปรุงคุณภาพภาพและปรับแสง/สีให้สม่ำเสมอข้ามทุกมุม (Classical DIP) → ตัดพื้นหลัง (เทียบวิธีพื้นฐาน vs Deep Learning) → สร้างโมเดล 3 มิติ (`.glb` / `.obj`) → *(เป้าหมายเสริม)* นำเข้าสู่เกม Unity อัตโนมัติด้วย `GLTFast`

---

## 👋 สำหรับคู่โปรเจกต์ (อ่านตรงนี้ก่อนเริ่มงาน!)

โปรเจกต์นี้ออกแบบให้ **แยกงานและแยกโค้ดตามสเปกเครื่องอย่างเด็ดขาด** เพื่อไม่ให้เกิดปัญหา Dependency ชนกัน และลด Git Merge Conflict ระหว่างที่เราทำงานคู่ขนานกัน:

| บทบาท | สเปกเครื่อง | โฟลเดอร์โค้ดที่ดูแล | ความรับผิดชอบหลัก |
| :--- | :--- | :--- | :--- |
| **คนที่ 1 (MimiPiyaphat)** | **CPU เท่านั้น** (ไม่มีการ์ดจอแยก) | `src/cpu/`, `evaluation/` (CPU metrics) | • Classical DIP & Multi-View Pre-processing (ปรับแสง, White Balance ข้ามมุม, ลด Noise, คัดภาพเบลอ, ดัดมุม)<br>• Classical Segmentation (HSV, Canny+Contours, GrabCut, Morphology)<br>• วัดผลเชิงตัวเลข (IoU, Dice, Sharpness, Latency) และทำกราฟเปรียบเทียบ |
| **คนที่ 2 (คู่โปรเจกต์)** | **มี NVIDIA GPU (CUDA)** | `src/gpu/`, `third_party/`, `unity/` | • Deep Learning Segmentation (`rembg` / `SAM 2` แบบ Multi-View)<br>• Multi-View 3D Reconstruction (`InstantMesh` และเทียบ Baseline เช่น `TripoSR`)<br>• จัดการไฟล์โมเดล 3D (`.glb` / `.obj`) และเชื่อมต่อ Unity (`GLTFast`) |
| **ทำร่วมกัน** | ทั้ง 2 เครื่อง | `src/common/`, `configs/`, `docs/`, `experiments/` | • ถ่ายชุดภาพทดสอบ 5 สภาพแสง (Multi-View)<br>• วาด Ground Truth Mask สำหรับวัด IoU<br>• สรุปผลการทดลองสัปดาห์ที่ 8, 12, 14 และเตรียมสอบอธิบายรายบุคคลสัปดาห์ที่ 15 |

---

## 🔄 ภาพรวมการทำงานแบบ Multi-View ระหว่าง 2 เครื่อง

```text
[ถ่ายภาพวัตถุ 6–8 มุมรอบตัว]
        │
        ▼  data/01_raw/<condition>/<object_id>/view_000.jpg ... view_005.jpg
┌──────────────────────────────────────────────────────────┐
│ [เครื่อง CPU - คนที่ 1] src/cpu/                          │
│ 1. คัดกรองภาพเบลอ (Laplacian Variance)                     │
│ 2. ปรับแสง/Gamma + CLAHE + White Balance ให้สีตรงกันทุกมุม  │
│ 3. ลด Noise แบบรักษาขอบ (Bilateral Filter) & จัดขนาดภาพ     │
│    ──► data/02_processed/<condition>/<object_id>/*.png   │
│ 4. ตัดพื้นหลังแบบพื้นฐาน (GrabCut / HSV / Contours)         │
│    ──► data/04_masks/classical/ & data/05_segmented/     │
└──────────────────────────┬───────────────────────────────┘
                           │ (ซิงก์โฟลเดอร์ data/ ผ่าน Google Drive / OneDrive / LAN)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [เครื่อง GPU - คนที่ 2] src/gpu/                          │
│ 1. ตัดพื้นหลังแบบ DL (rembg / SAM 2 ข้ามมุม)                │
│    ──► data/04_masks/deep_learning/ & data/05_segmented/ │
│ 2. สร้างโมเดล 3D จากชุดภาพหลายมุม (InstantMesh / Multi-View)│
│    ──► data/06_models_3d/<pipeline>/<object_id>.glb      │
│ 3. (เป้าหมายเสริม) โหลด .glb เข้าเกม Unity ด้วย GLTFast     │
└──────────────────────────┬───────────────────────────────┘
                           │ (ส่งไฟล์ .glb + Masks + Latency JSON กลับมาวัดผล)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [วัดผลร่วมกัน] evaluation/                                │
│ • เปรียบเทียบ Raw vs Pre-processed (สัปดาห์ 8)             │
│ • เปรียบเทียบ Classical CPU vs DL GPU: IoU & เวลา (สัปดาห์ 9-13)│
│ • ทดสอบ 5 สภาพแวดล้อมจริง + Failure Cases (สัปดาห์ 14)     │
└──────────────────────────────────────────────────────────┘
```

> 📖 **สำคัญมาก:** อ่านกติกาการถ่ายภาพ การตั้งชื่อไฟล์ และโครงสร้างโฟลเดอร์ `data/` แบบละเอียดที่ [docs/data_protocol.md](docs/data_protocol.md)

---

## 📂 โครงสร้างโปรเจกต์

```text
Itemize/
├── README.md
├── .gitignore
├── requirements/
│   ├── cpu.txt                     # [เครื่อง CPU] OpenCV, NumPy, SciPy, scikit-image, trimesh (ไม่มี CUDA)
│   └── gpu.txt                     # [เครื่อง GPU] rembg[gpu], onnxruntime-gpu, trimesh, pygltflib, FastAPI
├── configs/
│   ├── preprocessing.yaml          # พารามิเตอร์ DIP Pre-processing (ปรับได้โดยไม่แก้โค้ด)
│   ├── segmentation.yaml           # พารามิเตอร์ Classical Seg (CPU) และ DL Seg (GPU)
│   └── reconstruction.yaml         # พารามิเตอร์ Multi-View 3D (InstantMesh / TripoSR / Unity export)
├── src/
│   ├── common/                     # โค้ดกลางที่ใช้ร่วมกัน (ห้าม import torch)
│   ├── cpu/                        # [เครื่อง CPU - คนที่ 1]
│   │   ├── preprocessing/          #   intensity, histogram, white_balance, filtering, restoration, geometry
│   │   └── segmentation/           #   hsv_threshold, canny_contour, grabcut, morphology
│   └── gpu/                        # [เครื่อง GPU - คนที่ 2]
│       ├── segmentation/           #   rembg_segment, sam2_segment (Multi-View)
│       └── reconstruction/         #   instantmesh_runner, triposr_runner (baseline), export_model
├── data/                           # โฟลเดอร์ข้อมูลกลาง (ไม่ commit รูป/โมเดลหนักๆ ขึ้น Git)
│   ├── 01_raw/                     #   แยกตามสภาพแสง: normal, low_light, backlight, reflective, similar_bg
│   ├── 02_processed/               #   ภาพหลัง Pre-process (.png Lossless)
│   ├── 03_ground_truth/masks/      #   Mask เฉลยสำหรับวัด IoU
│   ├── 04_masks/                   #   classical/ vs deep_learning/
│   ├── 05_segmented/               #   ภาพ 4-Channel RGBA: from_raw/, classical/, deep_learning/
│   ├── 06_models_3d/               #   ไฟล์ .glb/.obj: from_raw/, from_classical/, from_dl/
│   └── 07_metadata/                #   ไฟล์ JSON/CSV บันทึกเวลา (Latency) และค่า Config
├── evaluation/                     # สคริปต์คำนวณ IoU, Image Metrics, Mesh Metrics, กราฟ
├── experiments/                    # สคริปต์รันการทดลองและ Ablation Studies
├── results/                        # ผลลัพธ์สำหรับใส่รายงาน (images, tables, graphs, logs)
├── tests/                          # Unit tests (pytest)
├── third_party/                    # สำหรับ git clone InstantMesh / TripoSR / SAM 2 (ไม่ commit ขึ้น Git)
├── unity/                          # [เป้าหมายเสริม] โปรเจกต์ Unity + GLTFast
└── docs/
    └── data_protocol.md            # ข้อตกลงมาตรฐานข้อมูล Multi-View ระหว่าง 2 เครื่อง
```

---

## 🛠️ คู่มือติดตั้งสภาพแวดล้อม (Setup Guide)

### 1. Clone โปรเจกต์ (ทำทั้ง 2 เครื่อง)

```powershell
git clone https://github.com/MimiPiyaphat/Itemize.git
cd Itemize
```

---

### 2A. สำหรับ `[เครื่อง GPU - คนที่ 2]` (คู่โปรเจกต์)

> ⚠️ **ข้อควรระวังเรื่องเวอร์ชัน Python:** แนะนำให้ใช้ **Python 3.10 หรือ 3.11** (หลีกเลี่ยง Python 3.12 บนเครื่อง GPU เพราะส่วนเสริม C++/CUDA ของโมเดล 3D เช่น `torchmcubes` / `nvdiffrast` มักคอมไพล์ไม่ผ่านบน 3.12)

```powershell
# 1. เช็กการ์ดจอ, VRAM (แนะนำ >= 8GB) และเวอร์ชัน CUDA
nvidia-smi

# 2. สร้าง Virtual Environment ด้วย Python 3.10 หรือ 3.11
py -3.10 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. ติดตั้ง PyTorch รุ่นที่ตรงกับ CUDA ของเครื่อง (ตัวอย่าง CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. ติดตั้งแพ็กเกจพื้นฐานฝั่ง GPU
pip install -r requirements/gpu.txt

# 5. ทดสอบความพร้อม GPU & CUDA
python -c "import torch, cv2, rembg, trimesh; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

*(หมายเหตุ: โมเดลขนาดใหญ่อย่าง **InstantMesh**, **SAM 2**, หรือ **TripoSR** ค่อย `git clone` ลงในโฟลเดอร์ `third_party/` ช่วงสัปดาห์ที่ 6–9 ตามแผนงาน)*

---

### 2B. สำหรับ `[เครื่อง CPU - คนที่ 1]` (MimiPiyaphat)

```powershell
# 1. สร้าง Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. ติดตั้งแพ็กเกจ Classical DIP & Metrics (ไม่มี CUDA)
pip install -r requirements/cpu.txt

# 3. ทดสอบความพร้อม CPU DIP
python -c "import cv2, numpy, scipy, skimage, PIL, matplotlib, pandas, yaml, trimesh; print('CPU DIP Env OK | OpenCV:', cv2.__version__, '| NumPy:', numpy.__version__)"
```

---

## 📅 แผนงานและจุดตรวจสำคัญ 15 สัปดาห์ (Syllabus Milestones)

| ช่วงสัปดาห์ | เป้าหมายหลัก | `[เครื่อง CPU - คนที่ 1]` | `[เครื่อง GPU - คนที่ 2]` |
| :--- | :--- | :--- | :--- |
| **สัปดาห์ 1–2** | เตรียมโครงสร้าง & เก็บข้อมูล | ตั้ง Repo, กติกา Multi-View, ถ่ายภาพทดสอบ | ตั้ง PyTorch+CUDA, ทดสอบ `rembg` |
| **สัปดาห์ 3–7** | **Classical DIP & Multi-View Pre-processing** | Intensity/Gamma, Histogram/CLAHE, White Balance ข้ามมุม, Bilateral/Median Denoise, Blur Screening, Geometry | เตรียม `InstantMesh` (Multi-View) + `TripoSR` (Baseline) ใน `third_party/` และทดสอบรัน End-to-End |
| **สัปดาห์ 8** | **สอบกลางภาค (Midterm Ablation)** | ส่งชุดภาพ `01_raw` vs `02_processed` พร้อมกราฟวัดคุณภาพภาพ (Sharpness, Histogram, Color consistency) | สร้างโมเดล 3D เทียบ **"ภาพดิบหลายมุม"** vs **"ภาพผ่าน Pre-processing หลายมุม"** |
| **สัปดาห์ 9–13** | **Segmentation Benchmark** | พัฒนา HSV, Canny+Contours, GrabCut, Morphology + คำนวณ **Mean IoU / เวลา** เทียบทุกวิธี | รัน `rembg` / `SAM 2` (Multi-View propagation) + บันทึก Mask & เวลา Inference |
| **สัปดาห์ 14** | **ประเมินระบบจริง (Real-World Eval)** | วิเคราะห์ 5 สภาพแสง (`normal`, `low_light`, `backlight`, `reflective`, `similar_bg`), วัด Latency รวม และสรุป Failure Cases | รัน Pipeline จริงครบทุกเงื่อนไข + *(เสริม)* ทดสอบโหลด `.glb` เข้า Unity (`GLTFast`) |
| **สัปดาห์ 15** | **นำเสนอ & สอบอธิบายรายบุคคล** | ทั้ง 2 คนต้องอธิบายทฤษฎี DIP, คณิตศาสตร์เบื้องหลัง, โครงสร้าง Multi-View 3D และผลวัดเชิงตัวเลขได้ครบทุกส่วน |

---

## 🤝 กติกาการใช้ Git ร่วมกัน (ป้องกันโค้ดชนกัน)

1. **Pull ก่อนเริ่มทำงานทุกครั้ง:** `git pull origin main`
2. **แยก Branch ตามงาน:**
   - คนที่ 1 (CPU): `feat/cpu-week03-intensity`, `feat/cpu-grabcut`
   - คนที่ 2 (GPU): `feat/gpu-rembg`, `feat/gpu-instantmesh`
3. **ห้าม Commit ไฟล์หนักขึ้น GitHub:** โฟลเดอร์ `data/`, `results/`, `third_party/`, `.venv/`, และไฟล์ `.glb`/`.pth`/`.ckpt` ถูกกันไว้ใน `.gitignore` แล้ว ให้ซิงก์โฟลเดอร์ `data/` ผ่าน **Google Drive / OneDrive ร่วมกัน** แทน
4. **เปลี่ยนค่าพารามิเตอร์ที่ `configs/*.yaml`:** หากปรับค่าทดลองให้บันทึกไว้ใน YAML หรือ `data/07_metadata/` เพื่อให้ทั้ง 2 เครื่องทำซ้ำผลได้ตรงกัน

# nnU-Net 2D 影像分割使用指南

本文件說明如何使用 nnU-Net 進行 2D 影像分割實驗。

## 目錄

1. [簡介](#簡介)
2. [系統需求](#系統需求)
3. [安裝步驟](#安裝步驟)
4. [環境變數設定](#環境變數設定)
5. [準備 2D 數據集](#準備-2d-數據集)
6. [數據集格式](#數據集格式)
7. [實驗規劃與預處理](#實驗規劃與預處理)
8. [模型訓練](#模型訓練)
9. [推論（預測）](#推論預測)
10. [最佳配置選擇](#最佳配置選擇)
11. [完整範例流程](#完整範例流程)

---

## 簡介

nnU-Net 是一個自動適應數據集的語義分割方法，能夠自動分析提供的訓練案例並配置相應的 U-Net 分割流程。nnU-Net V2 原生支援 2D 影像，無需將 2D 影像轉換為偽 3D 格式。

### 2D 影像分割的特點

- **原生支援**：nnU-Net V2 直接支援 2D 影像格式（如 .png, .bmp, .tif）
- **自動配置**：系統會自動分析 2D 數據集特性並配置最佳的分割流程
- **多種格式**：支援多種 2D 影像格式，無需轉換為 .nii.gz

---

## 系統需求

### 作業系統
- Linux (Ubuntu 18.04, 20.04, 22.04)
- Windows
- macOS

### 硬體需求

#### 訓練
- **GPU**：建議至少 10 GB VRAM（如 RTX 2080ti, RTX 3080/3090, RTX 4080/4090）
- **CPU**：至少 6 核心（12 執行緒）
- **RAM**：建議 64 GB
- **儲存**：SSD（M.2 PCIe Gen 3 或更高）

#### 推論
- **GPU**：至少 4 GB VRAM（建議）
- 也可使用 CPU 或 Apple M1/M2（MPS），但速度較慢

### 軟體需求
- Python 3.9 或更新版本
- PyTorch（需先安裝，支援 CUDA、MPS 或 CPU）

---

## 安裝步驟

### 1. 安裝 PyTorch

首先，根據您的硬體安裝 PyTorch：

```bash
# 訪問 https://pytorch.org/get-started/locally/ 獲取適合您系統的安裝命令
# 例如，CUDA 版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安裝 nnU-Net

#### 方式一：標準安裝（作為標準化基線或推論使用）

```bash
pip install nnunetv2
```

#### 方式二：開發模式安裝（可修改程式碼）

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

### 3. 驗證安裝

安裝完成後，您應該可以使用以下命令：

```bash
nnUNetv2_plan_and_preprocess -h
```

如果命令可以執行，表示安裝成功。

---

## 環境變數設定

nnU-Net 需要三個環境變數來指定數據和模型儲存位置：

### Windows (PowerShell)

```powershell
# 設定環境變數（臨時，僅當前會話有效）
$env:nnUNet_raw = "F:\nnUNet_raw"
$env:nnUNet_preprocessed = "F:\nnUNet_preprocessed"
$env:nnUNet_results = "F:\nnUNet_results"

# 永久設定（需要管理員權限）
[System.Environment]::SetEnvironmentVariable("nnUNet_raw", "F:\nnUNet_raw", "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_preprocessed", "F:\nnUNet_preprocessed", "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_results", "F:\nnUNet_results", "User")
```

### Linux/macOS

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加：
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# 然後執行：
source ~/.bashrc  # 或 source ~/.zshrc
```

### 環境變數說明

1. **nnUNet_raw**：存放原始數據集的位置
2. **nnUNet_preprocessed**：存放預處理後數據的位置（建議使用 SSD）
3. **nnUNet_results**：存放訓練好的模型權重的位置

---

## 準備 2D 數據集

### 數據集命名規則

數據集必須命名為 `DatasetXXX_Name` 格式，其中：
- `XXX` 是三位數的數據集 ID（如 001, 002, 120）
- `Name` 是數據集名稱（可自由選擇）

例如：`Dataset120_RoadSegmentation`

### 數據集目錄結構

```
nnUNet_raw/
└── Dataset120_My2DData/
    ├── dataset.json
    ├── imagesTr/          # 訓練影像
    │   ├── case001_0000.png
    │   ├── case002_0000.png
    │   └── ...
    ├── imagesTs/          # 測試影像（可選）
    │   ├── test001_0000.png
    │   └── ...
    └── labelsTr/          # 訓練標籤（分割遮罩）
        ├── case001.png
        ├── case002.png
        └── ...
```

---

## 數據集格式

### 影像檔案命名規則

對於 2D 影像，檔案命名格式為：
```
{CASE_IDENTIFIER}_{CHANNEL_ID}.{FILE_ENDING}
```

- `CASE_IDENTIFIER`：案例的唯一識別碼（如 `case001`）
- `CHANNEL_ID`：通道識別碼，四位數（如 `0000`, `0001`）
  - 對於單通道影像（灰階），使用 `0000`
  - 對於 RGB 影像，可以使用單一檔案（三個通道合併）或三個檔案（分別為 `0000`, `0001`, `0002`）
- `FILE_ENDING`：檔案副檔名（如 `.png`, `.bmp`, `.tif`）

### 標籤檔案命名規則

標籤檔案命名格式為：
```
{CASE_IDENTIFIER}.{FILE_ENDING}
```

標籤必須是整數遮罩：
- 背景必須為 `0`
- 類別標籤必須是連續的整數（0, 1, 2, 3, ...）
- 標籤檔案必須與對應的影像檔案使用相同的檔案格式

### 支援的檔案格式

nnU-Net V2 支援多種 2D 影像格式：
- **NaturalImage2DIO**：`.png`, `.bmp`, `.tif`
- **NibabelIO**：`.nii.gz`, `.nrrd`, `.mha`（也可用於 2D）
- **SimpleITKIO**：`.nii.gz`, `.nrrd`, `.mha`

**重要**：影像和標籤必須使用相同的檔案格式，且必須使用無損壓縮格式（不可使用 `.jpg`）。

### dataset.json 配置

每個數據集都需要一個 `dataset.json` 檔案，包含以下資訊：

```json
{
    "channel_names": {
        "0": "R",
        "1": "G",
        "2": "B"
    },
    "labels": {
        "background": 0,
        "class1": 1,
        "class2": 2
    },
    "numTraining": 100,
    "file_ending": ".png"
}
```

#### 欄位說明

- **channel_names**：通道名稱對應
  - 對於 RGB 影像：`{"0": "R", "1": "G", "2": "B"}`
  - 對於灰階影像：`{"0": "grayscale"}`
  - 對於 CT 影像：`{"0": "CT"}`（會觸發特殊的標準化方式）
- **labels**：標籤名稱與整數值的對應
  - 背景必須為 `0`
  - 其他類別使用連續整數
- **numTraining**：訓練案例數量
- **file_ending**：檔案副檔名（如 `.png`, `.bmp`）

#### 自動生成 dataset.json

您可以使用 nnU-Net 提供的工具自動生成 `dataset.json`：

```python
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

generate_dataset_json(
    output_folder="/path/to/Dataset120_My2DData",
    channel_names={0: "R", 1: "G", 2: "B"},  # 或 {0: "grayscale"} 對於灰階
    labels={"background": 0, "class1": 1, "class2": 2},
    num_training_cases=100,
    file_ending=".png",
    dataset_name="Dataset120_My2DData"
)
```

### 2D 數據集範例

以下是一個完整的 2D 數據集範例（道路分割）：

```
nnUNet_raw/Dataset120_RoadSegmentation/
├── dataset.json
├── imagesTr/
│   ├── image001_0000.png
│   ├── image002_0000.png
│   └── ...
└── labelsTr/
    ├── image001.png
    ├── image002.png
    └── ...
```

對應的 `dataset.json`：

```json
{
    "channel_names": {
        "0": "R",
        "1": "G",
        "2": "B"
    },
    "labels": {
        "background": 0,
        "road": 1
    },
    "numTraining": 1108,
    "file_ending": ".png"
}
```

---

## 實驗規劃與預處理

在開始訓練之前，nnU-Net 需要分析數據集並進行預處理。

### 執行規劃與預處理

使用以下命令進行實驗規劃和預處理：

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

其中 `DATASET_ID` 是數據集的 ID（如 `120`）或完整名稱（如 `Dataset120_RoadSegmentation`）。

#### 參數說明

- `-d DATASET_ID`：指定數據集 ID 或名稱
- `--verify_dataset_integrity`：驗證數據集完整性（建議首次執行時使用）
- `-c CONFIGURATION`：指定配置（可選，如 `2d`）
- `-np NUM_PROCESSES`：指定預處理使用的進程數（可選）

#### 範例

```bash
# 處理數據集 120
nnUNetv2_plan_and_preprocess -d 120 --verify_dataset_integrity

# 或使用完整名稱
nnUNetv2_plan_and_preprocess -d Dataset120_RoadSegmentation --verify_dataset_integrity
```

### 預處理輸出

執行完成後，會在 `nnUNet_preprocessed` 目錄下創建對應的資料夾：

```
nnUNet_preprocessed/
└── Dataset120_RoadSegmentation/
    ├── dataset_fingerprint.json
    ├── nnUNetPlans.json
    └── nnUNetPlans__2d/
        └── ...（預處理後的數據）
```

### 分步驟執行（可選）

如果您想分步驟執行，可以使用：

```bash
# 1. 提取數據集指紋
nnUNetv2_extract_fingerprint -d DATASET_ID

# 2. 規劃實驗
nnUNetv2_plan_experiment -d DATASET_ID

# 3. 預處理
nnUNetv2_preprocess -d DATASET_ID -c 2d
```

---

## 模型訓練

nnU-Net 會為 2D 數據集創建 `2d` 配置。訓練採用 5 折交叉驗證。

### 訓練命令格式

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options]
```

### 2D U-Net 訓練

對於 2D 影像分割，使用 `2d` 配置：

```bash
# 訓練第 0 折
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz

# 訓練第 1 折
nnUNetv2_train DATASET_NAME_OR_ID 2d 1 --npz

# 訓練第 2 折
nnUNetv2_train DATASET_NAME_OR_ID 2d 2 --npz

# 訓練第 3 折
nnUNetv2_train DATASET_NAME_OR_ID 2d 3 --npz

# 訓練第 4 折
nnUNetv2_train DATASET_NAME_OR_ID 2d 4 --npz
```

#### 參數說明

- `DATASET_NAME_OR_ID`：數據集名稱或 ID
- `2d`：2D U-Net 配置
- `FOLD`：交叉驗證折數（0-4）
- `--npz`：儲存 softmax 輸出（用於後續的最佳配置選擇和集成）
- `--c`：繼續之前的訓練（如果訓練中斷）
- `-device DEVICE`：指定設備（`cuda`, `cpu`, `mps`）

#### 範例

```bash
# 使用數據集 ID
nnUNetv2_train 120 2d 0 --npz

# 使用數據集名稱
nnUNetv2_train Dataset120_RoadSegmentation 2d 0 --npz

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 120 2d 0 --npz
```

### 訓練所有折數（批次執行）

#### Windows (PowerShell)

```powershell
# 訓練所有 5 折（假設有 5 個 GPU）
$env:CUDA_VISIBLE_DEVICES=0; Start-Process -NoNewWindow python -ArgumentList "-m", "nnunetv2.run.run_training", "120", "2d", "0", "--npz"
$env:CUDA_VISIBLE_DEVICES=1; Start-Process -NoNewWindow python -ArgumentList "-m", "nnunetv2.run.run_training", "120", "2d", "1", "--npz"
$env:CUDA_VISIBLE_DEVICES=2; Start-Process -NoNewWindow python -ArgumentList "-m", "nnunetv2.run.run_training", "120", "2d", "2", "--npz"
$env:CUDA_VISIBLE_DEVICES=3; Start-Process -NoNewWindow python -ArgumentList "-m", "nnunetv2.run.run_training", "120", "2d", "3", "--npz"
$env:CUDA_VISIBLE_DEVICES=4; Start-Process -NoNewWindow python -ArgumentList "-m", "nnunetv2.run.run_training", "120", "2d", "4", "--npz"
```

#### Linux/macOS

```bash
# 訓練所有 5 折（並行執行）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 120 2d 0 --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 120 2d 1 --npz &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 120 2d 2 --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 120 2d 3 --npz &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 120 2d 4 --npz &
wait
```

**重要**：首次訓練時，nnU-Net 會將預處理數據解壓縮為未壓縮的 numpy 陣列。請等待第一個訓練開始使用 GPU 後，再啟動其他折數的訓練。

### 訓練輸出

訓練完成後，模型會儲存在 `nnUNet_results` 目錄：

```
nnUNet_results/
└── Dataset120_RoadSegmentation/
    └── nnUNetTrainer__nnUNetPlans__2d/
        ├── fold_0/
        │   ├── checkpoint_final.pth
        │   ├── checkpoint_best.pth
        │   ├── progress.png
        │   └── validation/
        ├── fold_1/
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        ├── dataset.json
        ├── dataset_fingerprint.json
        └── plans.json
```

### 監控訓練進度

訓練過程中，可以查看 `progress.png` 檔案來監控訓練進度：
- 訓練損失（藍色）
- 驗證損失（紅色）
- Dice 分數近似值（綠色）

---

## 推論（預測）

訓練完成後，可以使用訓練好的模型進行推論。

### 準備推論數據

推論數據必須遵循與訓練數據相同的命名規則和格式：

```
input_folder/
├── test001_0000.png
├── test002_0000.png
└── ...
```

### 執行推論

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION [options]
```

#### 參數說明

- `-i INPUT_FOLDER`：輸入資料夾（包含要預測的影像）
- `-o OUTPUT_FOLDER`：輸出資料夾（預測結果將儲存在此）
- `-d DATASET_NAME_OR_ID`：數據集名稱或 ID
- `-c CONFIGURATION`：使用的配置（如 `2d`）
- `--save_probabilities`：儲存機率圖（用於集成）
- `-f FOLD`：指定使用的折數（預設使用所有 5 折的集成）

#### 範例

```bash
# 基本推論
nnUNetv2_predict -i ./test_images -o ./predictions -d 120 -c 2d

# 儲存機率圖（用於集成）
nnUNetv2_predict -i ./test_images -o ./predictions -d 120 -c 2d --save_probabilities

# 使用單一折數
nnUNetv2_predict -i ./test_images -o ./predictions -d 120 -c 2d -f 0
```

### 推論輸出

推論結果會儲存在指定的輸出資料夾中：

```
output_folder/
├── test001.png          # 預測的分割遮罩
├── test002.png
└── ...
```

如果使用 `--save_probabilities`，還會生成 `.npz` 檔案（包含 softmax 機率）。

---

## 最佳配置選擇

訓練完成所有折數後，可以讓 nnU-Net 自動選擇最佳配置：

```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS
```

### 範例

```bash
# 對於 2D 數據集
nnUNetv2_find_best_configuration 120 -c 2d
```

### 輸出

此命令會：
1. 評估所有配置的性能
2. 確定最佳後處理策略
3. 生成 `inference_instructions.txt` 檔案，包含推論所需的確切命令
4. 生成 `inference_information.json` 檔案，包含所有配置的性能資訊

---

## 完整範例流程

以下是一個完整的 2D 影像分割實驗流程：

### 步驟 1：準備數據集

```bash
# 1. 創建數據集目錄
mkdir -p F:\nnUNet_raw\Dataset120_My2DData\imagesTr
mkdir -p F:\nnUNet_raw\Dataset120_My2DData\labelsTr

# 2. 將影像和標籤複製到對應目錄
# 影像命名：case001_0000.png, case002_0000.png, ...
# 標籤命名：case001.png, case002.png, ...

# 3. 創建 dataset.json（或使用 Python 腳本生成）
```

### 步驟 2：設定環境變數

```powershell
$env:nnUNet_raw = "F:\nnUNet_raw"
$env:nnUNet_preprocessed = "F:\nnUNet_preprocessed"
$env:nnUNet_results = "F:\nnUNet_results"
```

### 步驟 3：實驗規劃與預處理

```bash
nnUNetv2_plan_and_preprocess -d 120 --verify_dataset_integrity
```

### 步驟 4：訓練模型

```bash
# 訓練所有 5 折
nnUNetv2_train 120 2d 0 --npz
nnUNetv2_train 120 2d 1 --npz
nnUNetv2_train 120 2d 2 --npz
nnUNetv2_train 120 2d 3 --npz
nnUNetv2_train 120 2d 4 --npz
```

### 步驟 5：選擇最佳配置

```bash
nnUNetv2_find_best_configuration 120 -c 2d
```

### 步驟 6：執行推論

```bash
nnUNetv2_predict -i ./test_images -o ./predictions -d 120 -c 2d
```

---

## 常見問題

### Q1：如何處理 RGB 影像？

對於 RGB 影像，您有兩種選擇：

1. **單一檔案**：將 RGB 三個通道合併在一個 `.png` 檔案中，使用通道識別碼 `0000`
2. **三個檔案**：分別儲存 R、G、B 通道為三個檔案，使用 `0000`、`0001`、`0002`

在 `dataset.json` 中設定：
```json
"channel_names": {
    "0": "R",
    "1": "G",
    "2": "B"
}
```

### Q2：訓練需要多長時間？

訓練時間取決於：
- 影像大小
- 數據集大小
- GPU 性能
- 網絡配置

一般來說，在 RTX 3090 上訓練一個 2D 配置的所有 5 折可能需要數小時到數天。

### Q3：如何繼續中斷的訓練？

使用 `--c` 參數：
```bash
nnUNetv2_train 120 2d 0 --c --npz
```

### Q4：如何評估預測結果？

使用評估命令：
```bash
nnUNetv2_evaluate_folder -ref REFERENCE_FOLDER -pred PREDICTION_FOLDER
```

### Q5：可以使用 CPU 訓練嗎？

可以，但速度會非常慢。建議使用 GPU：
```bash
nnUNetv2_train 120 2d 0 --npz -device cpu
```

---

## 參考資源

- [nnU-Net 官方文檔](https://github.com/MIC-DKFZ/nnUNet)
- [數據集格式詳細說明](documentation/dataset_format.md)
- [安裝說明](documentation/installation_instructions.md)
- [使用指南](documentation/how_to_use_nnunet.md)

---

## 總結

本指南涵蓋了使用 nnU-Net 進行 2D 影像分割的完整流程：

1. ✅ 安裝 nnU-Net
2. ✅ 設定環境變數
3. ✅ 準備 2D 數據集
4. ✅ 執行實驗規劃與預處理
5. ✅ 訓練 2D U-Net 模型
6. ✅ 執行推論
7. ✅ 選擇最佳配置

遵循這些步驟，您就可以開始使用 nnU-Net 進行 2D 影像分割實驗了！

如有問題，請參考官方文檔或開源專案的 Issues 頁面。

Other Links.

- https://github.com/IML-DKFZ/nnunet-workshop

- https://github.com/kancheng/nnUNet
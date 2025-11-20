# ISIC2018 Demo 訓練指引

本文件說明如何使用 `isic2018` 目錄中的 dermoscopy 影像，在 nnU-Net v2 中快速完成一輪 Demo 訓練與推論。內容依序涵蓋環境設定、資料準備、預處理、訓練、推論與驗證。

---

## 1. 環境準備

1. 建議使用 Conda 建立虛擬環境（Python 3.9 以上）：
   ```
   conda create -n nnunet python=3.10
   conda activate nnunet
   ```
2. 安裝 nnU-Net v2 依賴：
   ```
   pip install -e .
   ```
3. 依 `documentation/setting_up_paths.md` 設定三個必要環境變數（可寫入 `.bashrc` 或 PowerShell profile）：
   - `nnUNet_raw`：原始資料集儲存位置（例：`F:\nnunet_raw`）
   - `nnUNet_preprocessed`：預處理輸出（例：`F:\nnunet_preprocessed`）
   - `nnUNet_results`：訓練結果與權重（例：`F:\nnunet_results`）
4. 確保 GPU Driver/CUDA 對應 PyTorch 版本；若僅 Demo 可使用 CPU 但訓練時間會較長。

```
   $env:nnUNet_raw = "F:/newproj/nnUNet/nnunet_raw"
   $env:nnUNet_preprocessed = "F:/newproj/nnUNet/nnunet_preprocessed"
   $env:nnUNet_results = "F:/newproj/nnUNet/nnunet_results"
```

---

## 2. 資料集重組

現有資料結構：
```
isic2018/
├── train
│   ├── images
│   └── masks
└── val
    ├── images
    └── masks
```

nnU-Net v2 需要官方定義的 Dataset 格式（詳見 `documentation/dataset_format.md`）。以下以 `Dataset201_ISIC2018` 為例：

1. 建立目錄：
   ```
   %nnUNet_raw%/Dataset201_ISIC2018/
     ├── imagesTr
     ├── labelsTr
     └── imagesTs
   ```
2. 將 `train/images` 逐一複製到 `imagesTr`，並依照單通道命名規則 `ISIC_XXXX_0000.png`。若原檔多通道，可拆成對應 `_0000`, `_0001`...
3. 將 `train/masks` 複製到 `labelsTr`，檔名與影像一致但不用 `_0000` 後綴，例如 `ISIC_1234.png`。
4. 將 `val/images` 複製到 `imagesTs`。若有標註，可同步放入 `labelsVer` 便於驗證，但對 Demo 可不強制。
5. 在 `Dataset201_ISIC2018` 內建立 `dataset.json`（精簡示例）：
   ```json
   {
     "name": "ISIC2018",
     "description": "Skin lesion segmentation demo",
     "reference": "https://challenge2018.isic-archive.com/",
     "licence": "CC-BY",
     "release": "1.0",
     "channel_names": { "0": "Dermoscopic" },
     "labels": { "background": 0, "lesion": 1 },
     "numTraining": <train_count>,
     "numTest": <val_count>,
     "file_ending": ".png",
     "training": [
       { "image": "./imagesTr/ISIC_0000_0000.png", "label": "./labelsTr/ISIC_0000.png" }
     ],
     "test": [
       "./imagesTs/ISIC_val_0000_0000.png"
     ]
   }
   ```
   - `numTraining`、`numTest` 對應實際檔數。
   - 填寫 `training`/`test` 陣列時可用腳本自動生成，確保相對路徑正確。

---

## 3. Dataset Fingerprinting 與預處理

執行：
```
nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity
```
- `-d 201`：對應 `dataset.json` 指定的資料集 ID。
- `--verify_dataset_integrity`：檢查影像、標籤對應並回報統計資訊。
- 完成後，預處理結果位於 `%nnUNet_preprocessed%/Dataset201_ISIC2018/`。
- 若硬體受限，可新增 `-np 4` 限制 CPU 使用、或以 `--no_pp` 跳過預處理（僅調試用）。

---

## 4. Demo 訓練

若 GPU 記憶體足夠，可採 `3d_fullres`；若資源有限可改 `2d`。以下示範輕量訓練（50 epochs）：
```
nnUNetv2_train 201 3d_fullres 0 -tr nnUNetTrainer_50epochs -p nnUNetPlans
```
參數說明：
- `201`：資料集 ID。
- `3d_fullres`：配置（可替換為 `2d` 或 `3d_lowres`）。
- `0`：fold index，Demo 可僅訓練 fold 0。
- `-tr nnUNetTrainer_50epochs`：nnU-Net 附的輕量訓練器（位於 `nnunetv2/training/nnUNetTrainer/`）。
- `-p nnUNetPlans`：使用預設 plans；如需自訂，可在 `nnunetv2/plans` 新增。

輸出路徑：
```
%nnUNet_results%/
└── Dataset201_ISIC2018/
    └── nnUNetPlans/
        └── 3d_fullres/
            └── fold_0/
                ├── checkpoint_final.pth
                ├── training_log.json
                └── progress.png
```

---

## 5. 推論與驗證

### 5.1 推論指令
```
nnUNetv2_predict ^
  -i %nnUNet_raw%/Dataset201_ISIC2018/imagesTs ^
  -o F:/nnunet_predictions/isic2018_demo ^
  -d 201 ^
  -c 3d_fullres ^
  -f 0
```
- `-i`：輸入影像資料夾，通常對應 `imagesTs` 或 `val/images`。
- `-o`：輸出資料夾，將生成 `.nii.gz` 或 `.npz` 預測。
- `-f 0`：使用與訓練相同的 fold。
- 若要啟用測試時資料增量（TTA），可加 `--enable_tta 1`，但會增加時間。

### 5.2 驗證方式
- 若在訓練階段啟用內建驗證，結果會存於 `validation_raw/` 內，可直接檢查 Dice/IOU。
- 將 `val/masks` 與 `predict` 輸出比對，可使用簡單 Python 腳本：
  ```python
  import nibabel as nib
  import numpy as np
  from PIL import Image
  pred = nib.load('prediction.nii.gz').get_fdata()
  Image.fromarray((pred[0] > 0.5).astype(np.uint8)*255).save('pred.png')
  ```
- 視覺化可參考 `documentation/run_inference_with_pretrained_models.md` 與 `documentation/dataset_format_inference.md`。

---

## 6. 常見問題 (FAQ)

- **Dataset 結構錯誤**：確認 `imagesTr` 與 `labelsTr` 檔名一致，影像需以 `_0000` 頻道後綴命名。
- **GPU 記憶體不足**：改用 `nnUNetv2_train 201 2d 0 ...` 或調整 `plans` 內之 `patch_size`、`batch_size`。
- **預處理緩慢**：新增 `-np <num_workers>` 控制 CPU，或以 SSD 存放 `%nnUNet_preprocessed%` 提升 I/O。
- **預測輸出為 `.nii.gz`**：可使用 `nnunetv2/imageio/` 內提供的工具轉 PNG，或自寫腳本。

---

## 7. Demo 建議流程總結
1. 建立虛擬環境並安裝 nnU-Net v2。
2. 設定 `nnUNet_*` 環境變數。
3. 將 `isic2018/train`、`val` 轉為 `Dataset201_ISIC2018` 結構並生成 `dataset.json`。
4. 執行 `nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity`。
5. 以 `nnUNetv2_train 201 3d_fullres 0 -tr nnUNetTrainer_50epochs` 進行 Demo 訓練。
6. 使用 `nnUNetv2_predict` 在 `val/images` 上產生 Demo 結果，並與 `val/masks` 比對。

依照上述步驟即可完成 ISIC2018 Demo 訓練流程並產生可視化結果。


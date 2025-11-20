# ISIC2018 → nnU-Net v2 自動轉換流程

本說明文件搭配 `tools/prepare_isic2018_dataset.py`，協助你將 `isic2018/train|val` 目錄轉成 nnU-Net v2 需要的 `DatasetXXX_*` 結構與 `dataset.json`。

## 1. 目錄假設
```
isic2018/
├── train
│   ├── images (*.png)
│   └── masks (*.png)
└── val
    ├── images (*.png)
    └── masks (*.png)  # 若無，可忽略
```

## 2. 執行條件
- 已在專案根目錄 `F:\newproj\nnUNet`。
- 已建立虛擬環境並安裝依賴：`pip install -e .`
- 已設定 `nnUNet_raw`（或者改用 `--output-root` 指定實體路徑）。

## 3. 指令範例
```
python tools/prepare_isic2018_dataset.py ^
  --input-root isic2018 ^
  --dataset-id 201 ^
  --dataset-name ISIC2018 ^
  --case-prefix ISIC ^
  --val-prefix ISICVAL ^
  --copy-val-labels
```

### Note.

```
  python tools/prepare_isic2018_dataset.py ^
    --input-root isic2018 ^
    --output-root F:/newproj/nnUNet/nnunet_raw ^
    --dataset-id 201 ^
    --dataset-name ISIC2018 ^
    --case-prefix ISIC ^
    --val-prefix ISICVAL ^
    --copy-val-labels
```

### 參數說明
- `--input-root`：來源資料根目錄，預設 `isic2018`。
- `--output-root`：輸出到哪個 nnU-Net 原始資料資料夾，預設讀取 `nnUNet_raw`。
- `--dataset-id`/`--dataset-name`：會生成 `Dataset{ID}_{name}` 目錄。
- `--case-prefix`/`--val-prefix`：決定影像重命名時使用的前綴，例如 `ISIC_0001_0000.png`。
- `--file-suffix`：預設 `.png`，若資料為 `.jpg` 可自行覆寫。
- `--copy-val-labels`：若 val/masks 也存在，開啟此旗標會同步複製到 `labelsTs` 方便驗證。

## 4. 產出結構
假設 `nnUNet_raw=F:\nnunet_raw`，執行後會生成：
```
F:\nnunet_raw\Dataset201_ISIC2018\
├── dataset.json
├── imagesTr\ISIC_0000_0000.png
├── labelsTr\ISIC_0000.png
├── imagesTs\ISICVAL_0000_0000.png
└── labelsTs\ISICVAL_0000.png   # 僅在 --copy-val-labels 時存在
```

## 5. 單核心後續流程
因系統僅配置 1 核心，建議於每個階段加上 `-np 1` 或等效參數，避免多工衝突。

1. Fingerprint + 預處理（僅啟用 1 個 worker）：
   ```
   nnUNetv2_plan_and_preprocess `
     -d 201 `
     --verify_dataset_integrity `
     -np 1
   ```
2. 訓練（可選 2D 以降低資源需求）：
   ```
   nnUNetv2_train 201 2d 0 `
     -tr nnUNetTrainer_50epochs `
     -p nnUNetPlans
   ```
   - 若需完整 5 folds，可將最後一個參數改成 `0..4` 逐一執行。
3. 推論/驗證（單核心 I/O）：
   ```
   nnUNetv2_predict `
     -i F:/newproj/nnUNet/nnunet_raw/Dataset201_ISIC2018/imagesTs `
     -o F:/newproj/nnUNet/nnunet_predictions/isic2018_demo `
     -d 201 `
     -c 2d `
     -f 0 `
     -np 1
   ```
4. 參考 `Readme_demo.md` 了解結果檢查、Dice 量測與可視化。

若需擴充命名、加入多通道或讀取其他格式，可修改 `tools/prepare_isic2018_dataset.py` 中的命名邏輯。歡迎將額外需求記錄回本文件。***

nnUNetv2_train 201 2d 0 `
  -tr nnUNetTrainer_50epochs `
  -p nnUNetPlans
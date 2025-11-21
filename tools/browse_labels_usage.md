## browse_labels.py 使用指南

`tools/browse_labels.py` 能同時列出指定資料夾內所有標籤檔，並在需要時直接開啟特定檔案供檢視。支援 PNG/JPG/BMP/TIF 直接預覽，NIfTI 檔則會提示改以 3D Viewer 查看。

### 基本指令

1. **列出所有標籤檔**
   ```
   python tools/browse_labels.py
   ```
   - 預設掃描 `F:\newproj\nnUNet\nnunet_raw\Dataset201_ISIC2018\labelsTr`
   - 列出索引、檔名與大小，可用於快速定位要檢視的檔案

2. **列出後開啟特定檔案**
   ```
   python tools/browse_labels.py --open --index 5
   ```
   - `--open`：列出清單後自動打開 `--index` 指定的檔案
   - `--index`：對應清單中的編號（由 0 起算）

3. **將非零像素強化（避免 0/1 看起來全黑）**
   ```
   python tools/browse_labels.py --open --index 5 --brighten
   ```
   - `--brighten` 會把所有非零像素放大至 255，利於肉眼辨識

4. **檢查其他資料夾**
   ```
   python tools/browse_labels.py --folder "D:/path/to/labels" --open --index 0
   ```
   - `--folder` 指定自訂路徑；路徑含空白時請加引號

### 注意事項

- 目前僅能直接預覽 2D 影像格式；NIfTI (*.nii / *.nii.gz) 仍需要利用 3D Viewer 檢視。
- 若僅需列出檔案，不要加 `--open`；腳本仍會輸出完整清單，不影響使用。
- 所有訊息皆為繁體中文，方便記錄於工作流程或 README。


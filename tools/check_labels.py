import os
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# =================設 定=================
# 請將此路徑改為你存放 Label (Mask) 的資料夾路徑
# 例如: 'F:/newproj/nnUNet/nnUNet_raw/Dataset201_ISIC2018/labelsTr'
folder_path = Path(r'F:\newproj\nnUNet\nnunet_raw\Dataset201_ISIC2018\labelsTr')
# ======================================

def check_labels(folder):
    if not folder.exists():
        print(f"錯誤: 找不到路徑 {folder}")
        return
    
    files = list(folder.glob('*.png')) + list(folder.glob('*.nii.gz'))
    print(f"正在檢查資料夾: {folder}")
    print(f"找到 {len(files)} 個檔案。開始檢查...\n")
    
    issues_found = False
    checked = 0
    max_check = min(20, len(files))  # 檢查前 20 個檔案
    
    for file_path in files[:max_check]:
        try:
            # 讀取影像數據
            if (file_path.suffix == '.gz' or file_path.suffixes == ['.nii', '.gz']):
                if not HAS_NIBABEL:
                    print(f"跳過 {file_path.name}: 需要 nibabel 來讀取 .nii.gz 檔案")
                    continue
                img = nib.load(str(file_path))
                data = img.get_fdata()
            else:  # PNG, JPG, BMP
                img = Image.open(str(file_path))
                data = np.array(img)
            
            # 獲取唯一值 (Unique values)
            unique_values = np.unique(data)
            
            # 輸出結果
            print(f"檔案: {file_path.name} | 包含數值: {unique_values}")
            
            # 判斷是否異常 (針對二分類任務)
            # nnUNet 期望的標籤應該是 [0, 1] 或 [0, 1, 2...]
            # 如果出現 255，這就是導致 NaN 的元兇
            if 255 in unique_values:
                print(f"  >>> [警告] 發現數值 255！這會導致 Loss NaN！")
                issues_found = True
            elif len(unique_values) > 1 and np.max(unique_values) > 10:
                # 假設你的類別不超過 10 個，如果數值很大通常也是錯的
                print(f"  >>> [注意] 數值似乎過大，請確認這是否正確。")
            elif not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]):
                if len(unique_values) > 2:
                    print(f"  >>> [注意] 發現多個標籤值，請確認是否為多分類任務。")
            
            checked += 1
            
        except Exception as e:
            print(f"無法讀取 {file_path.name}: {e}")
    
    print("-" * 50)
    if issues_found:
        print("[錯誤] 結論: 標籤數據有問題。nnUNet 無法處理 255 的數值。")
        print("   解決方法: 請重新執行 prepare_isic2018_dataset.py 轉換數據集。")
    else:
        print(f"[成功] 結論: 已檢查 {checked} 個檔案，標籤數值看起來正常。")
        print("   如果仍有 NaN，請檢查預處理數據或硬體設定。")

if __name__ == '__main__':
    check_labels(folder_path)


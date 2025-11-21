import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _sorted_files(directory: Path, suffix: str) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"找不到資料夾: {directory}")
    files = sorted(p for p in directory.glob(f"*{suffix}") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{directory} 沒有任何 {suffix} 檔案")
    return files


def _match_pairs(image_paths: List[Path], mask_dir: Path, suffix: str) -> List[Tuple[Path, Path]]:
    pairs = []
    missing = []
    for img_path in image_paths:
        mask_path = mask_dir / (img_path.stem + suffix)
        if not mask_path.exists():
            missing.append(mask_path)
        else:
            pairs.append((img_path, mask_path))
    if missing:
        raise FileNotFoundError(f"以下標籤檔不存在:\n" + "\n".join(str(p) for p in missing))
    return pairs


def _binarize_and_save_mask(src: Path, dst: Path) -> None:
    """
    將 mask 二值化為 [0, 1]，確保符合 nnU-Net 要求。
    處理可能的 255 值或其他非標準標籤。
    """
    arr = np.array(Image.open(src))
    
    # 如果是多通道，只取第一個通道
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    
    # 將所有非零值轉為 1，確保結果只有 [0, 1]
    # 這會處理 255、128 等任何非零值
    bin_arr = (arr > 0).astype(np.uint8)
    
    # 驗證結果只包含 [0, 1]
    unique_vals = np.unique(bin_arr)
    if not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [0, 1]):
        raise ValueError(
            f"二值化後標籤包含意外數值: {unique_vals}。"
            f"檔案: {src}"
        )
    
    Image.fromarray(bin_arr, mode='L').save(dst)


def _save_rgb_channels(src: Path, dst_dir: Path, case_id: str, file_suffix: str) -> str:
    arr = np.array(Image.open(src))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{src} 不是 3 通道影像，請確認資料格式")
    for channel in range(3):
        channel_arr = arr[..., channel]
        out_path = dst_dir / f"{case_id}_{channel:04d}{file_suffix}"
        Image.fromarray(channel_arr).save(out_path)
    return f"{case_id}_0000{file_suffix}"


def prepare_dataset(
    input_root: Path,
    output_root: Path,
    dataset_id: int,
    dataset_name: str,
    channel_name: str,
    case_prefix: str,
    val_prefix: str,
    file_suffix: str,
    copy_val_labels: bool,
):
    train_img_dir = input_root / "train" / "images"
    train_mask_dir = input_root / "train" / "masks"
    val_img_dir = input_root / "val" / "images"
    val_mask_dir = input_root / "val" / "masks"

    train_images = _sorted_files(train_img_dir, file_suffix)
    train_pairs = _match_pairs(train_images, train_mask_dir, file_suffix)
    val_images = _sorted_files(val_img_dir, file_suffix)

    dataset_dir = output_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    training_entries = []
    test_entries = []

    print(f"開始複製訓練資料 ({len(train_pairs)} 筆)...")
    for idx, (img_path, mask_path) in enumerate(train_pairs):
        case_id = f"{case_prefix}_{idx:04d}"
        first_channel_name = _save_rgb_channels(img_path, images_tr, case_id, file_suffix)
        img_target = images_tr / first_channel_name
        mask_target = labels_tr / f"{case_id}{file_suffix}"
        _binarize_and_save_mask(mask_path, mask_target)
        training_entries.append(
            {
                "image": f"./imagesTr/{img_target.name}",
                "label": f"./labelsTr/{mask_target.name}",
            }
        )

    print(f"開始複製驗證/測試資料 ({len(val_images)} 筆)...")
    for idx, img_path in enumerate(val_images):
        case_id = f"{val_prefix}_{idx:04d}"
        first_channel_name = _save_rgb_channels(img_path, images_ts, case_id, file_suffix)
        img_target = images_ts / first_channel_name
        test_entries.append(f"./imagesTs/{img_target.name}")
        if copy_val_labels:
            mask_path = val_mask_dir / (img_path.stem + file_suffix)
            if mask_path.exists():
                mask_target = labels_ts / f"{case_id}{file_suffix}"
                _binarize_and_save_mask(mask_path, mask_target)

    dataset_json = {
        "name": dataset_name,
        "description": "ISIC 2018 dermoscopy segmentation demo",
        "reference": "https://challenge2018.isic-archive.com/",
        "licence": "CC-BY",
        "release": "1.0",
        "channel_names": {
            "0": f"{channel_name}_R",
            "1": f"{channel_name}_G",
            "2": f"{channel_name}_B",
        },
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(training_entries),
        "numTest": len(test_entries),
        "file_ending": file_suffix,
        "training": training_entries,
        "test": test_entries,
    }

    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"dataset.json 已建立：{dataset_json_path}")
    print("轉換完成。")


def parse_args():
    parser = argparse.ArgumentParser(description="將 isic2018 資料轉為 nnU-Net v2 格式")
    parser.add_argument("--input-root", type=Path, default=Path("isic2018"), help="原始資料根目錄")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="輸出 DatasetXXX 位置（預設使用環境變數 nnUNet_raw）",
    )
    parser.add_argument("--dataset-id", type=int, default=201, help="資料集 ID")
    parser.add_argument("--dataset-name", type=str, default="ISIC2018", help="資料集名稱")
    parser.add_argument("--channel-name", type=str, default="Dermoscopic", help="channel_names 中的名稱")
    parser.add_argument("--case-prefix", type=str, default="ISIC", help="訓練資料命名前綴")
    parser.add_argument("--val-prefix", type=str, default="ISICVAL", help="驗證資料命名前綴")
    parser.add_argument("--file-suffix", type=str, default=".png", help="影像與標籤副檔名")
    parser.add_argument(
        "--copy-val-labels",
        action="store_true",
        help="是否將 val/masks 一併複製至 labelsTs（非 nnU-Net 必要，但可供驗證）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root or os.environ.get("nnUNet_raw")
    if not output_root:
        raise EnvironmentError("請透過 --output-root 或環境變數 nnUNet_raw 指定輸出路徑")
    prepare_dataset(
        input_root=args.input_root,
        output_root=Path(output_root),
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        channel_name=args.channel_name,
        case_prefix=args.case_prefix,
        val_prefix=args.val_prefix,
        file_suffix=args.file_suffix,
        copy_val_labels=args.copy_val_labels,
    )


if __name__ == "__main__":
    main()


import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def list_label_files(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} 不是資料夾")

    candidates = []
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".nii.gz"):
        candidates.extend(folder.glob(f"*{suffix}"))

    files = sorted(f for f in candidates if f.is_file())
    if not files:
        raise FileNotFoundError(f"{folder} 沒有找到任何標籤檔（支援 PNG/JPG/BMP/TIF/NIfTI）")

    print(f"共找到 {len(files)} 個標籤檔：")
    for idx, file in enumerate(files):
        size_kb = file.stat().st_size / 1024
        print(f"[{idx:04d}] {file.name:<40} {size_kb:8.1f} KB")
    return files


def open_label_image(path: Path, brighten: bool):
    if path.suffix.lower() in {".nii", ".gz"}:
        print("目前僅支援直接預覽影像類別檔（PNG/JPG/BMP/TIF）。NIfTI 請用 3D Viewer。")
        return

    arr = np.array(Image.open(path))
    if brighten and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if brighten:
        # 將任何非零值放大到 255，方便視覺化
        mask = arr > 0
        arr = np.zeros_like(arr, dtype=np.uint8)
        arr[mask] = 255

    Image.fromarray(arr).show(title=path.name)
    print(f"已開啟 {path.name}，請於跳出的視窗檢視。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="列出並快速檢視 nnU-Net 標籤檔（labelsTr/labelsTs）"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(r"F:\newproj\nnUNet\nnunet_raw\Dataset201_ISIC2018\labelsTr"),
        help="標籤資料夾路徑，預設指向 Dataset201_ISIC2018/labelsTr",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="指定要開啟的檔案索引（搭配 --open 使用）",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="列出檔案後立即開啟 --index 指定的檔案",
    )
    parser.add_argument(
        "--brighten",
        action="store_true",
        help="預覽時將所有非零像素放大為 255，方便肉眼辨識",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        files = list_label_files(args.folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(exc)
        sys.exit(1)

    if args.open:
        if args.index is None:
            print("請使用 --index 指定要開啟的檔案編號，例如：--open --index 5")
            return
        if args.index < 0 or args.index >= len(files):
            print(f"索引 {args.index} 超出範圍（0 ~ {len(files) - 1}）")
            return
        open_label_image(files[args.index], brighten=args.brighten)

    print("完成。")


if __name__ == "__main__":
    main()


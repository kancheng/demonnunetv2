$env:nnUNet_raw = "F:/newproj/nnUNet/nnunet_raw"
$env:nnUNet_preprocessed = "F:/newproj/nnUNet/nnunet_preprocessed"
$env:nnUNet_results = "F:/newproj/nnUNet/nnunet_results"

setx nnUNet_n_proc_DA 1
setx nnUNet_n_proc_DA_val 1
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_n_proc_DA_val = "1"
$env:nnUNet_dataloader_queue = "2"

python tools/prepare_isic2018_dataset.py   --input-root isic2018   --output-root F:/newproj/nnUNet/nnunet_raw   --dataset-id 201   --dataset-name ISIC2018  --case-prefix ISIC   --val-prefix ISICVAL   --copy-val-labels

nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity  -np 1

nnUNetv2_train 201 2d 0  -tr nnUNetTrainer_50epochs  -p nnUNetPlans


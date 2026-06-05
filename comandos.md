# Clasificación binaria maíz — Comandos

## 1. Split train/val/test

```bash
python src/0_data_split.py \
    --how random \
    --data_path /mnt/yacy_1/prod/christener/S4A \
    --coco_path data/coco_split \
    --prefix test_1 \
    --ratio 60 20 20 \
    --tiles all \
    --years all \
    --experiment 2
```

## 2. Precomputar medianas

```bash
nohup python3 src/1_medians_preprocessing.py \
    --data /mnt/yacy_1/prod/ferreyra/S4A_30m \
    --root_coco_path /home1/ferreyra/prod_ia_unne/data/coco_split \
    --out_path /mnt/yacy_1/prod/ferreyra/medians_full \
    --output_size 61 61 \
    --bands B02 B03 B04 B08 \
    --num_workers 4 \
    --group_freq 1MS
```

## 3. Entrenar

```bash
python src/2_train_model.py \
    --prefix test_1 \
    --train \
    --model unet \
    --saved_medians \
    --medians_path /mnt/yacy_1/prod/ferreyra/S4A/medians/ \
    --parcel_loss \
    --weighted_loss \
    --root_path_coco data/coco_split \
    --netcdf_path /mnt/yacy_1/prod/ferreyra/S4A_30m/ \
    --num_epochs 200 \
    --batch_size 4 \
    --bands B02 B03 B04 B08 \
    --img_size 61 61 \
    --requires_norm \
    --num_workers 14 \
    --num_gpus 1 \
    --fixed_window
```

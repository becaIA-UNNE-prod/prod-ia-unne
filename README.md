# Beca IA UNNE. Módulo producción

La documentación está en `doc/`

# Pasos realizados
Primero, hago el split en train, test y val.
```{python}
python src/0_data_split.py --how random --data_path=/mnt/yacy_1/prod/christener/S4A --coco_path=data/coco_split --prefix=test_1 --ratio 60 20 20 --tiles all --years all --experiment 2
```

Precomputo medianas. Esto es muy importante para no perder tiempo en el entrenamiento y usar la GPU al 100%.
```{python}
nohup python src/1_medians_preprocessing.py --data=/mnt/yacy_1/prod/christener/S4A --root_coco_path=data/coco_split --prefix=test_1 --out_path=/mnt/yacy_1/prod/ferreyra/S4A/medians --num_workers 24 --bands B02 B03 B04 B08 --output_size 128 128
```

Entreno
```{python}
python src/2_train_model.py --prefix test_1 --train --model unet --saved_medians --medians_path /mnt/yacy_1/prod/ferreyra/S4A/medians/ --parcel_loss --weighted_loss --root_path_coco data/coco_split --netcdf_path /mnt/yacy_1/prod/ferreyra/S4A_30m/ --num_epochs 200 --batch_size 4 --bands B02 B03 B04 B08 --img_size 61 61 --requires_norm --num_workers 14 --num_gpus 1 --fixed_window
```


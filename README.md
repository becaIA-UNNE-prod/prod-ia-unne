# Beca IA UNNE. Módulo producción

La documentación está en `doc/`

# Pasos realizados
Primero, hago el split en train, test y val.
```{python}
python src/0_data_split.py --how random --data_path=/mnt/yacy_3/prod/christener/S4A --coco_path=data/coco_split --prefix=test_1 --ratio 60 20 20 --tiles all --years all --experiment 2
```

Precomputo medianas. Esto es muy importante para no perder tiempo en el entrenamiento y usar la GPU al 100%.
```{python}
nohup python src/1_medians_preprocessing.py --data=/mnt/yacy_3/prod/christener/S4A --root_coco_path=data/coco_split --prefix=test_1 --out_path=/mnt/yacy_3/prod/ferreyra/S4A/medians --num_workers 24 --bands B02 B03 B04 B08
```

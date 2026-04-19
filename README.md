# Beca IA UNNE. Módulo producción

La documentación está en `doc/`

# Pasos realizados
Primero, hago el split en train, test y val.
```{python}
python src/0_data_split.py --how random --data_path=/mnt/yacy_3/prod/christener/S4A --coco_path=data/coco_split --prefix=test_1 --ratio 60 20 20 --tiles all --years all --experiment 2
```

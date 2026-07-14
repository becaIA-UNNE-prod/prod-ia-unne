# Documentación: 0_data_split.py

Este script es responsable de dividir el conjunto de datos original (en formato NetCDF) en subconjuntos de **entrenamiento (train)**, **validación (val)** y **prueba (test)**. La salida consiste en archivos JSON siguiendo el estándar de anotaciones **MS COCO**.

Es importante ver `utils/coco_tools.py`

## Descripción General

El script permite realizar particiones de tres formas:
1.  **Estratificada (`stratified`):** Asegura que la distribución de etiquetas en cada subconjunto sea proporcional a la del dataset original, utilizando un enfoque multietiqueta a nivel de imagen (patch).
2.  **Aleatoria (`random`):** Realiza una división simple al azar a nivel de patch.
3.  **Group-stratified (`group_stratified`):** pensada para el dataset de Córdoba (maíz), donde el número de tiles crece continuamente. Ver sección dedicada más abajo.

Los modos `stratified` y `random` son legacy del dataset original (Sen4AgriNet, Cataluña/Francia) y gestionan la compatibilidad de etiquetas entre esas dos regiones vía `--experiment`. **No usar `--experiment` para el dataset de Córdoba**, ya que las listas de tiles están hardcodeadas a Cataluña/Francia y no aplican.

---

## Modo `group_stratified` (recomendado para Córdoba / datasets que crecen en tiles)

Los otros dos modos particionan a nivel de **patch individual**. Como los patches son sub-recortes espaciales contiguos dentro de un mismo tile (`{year}_{tile}_patch_{row}_{col}.nc`), un split random o estratificado a nivel de patch mezcla patches vecinos del mismo tile entre train/val/test. Esto es data leakage espacial: patches adyacentes comparten campo, nubosidad y fecha de imagen, así que el modelo puede "memorizar" el tile en vez de generalizar, e infla las métricas de val/test.

`group_stratified` resuelve esto particionando por **grupo `(tile, year)`**: todos los sub-patches de un mismo tile-año van siempre al mismo split. Además:

- **Estratifica por prevalencia de la clase objetivo** (maíz): los grupos se bucketizan según el % de patches que contienen maíz, para que train/val/test mantengan un balance de clase similar.
- **Asignación incremental y estable**: una vez que un grupo (tile-año) fue asignado a un split, nunca se mueve. Al bajar tiles nuevos, sólo se asignan los grupos nuevos (al split que más los necesite para acercarse al ratio pedido), así las métricas siguen siendo comparables entre corridas a medida que el dataset crece.

### Paso previo: catálogo de patches

`group_stratified` necesita un catálogo (csv) con metadata liviana por patch (tile, año, si contiene maíz). Se genera/actualiza con:

```bash
python src/0a_build_patch_catalog.py \
    --data_path /mnt/yacy_1/prod/christener/S4A \
    --catalog_path data/patch_catalog.csv \
    --num_workers 8
```

Sólo lee el grupo `labels` de cada `.nc` (no las bandas espectrales), y es **incremental**: si se corre de nuevo después de bajar tiles nuevos, sólo procesa los archivos que todavía no están en el catálogo.

### Correr el split

```bash
python src/0_data_split.py \
    --how group_stratified \
    --data_path /mnt/yacy_1/prod/christener/S4A \
    --coco_path data/coco_split \
    --prefix test_2 \
    --ratio 60 20 20 \
    --catalog_path data/patch_catalog.csv \
    --assignments_path data/coco_split/test_2_group_assignments.json
```

`--assignments_path` guarda el mapeo `tile_year -> split`. Si el archivo ya existe, esos grupos quedan fijos y sólo se agregan los tile-años nuevos presentes en el catálogo. Para "congelar" un split existente y simplemente extenderlo con tiles nuevos, basta con reutilizar el mismo `--assignments_path`.

`--n_buckets` (default 3) controla cuántos buckets de prevalencia de maíz se usan para estratificar; se reduce automáticamente si hay pocos grupos o muchos empates (por ejemplo, muchos tile-años con 0% de maíz).

---

## Configuración de Experimentos

El parámetro `--experiment` permite seleccionar configuraciones predefinidas:

| Experimento | Propósito | Train / Val | Test |
| :--- | :--- | :--- | :--- |
| **1** | Evaluación general | Muestreo aleatorio de todos los tiles y años. | Muestreo aleatorio de todos los tiles y años. |
| **2** | Generalización geográfica | Solo tiles de **Cataluña** (2019, 2020). | Solo tiles de **Francia** (2019). |
| **3** | Generalización espacio-temporal | Solo tiles de **Francia** (2019). | Solo tiles de **Cataluña** (2020). |

---

## Parámetros de la Línea de Comandos (CLI)

| Argumento | Tipo | Requerido | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `--how` | `str` | **Sí** | - | Método de división: `stratified` o `random`. |
| `--data_path` | `str` | No | `dataset/netcdf/` | Ruta a los archivos `.nc`. |
| `--coco_path` | `str` | No | `coco_files/` | Carpeta de salida para los archivos `.json`. |
| `--ratio` | `int list`| No | `60 20 20` | Proporción para Train, Val y Test. |
| `--prefix` | `str` | No | *Timestamp* | Prefijo para los nombres de los archivos generados. |
| `--plot_distros` | `flag` | No | `False` | Si se activa, muestra gráficos de frecuencia de clases. |
| `--tiles` | `str list`| No | `all` | Lista de tiles específicos a incluir (ej: `31TCG 31TDG`). |
| `--years` | `str list`| No | `all` | Lista de años específicos (ej: `2019 2020`). |
| `--num_patches` | `int` | No | `None` | Máximo de imágenes totales a procesar. |
| `--experiment` | `int` | No | `None` | Selecciona el experimento predefinido (1, 2 o 3). |
| `--seed` | `int` | No | `None` | Semilla para reproducibilidad. |

---

## Funciones Principales

### `create_dataframe()`
Escanea la carpeta de datos y construye un `pandas.DataFrame` que mapea cada archivo de parche con su conjunto de etiquetas únicas. Filtra etiquetas que no están en el `LINEAR_ENCODER` y asegura que existan etiquetas comunes si se requiere.

### `plot_label_frequencies()`
Calcula y grafica cuántas imágenes contienen cada tipo de cultivo para cada set generado. Es útil para verificar visualmente que la estratificación funcionó correctamente.

---

## Flujo de Trabajo

1.  **Carga y Filtrado:** El script lee los archivos NetCDF y filtra por año/tile/etiqueta según los argumentos o el experimento elegido.
2.  **Estratificación Iterativa:** Si se elige `--how stratified`, se utiliza `IterativeStratification` de la librería `skmultilearn`. Este proceso ocurre en dos pasos:
    * División de los datos originales en `Train` y un set temporal `Val+Test`.
    * División del set temporal en `Val` y `Test` definitivos.
3.  **Exportación:** Los DataFrames resultantes se procesan para crear los archivos JSON COCO mediante las funciones utilitarias `create_coco_dataframe` o `create_coco_netcdf`.

---

## Ejemplos de Uso

**Ejecución básica estratificada:**
```bash
python src/0_data_split.py --how stratified --experiment 1 --num_patches 1000
```

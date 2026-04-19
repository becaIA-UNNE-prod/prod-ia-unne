# Documentación: 0_data_split.py

Este script es responsable de dividir el conjunto de datos original (en formato NetCDF) en subconjuntos de **entrenamiento (train)**, **validación (val)** y **prueba (test)**. La salida consiste en archivos JSON siguiendo el estándar de anotaciones **MS COCO**.

Es importante ver `utils/coco_tools.py`

## Descripción General

El script permite realizar particiones de dos formas:
1.  **Estratificada:** Asegura que la distribución de etiquetas en cada subconjunto sea proporcional a la del dataset original, utilizando un enfoque multietiqueta a nivel de imagen.
2.  **Aleatoria:** Realiza una división simple al azar.

Además, el script gestiona la compatibilidad de etiquetas entre diferentes regiones geográficas (Cataluña y Francia) y años, permitiendo experimentos de generalización.

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

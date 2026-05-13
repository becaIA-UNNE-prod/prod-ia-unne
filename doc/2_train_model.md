# Documentación del Script de Entrenamiento y Evaluación de Modelos

Este script está diseñado para ejecutar los modelos usando **PyTorch
Lightning**.


## Dependencias Principales

- **Frameworks:** `torch`, `pytorch\_lightning`
- **Módulos estándar:** `argparse`, `pathlib`, `datetime`
- **Módulos locales (esperados en el proyecto):
  - `model.\*` (Contiene las arquitecturas: `ConvLSTM`, `TempCNN`, `ConvSTAR`, `UNet`)
  - `utils.PAD\_datamodule.PADDataModule` (Gestión de carga de datos NetCDF y COCO)
  - `utils.settings.config` (Configuraciones globales como constantes y semillas)
  - `utils.tools.font\_colors` (Para dar formato a la salida en consola)

## Argumentos de CLI

### 1. Modo de Ejecución

- `--train`: Ejecutar el modelo en modo entrenamiento (por defecto es Test).
- `--devtest`: Ejecuta una prueba rápida de desarrollo (fast dev run) para verificar que el código no tenga errores antes del entrenamiento real.
- `--resume`: Ruta del checkpoint desde el cual reanudar el entrenamiento, o el string `'last'` para retomar la última ejecución disponible.
- `--load\_checkpoint`: Ruta del checkpoint que se cargará explícitamente para evaluar el modelo (Test).

### 2. Selección de Modelo

- `--model` **(Requerido)**: Arquitectura a utilizar. Opciones válidas: `\['convlstm', 'tempcnn', 'convstar', 'unet'\]`.

### 3. Rutas y Datos

- `--root\_path\_coco`: Directorio raíz de los archivos JSON de COCO. *(Por defecto: "coco\_files/")*
- `--prefix\_coco`: Prefijo para los archivos COCO (ej. si es "x", buscará "x\_coco\_train.json").
- `--netcdf\_path`: Ruta al directorio que contiene los datos NetCDF. *(Por defecto: "dataset/netcdf")*
- `--prefix`: Prefijo utilizado para nombrar la carpeta de volcado de datos/logs. Si no se provee, se usa un *timestamp*.
- `--group\_freq`: Frecuencia de agrupación de los datos temporales (alias de Pandas). *(Por defecto: "1MS" - 1 Mes)*
- `--fixed\_window`: Bandera para forzar una ventana temporal fija que abarque desde abril (mes 4) hasta septiembre (mes 9).
- `--medianas\_path`: Directorio en donde se encuentran las medianas precomputadas

### 4. Hiperparámetros y Preprocesamiento

- `--num\_epochs`: Número de épocas de entrenamiento. *(Por defecto: 10)*
- `--batch\_size`: Tamaño del lote de datos. *(Por defecto: 4)*
- `--lr`: Tasa de aprendizaje inicial. *(Por defecto: 1e-1)*
- `--window\_len`: Longitud de la ventana móvil (secuencia de entrada). *(Por defecto: 6)*
- `--bands`: Bandas de la imagen a procesar separadas por espacio. *(Por defecto: todas las configuradas en BANDS)*
- `--img\_size`: Tamaño del parche (subpatch) de imagen a usar como entrada (ej. `--img\_size 64 64`). Si se usa `tempcnn`, esto se sobrescribe a `(1, 1)`.
- `--requires\_norm`: Normaliza los datos al rango \[0, 1\]. *(Por defecto: False)*
- `--saved\_medians`: Precalcula y exporta las medianas de las imágenes.

### 5. Configuración de la Función de Pérdida (Loss) y Clases

- `--parcel\_loss`: Usa una función de pérdida que solo toma en cuenta los píxeles que pertenecen a una parcela de cultivo (ignora el fondo).
- `--weighted\_loss`: Usa una función de pérdida ponderada con pesos precalculados por clase.
- `--binary\_labels`: Mapea las categorías a un problema binario (0: fondo, 1: parcela). *(Por defecto: Multiclase)*

### 6. Máscaras de Hollstein (Condiciones Meteorológicas)

Banderas para incluir máscaras específicas en los datos de entrada:

- `--return\_masks`: Bandera general para usar las máscaras.
- `--clouds`: Máscara para nubes.
- `--cirrus`: Máscara para cirros.
- `--shadow`: Máscara para sombras.
- `--snow`: Máscara para nieve.

### 7. Configuración de Hardware

- `--num\_workers`: Número de subprocesos (workers) para el `DataLoader`. *(Por defecto: 6)*
- `--num\_gpus`: Número de GPUs a usar por nodo. Utiliza la estrategia `ddp` si es mayor a 1. *(Por defecto: 1)*
- `--num\_nodes`: Número de nodos para entrenamiento distribuido. *(Por defecto: 1)*


## Flujo de Ejecución (Paso a Paso)

1. **Parseo y Validación:** El script lee los argumentos e interrumpe la
ejecución si se intenta hacer *Test* (sin bandera `--train`) pero no se provee
un `--load\_checkpoint`.

2. **Preparación de Directorios:** Comprueba la existencia de los archivos COCO
y crea la estructura `logs/loaders` para guardar los resultados de
`TensorBoard` y los `checkpoints`.

3. **Configuración del Modelo:** Dependiendo del `--model` elegido, se crea la
instancia correspondiente (`ConvLSTM`, `ConvSTAR`, `UNet` o `TempCNN`). Si es
modo Test o Resume, se cargan los pesos desde el checkpoint especificado. Las
arquitecturas reconstruyen parámetros dinámicos (como la tasa de aprendizaje
                                                   guardada en `lrs.txt` si se
                                                   está reanudando).

4. **Carga de Datos:** Se inicializa `PADDataModule` inyectando toda la
configuración de entrada, procesamiento de parches (`img\_size`), y aplicación
de máscaras de nubes.

5. **Ejecución del Trainer:** \* **Si es Entrenamiento:** Configura
`ModelCheckpoint`, `LearningRateMonitor` y `TensorBoardLogger`. Luego invoca
`trainer.fit(model, datamodule=dm)`.
   - **Si es Test:** Pone el modelo en `eval()` e invoca `trainer.test(model, datamodule=dm)`.


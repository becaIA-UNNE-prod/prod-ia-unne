Este script procesa conjuntos de datos de imágenes satelitales
Sentinel-2 almacenados en formato NetCDF y genera archivos de medianas
temporales y etiquetas en formato .npy (NumPy).

Su objetivo principal es preparar los datos para el entrenamiento de
modelos. Para ello, el script:

1. Agrupa las imágenes satelitales en ventanas de tiempo (ej. meses).

2. Calcula la mediana de los píxeles para manejar la nubosidad y datos
   faltantes (utilizando interpolación y extrapolación).

3. Armoniza la resolución espacial de todas las bandas
    multiespectrales (remuestreando bandas de 20m y 60m a 10m).

4. Divide las imágenes grandes en sub-parches (sub-patches) más
    pequeños mediante una técnica de ventana deslizante (sliding
    window).

5. Ejecuta este proceso en paralelo para optimizar el tiempo de
    cómputo.

Argumento,Tipo,Por defecto,Descripción
--data,str,dataset/netcdf,Ruta al directorio que contiene los archivos NetCDF originales.
--root_coco_path,str,coco_files/,Ruta al directorio donde se encuentran los archivos JSON de COCO (que definen los splits de train/val/test).
--prefix_coco,str,None,"Prefijo opcional para los archivos COCO (ej. si es exp1, buscará exp1_coco_train.json)."
--out_path,str,logs/medians,Directorio de salida donde se guardarán los sub-parches en formato .npy.
--group_freq,str,1MS,"Frecuencia de agrupación temporal de Pandas (ej. 1MS = 1 Month Start). Define el tamaño del ""bin"" de tiempo."
--output_size,list,None,"Tamaño deseado para los sub-parches (ej. --output_size 128 128). Si no se provee, se asume [366, 366]."
--bands,list,Todas las bandas,Permite especificar un subconjunto de bandas a procesar (ej. --bands B02 B03 B04 B08).
--num_workers,int,8,Cantidad de procesos en paralelo a utilizar para acelerar el procesamiento.

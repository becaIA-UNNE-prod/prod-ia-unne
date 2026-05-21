#!/bin/bash

DIRECTORIO="/mnt/yacy_1/prod/dat/S4A/"
TMP_LOG="estructuras_temp.log"

# Limpiamos el log temporal por si quedó de una corrida anterior
> "$TMP_LOG"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Iniciando... Contando la cantidad total de archivos .nc (esto lleva unos segundos)..."
TOTAL_ARCHIVOS=$(find "$DIRECTORIO" -type f -name "*.nc" | wc -l)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Se encontraron $TOTAL_ARCHIVOS archivos para procesar. ¡Arrancamos!"

CONTADOR=0

find "$DIRECTORIO" -type f -name "*.nc" | while read -r archivo; do

    # Extraemos la estructura y la guardamos
    estructura=$(gdalinfo "$archivo" 2>/dev/null | grep "SUBDATASET_.*_DESC" | cut -d'=' -f2 | tr '\n' '|')
    echo "$estructura" >> "$TMP_LOG"

    ((CONTADOR++))

    # Tiramos un print al log cada 50 archivos
    if (( CONTADOR % 50 == 0 )); then
        PORCENTAJE=$(( 100 * CONTADOR / TOTAL_ARCHIVOS ))
        echo "[$(date +'%H:%M:%S')] Procesados $CONTADOR de $TOTAL_ARCHIVOS ($PORCENTAJE%)"
    fi
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracción terminada. Calculando el ranking de estructuras..."
echo ""
echo "📊 TOP 10 ESTRUCTURAS MÁS REPETIDAS:"
echo "------------------------------------------------"
sort "$TMP_LOG" | uniq -c | sort -nr | head -n 10

# Limpieza
rm "$TMP_LOG"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Proceso 100% finalizado. Podés ir en paz."

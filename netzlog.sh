#!/data/data/com.termux/files/usr/bin/bash

# Ping-Ziel
TARGETS=("8.8.8.8" "1.1.1.1" "9.9.9.9")
INTERVAL=2  # Wie oft GPS geprÃ¼ft wird (Sekunden)

# Vorbereitung
LOGFILE="/storage/emulated/0/Download/netzlog_adaptive_$(date +%Y%m%d_%H%M%S).csv"
# Header anpassen 
if [ ! -f "$LOGFILE" ]; then
    echo -n "timestamp,latitude,longitude,distance_m" > "$LOGFILE"
    for T in "${TARGETS[@]}"; do
        echo -n ",ping_${T}" >> "$LOGFILE"
    done
    echo "" >> "$LOGFILE"
fi

echo "[*] Starte adaptives Logging nach Geschwindigkeit & Genauigkeit"
echo "[*] Logfile: $LOGFILE"
echo "[*] Abbrechen mit CTRL+C"

# Haversine-Formel zur Distanzberechnung
function calc_distance() {
  awk -v lat1="$1" -v lon1="$2" -v lat2="$3" -v lon2="$4" '
    BEGIN {
      pi = atan2(0, -1)
      r = 6371000
      dlat = (lat2 - lat1) * pi / 180
      dlon = (lon2 - lon1) * pi / 180
      a = sin(dlat/2)^2 + cos(lat1*pi/180) * cos(lat2*pi/180) * sin(dlon/2)^2
      c = 2 * atan2(sqrt(a), sqrt(1-a))
      print r * c
    }'
}

# Warten auf brauchbare GPS-Referenz
echo "[*] Warte auf erste gültige GPS-Position..."

while true; do
    LOC=$(termux-location -p gps -r once)

    # Wenn leer oder fehlerhaft, erneut versuchen
    if [ -z "$LOC" ]; then
        echo "[-] Keine Standortdaten erhalten, warte..."
        sleep 2
        continue
    fi

    LAT=$(echo "$LOC" | jq -r '.latitude // 0')
    LON=$(echo "$LOC" | jq -r '.longitude // 0')
    ACCURACY=$(echo "$LOC" | jq -r '.accuracy // 999')

    if [ "$LAT" = "0" ] || [ "$LON" = "0" ]; then
        echo "[-] Ungueltige Koordinaten (0,0), warte..."
        sleep 2
        continue
    fi

    if [ "$(echo "$ACCURACY > 50" | bc)" -eq 1 ]; then
        echo "[-] GPS zu ungenau (${ACCURACY}), warte..."
        sleep 2
        continue
    fi

    # Referenzdaten setzen
    REF_LAT=$LAT
    REF_LON=$LON
    echo "[*] Referenz gesetzt bei: $REF_LAT, $REF_LON (Genauigkeit: ${ACCURACY})"
    break
done

# Regelmäßige Messung
while true; do
    # Aktuelle Position mit gps abfragen
    LOC=$(termux-location -p gps -r once)

    # kein gps signal -> keine Messung
    if [ -z "$LOC" ]; then
	echo "Kein GPS-Signal erhalten - überspringen..."
	sleep "$INTERVAL"
	continue
    fi
    
    LAT=$(echo "$LOC" | jq -r '.latitude // 0')
    LON=$(echo "$LOC" | jq -r '.longitude // 0')
    SPEED=$(echo "$LOC" | jq -r '.speed // 0')
    ACCURACY=$(echo "$LOC" | jq -r '.accuracy // 999')

    # Messung nur bei brauchbarer Genauigkeit
    if [ "$(echo "$ACCURACY > 50" | bc)" -eq 1 ]; then
        echo "GPS ungenau ($ACCURACY m), Ueberspringe Messung..."
        sleep "$INTERVAL"
        continue
    fi

    # Abstand berechnen
    DIST=$(calc_distance "$REF_LAT" "$REF_LON" "$LAT" "$LON")
    DIST_INT=${DIST%.*}

    # Dynamisches Intervall je nach Geschwindigkeit
    if (( $(echo "$SPEED <= 4" | bc -l) )); then
        MIN_DIST=10
    elif (( $(echo "$SPEED <= 10" | bc -l) )); then
        MIN_DIST=25
    else
        MIN_DIST=50
    fi

    if [ "$DIST_INT" -ge "$MIN_DIST" ]; then
        TS=$(date "+%Y-%m-%d %H:%M:%S")

       # SpÃ¤ter im Loop: Pings ausfÃ¼hren
	PING_RESULTS=()
	for T in "${TARGETS[@]}"; do
		P=$(ping -c 1 -W 1 "$T" | grep -oP 'time=\K[0-9.]+')
    		PING_RESULTS+=("$P")
		echo "Ping to $T took $P ms"
	done

# In Datei schreiben
echo -n "$TS,$LAT,$LON,$DIST_INT" >> "$LOGFILE"
for P in "${PING_RESULTS[@]}"; do
    echo -n ",$P" >> "$LOGFILE"
done
echo "" >> "$LOGFILE"

        REF_LAT=$LAT
        REF_LON=$LON
    else
        echo "Noch zu nah ($DIST_INT m), warte..."
    fi

    sleep "$INTERVAL"
done

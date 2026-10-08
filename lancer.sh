#!/bin/bash
# 1-clic : double-cliquer ou lancer depuis le menu. Ouvre le navigateur + démarre le serveur.
# Fermer le terminal arrête l'appli.
cd "$(dirname "$0")" || exit 1
command -v python3 >/dev/null || { echo "python3 manquant"; read -p "Entrée pour fermer"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg manquant : sudo apt install ffmpeg"; read -p "Entrée pour fermer"; exit 1; }
[ -x bin/realesrgan-ncnn-vulkan ] || echo "Note: binaire IA absent -> mode fallback ffmpeg (sans IA)."

PORT=8000
for p in 8000 8001 8002 8003 8004 8005; do
  (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null || { PORT=$p; break; }
done
echo "Upscale IA -> http://localhost:$PORT"
echo "(ferme ce terminal pour arrêter)"
python3 app.py --serve --port "$PORT" &
SRV=$!
trap "kill $SRV 2>/dev/null; exit" INT TERM EXIT
sleep 1.5
(xdg-open "http://localhost:$PORT" >/dev/null 2>&1 &)
wait $SRV

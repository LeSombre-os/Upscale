#!/bin/bash
# Récupère binaire + modèles Real-ESRGAN (non versionnés, ~56 Mo).
# Usage: ./setup.sh [--bureau]   (--bureau = crée aussi l'icône 1-clic avec TON chemin)
set -e
cd "$(dirname "$0")"
URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
if [ -x bin/realesrgan-ncnn-vulkan ] && [ -f models/realesrgan-x4plus.bin ]; then
  echo "bin/ + models/ déjà présents."
else
  echo "Téléchargement Real-ESRGAN NCNN (~56 Mo)..."
  curl -L -o /tmp/upscale-ncnn.zip "$URL"
  rm -rf /tmp/upscale-ncnn && mkdir -p /tmp/upscale-ncnn
  unzip -o -q /tmp/upscale-ncnn.zip -d /tmp/upscale-ncnn
  mkdir -p bin models
  cp /tmp/upscale-ncnn/realesrgan-ncnn-vulkan bin/
  chmod +x bin/realesrgan-ncnn-vulkan
  cp /tmp/upscale-ncnn/models/* models/
  rm -rf /tmp/upscale-ncnn /tmp/upscale-ncnn.zip
  echo "OK : $(du -sh bin models | tr '\n' ' ')"
fi
command -v ffmpeg >/dev/null || echo "Il manque ffmpeg : sudo apt install ffmpeg"
if [ "$1" = "--bureau" ]; then
  HERE="$(pwd)"
  cat > Upscale.desktop <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Upscale IA
Comment=Upscale vidéo IA local — 1 clic
Exec=$HERE/lancer.sh
Path=$HERE
Icon=video-display
Terminal=true
Categories=AudioVideo;Video;
StartupNotify=true
EOF
  chmod +x Upscale.desktop
  gio set Upscale.desktop metadata::trusted true 2>/dev/null || true
  cp Upscale.desktop "$(xdg-user-dir DESKTOP)/" 2>/dev/null || true
  cp Upscale.desktop ~/.local/share/applications/ 2>/dev/null || true
  echo "Icône 1-clic installée (Bureau + menu)."
fi
python3 -m py_compile app.py && echo "app.py OK" && python3 app.py --list-gpu

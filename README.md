# Upscale IA locale — RX 6600

Glisse un rush, récupère une version upscalée. 100% local, sans compte, sans cloud.
Testé sur : Ryzen 5 5600G + **RX 6600 8 Go (Vulkan RADV)** + 32 Go RAM + ffmpeg 6.1.

## Installer depuis GitHub (nouvelle machine)

```bash
git clone <ton-repo> Upscale && cd Upscale
./setup.sh --bureau   # télécharge IA (~56 Mo) + crée l'icône 1-clic
./lancer.sh
```

Sans git : télécharge le ZIP du repo, décompresse, puis :
```bash
chmod +x setup.sh lancer.sh && ./setup.sh --bureau
```
Prérequis : `python3` + `ffmpeg` (`sudo apt install ffmpeg`). GPU AMD/NVIDIA
recommandé (Vulkan) — sinon mode `--fast` (CPU).

## Ça fait quoi

- `720p/1080p → 1080p/4K` avec Real-ESRGAN (NCNN Vulkan, utilise ta RX 6600)
- Interface web drag-drop `http://localhost:8000` avec progression + aperçu
- CLI pour scripts / CutFlow
- Fallback ffmpeg (lanczos + netteté) si IA trop lente ou binaire absent

## Lancer en 1 clic (recommandé)

Double-clique **Upscale** sur le Bureau (ou cherche Upscale dans le menu Zorin),
ou double-clique `lancer.sh` dans ce dossier. Ça ouvre le navigateur tout seul.
Ferme le terminal pour arrêter.

## Lancer en terminal

```bash
cd Upscale
python3 app.py --serve
# ouvre http://localhost:8000
```

CLI :

```bash
# rapide, recommandé vidéo
python3 app.py -i mon_rush.mp4 -s 2 -m anime
# HQ photo (x4 seul, lent)
python3 app.py -i photo.mp4 -s 4 -m general
# sans IA (instantané)
python3 app.py -i mon_rush.mp4 -s 2 --fast
# qualité max (plus lent, meilleurs détails)
python3 app.py -i mon_rush.mp4 -s 2 --hq
# vérifie GPU
python3 app.py --list-gpu
```

Sorties dans `output/` : `nom_x2_anime.mp4` (H.264 + audio copié).

## Modes Vite / HQ

- **Vite (défaut)** : frames JPG, IA directe, encodage GPU `h264_vaapi` (repli x264 auto).
  Réglages Vulkan mesurés les plus rapides sur RX 6600 (`-g 0 -t 256 -j 2:4:4`).
- **HQ (`--hq` ou sélecteur web)** : frames PNG sans perte, IA x4 puis descente
  lanczos à x2 (x2 uniquement, modèle anime), encodage `x264 slow crf16`.
  ~2-3x plus lent sur 1080p, détails plus fins et moins de flicker. Sortie `*_hq.mp4`.

## Modèles

| Choix UI | Fichier NCNN | Vitesse RX 6600 | Usage |
|---|---|---|---|
| `anime` (défaut) | `realesr-animevideov3` x2/x3/x4, 1.2 Mo | ~8 fps en 480p, ~3-5 fps en 1080p | vidéo, tous rushs |
| `general` | `realesrgan-x4plus` x4, 32 Mo | lent | photo, HQ |
| `anime-hq` | `realesrgan-x4plus-anime` x4 | lent | animé HQ |

`general` / `anime-hq` forcent x4 (le modèle ne sait faire que ça).

## GPU en priorité (RX 6600 verrouillée)

- IA : Vulkan forcé sur la RX 6600 (`-g 0`, jamais llvmpipe/CPU). Sans GPU détecté : erreur claire au lieu d'un calcul CPU silencieux de plusieurs heures.
- Encodage : `h264_vaapi` GPU d'abord, repli x264 auto. Même le fallback `--fast` encode en GPU.
- Chaque job affiche son moteur : `[vulkan:RX 6600+vaapi-gpu]`, badge 🟢GPU/🔴CPU dans l'UI.
- `python3 app.py --list-gpu` : rapport Vulkan + VA-API.
- Limite connue de ce driver (Mesa RADV) : `scale_vaapi` plante → le redimensionnement du fallback reste en CPU (négligeable, l'IA pèse 50-100x plus).

## Temps réels mesurés ici

- 320x240 3s (45 frames) → x2 en **2s** (~23 fps)
- 640x480 5s (150 frames) → x2 en **18s** (~8.5 fps)
- Règle : 1 min 1080p30 → **10-30 min** selon x2/x4. Un seul job à la fois (8 Go VRAM).

## Fichiers

- `app.py` — tout : pipeline + serveur web (stdlib uniquement, zéro `pip install`)
- `index.html` — UI drag-drop
- `bin/realesrgan-ncnn-vulkan` — binaire Vulkan (Real-ESRGAN v0.2.0 + pack 20220424)
- `models/` — 3 modèles NCNN
- `input/` `output/` `jobs/` — entrées, résultats, états JSON

## Limites V1 (honnêtes)

- Image par image : léger flicker possible sur aplats, pas de cohérence temporelle (un BasicVSR ferait mieux mais demande PyTorch ROCm, galère sur AMD).
- JPG q2 en temporaire : ~1-2 Go/min en 1080p, nettoyé après chaque job.
- Fichiers > 4 Go refusés, 1 job à la fois, tuilage auto `-t 0`.
- Si image noire en sortie : mets à jour Mesa/Vulkan, ou relance avec `--fast`.

## Dépannage

```bash
ffmpeg -version   # requis ≥ 5
./bin/realesrgan-ncnn-vulkan -h   # doit lister "AMD Radeon RX 6600"
python3 app.py --list-gpu
```

Refaire l'install binaire + modèles si effacés :

```bash
curl -L -o /tmp/n.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip -o /tmp/n.zip -d /tmp/n && cp /tmp/n/realesrgan-ncnn-vulkan bin/ && cp /tmp/n/models/* models/
```

Licences : Real-ESRGAN (BSD-3), NCNN (BSD-3), ce code V1 : même esprit, usage perso.

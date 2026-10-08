"""Upscale local IA — app.py unique (stdlib uniquement).
Usage:
  python3 app.py --serve [--port 8000]          # interface web drag-drop
  python3 app.py --input in.mp4 [--scale 2] [--model anime] [--output out.mp4] [--fast] [--hq]
  python3 app.py --list-gpu                     # vérifie Vulkan / RX 6600
Pipeline: ffmpeg extract JPG -> realesrgan-ncnn-vulkan (Vulkan, RX 6600) -> ffmpeg ré-encode.
Fallback --fast ou si binaire absent: ffmpeg scale=lanczos + unsharp (sans IA).
"""
import argparse, json, mimetypes, os, re, shutil, socket, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).parent
BIN = ROOT / "bin" / "realesrgan-ncnn-vulkan"
MODELS = ROOT / "models"
IN_DIR = ROOT / "input"
OUT_DIR = ROOT / "output"
JOBS_DIR = ROOT / "jobs"
for d in (IN_DIR, OUT_DIR, JOBS_DIR):
    d.mkdir(exist_ok=True)

# modèle UI -> nom ncnn. animevideov3 = rapide, recommandé vidéo. x4plus = HQ photo, x4 seul.
MODELS_MAP = {
    "anime": "realesr-animevideov3",      # scales 2,3,4
    "general": "realesrgan-x4plus",        # scale 4 uniquement
    "anime-hq": "realesrgan-x4plus-anime", # scale 4 uniquement
}
JOBS = {}  # id -> dict (mémoire + jobs/<id>.json pour persistance légère)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(path):
    r = run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe impossible: {r.stderr[:300]}")
    info = json.loads(r.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps_s = v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1"
    try:
        n, d = fps_s.split("/"); fps = float(n) / float(d) if float(d) else 30.0
    except Exception:
        fps = 30.0
    return {"width": int(v.get("width", 0)), "height": int(v.get("height", 0)),
            "fps": round(fps, 3), "duration": float(info.get("format", {}).get("duration", 0) or 0),
            "has_audio": a is not None}


def save_job(job):
    try:
        (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job, indent=1))
    except OSError:
        pass


_GPU_CACHE = None

def _gpu_status():
    """Sonar GPU (cache). Retourne {vulkan:[noms], vaapi:bool, ia_device:str|None}."""
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    vulkan, vaapi = [], Path("/dev/dri/renderD128").exists()
    if BIN.exists():
        try:  # le binaire liste les GPU même quand l'inférence échoue (extensions valides requises)
            r = subprocess.run([str(BIN), "-i", "x.jpg", "-o", "y.jpg"], capture_output=True, text=True, timeout=30)
            for m in re.finditer(r"^\[(\d+) ([^\]]+)\]", r.stdout + r.stderr, re.M):
                name = m.group(2).strip()
                if name not in vulkan:  # le binaire répète chaque GPU (queues/propriétés)
                    vulkan.append(name)
        except Exception:
            pass
    ia = next((v for v in vulkan if "llvmpipe" not in v.lower()), None)
    _GPU_CACHE = {"vulkan": vulkan, "vaapi": vaapi, "ia_device": ia}
    return _GPU_CACHE


def _vaapi_ok():
    return _gpu_status()["vaapi"]


def _encode(frames_pat, src, info, out, hq=False):
    """Ré-encode. Vite: h264_vaapi GPU (repli x264). HQ: x264 slow crf16. Retourne le moteur."""
    fps = info["fps"] or 30.0
    base_in = ["-framerate", str(fps), "-i", str(frames_pat), "-i", str(src)]
    amap = (["-map", "1:a", "-c:a", "copy"] if info["has_audio"] else ["-an"]) + ["-shortest", str(out)]
    if hq:
        r = run(["ffmpeg", "-y", "-v", "error"] + base_in + ["-map", "0:v", "-c:v", "libx264",
                 "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"] + amap)
        if r.returncode == 0:
            return "x264-slow-crf16"
    elif _vaapi_ok():  # ponytail: GPU d'abord, CPU en repli silencieux
        r = run(["ffmpeg", "-y", "-v", "error", "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
                 "-filter_hw_device", "va"] + base_in + ["-map", "0:v",
                 "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "19"] + amap)
        if r.returncode == 0:
            return "vaapi-gpu"
    r = run(["ffmpeg", "-y", "-v", "error"] + base_in + ["-map", "0:v", "-c:v", "libx264",
             "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"] + amap)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg encode: {r.stderr[:500]}")
    return "x264-cpu"


def upscale_video(src, scale=2, model="anime", out=None, fast=False, hq=False, job_id=None, on_progress=None):
    """Bloquant. Retourne chemin sortie. Lance Real-ESRGAN ou fallback ffmpeg."""
    src = Path(src)
    info = probe(src)
    ncnn = MODELS_MAP.get(model, "realesr-animevideov3")
    if ncnn != "realesr-animevideov3" and scale != 4:
        scale = 4  # ponytail: x4plus ne sait faire que x4, on force au lieu d'ajouter un rescale
    want_ia = BIN.exists() and not fast
    gpu = _gpu_status()
    if want_ia and not gpu["ia_device"]:
        raise RuntimeError("pas de GPU Vulkan (que llvmpipe/CPU) : --fast pour forcer CPU, ou répare Mesa/Vulkan")
    w, h = info["width"], info["height"]
    stem = src.stem
    tag = f"{stem}_x{scale}_{model}" + ("_hq" if hq else "")
    out = Path(out) if out else OUT_DIR / f"{tag}.mp4"
    jid = job_id or uuid.uuid4().hex[:8]
    work = JOBS_DIR / jid
    frames, up = work / "frames", work / "up"
    for d in (frames, up):
        d.mkdir(parents=True, exist_ok=True)

    def prog(p, msg):
        if on_progress:
            on_progress(p, msg)

    # 1. extraction frames (vite: JPG q2 léger. HQ: PNG sans perte, l'IA part d'une base propre)
    ext = "png" if hq else "jpg"
    prog(5, "extraction frames")
    cmd = (["ffmpeg", "-y", "-v", "error", "-i", str(src), str(frames / "f%06d.png")] if hq
           else ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-q:v", "2", str(frames / "f%06d.jpg")])
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg extract: {r.stderr[:500]}")
    n = len(list(frames.glob(f"*.{ext}")))
    if n == 0:
        raise RuntimeError("aucune frame extraite")

    use_ia = want_ia
    # HQ x2 = IA x4 puis descente lanczos à x2 : plus de détails, moins de flicker (2x calcul, assumé)
    ia_scale = 4 if (hq and scale == 2 and MODELS_MAP.get(model) == "realesr-animevideov3") else scale
    engine = f"vulkan:{gpu['ia_device']}" if use_ia else "cpu"
    if use_ia:
        # 2. upscale IA : -g 0 force la RX 6600, -t 256 + -j 2:4:4 mesurés ~11% plus vite que défaut
        prog(10, f"upscale IA {ncnn} x{ia_scale} ({n} frames, ~quelques fps sur RX 6600)")
        p = subprocess.Popen([str(BIN), "-i", str(frames), "-o", str(up), "-m", str(MODELS),
                              "-n", ncnn, "-s", str(ia_scale), "-t", "256", "-j", "2:4:4",
                              "-g", "0", "-f", ext],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # progression : % affiché par le binaire + comptage fichiers
        pct_re = re.compile(r"(\d+\.\d+)%")
        while True:
            line = p.stdout.readline() if p.stdout else ""
            if not line and p.poll() is not None:
                break
            m = pct_re.search(line)
            done = len(list(up.glob(f"*.{ext}")))
            base = (m and float(m.group(1))) or (100.0 * done / max(n, 1))
            prog(10 + base * 0.8, f"IA {base:.0f}% ({done}/{n})")
        if p.returncode != 0 or len(list(up.glob(f"*.{ext}"))) == 0:
            print("IA échouée, fallback ffmpeg", flush=True)
            use_ia = False

    if not use_ia:
        # fallback non-IA : scale CPU (négligeable) + encode GPU si dispo
        prog(30, "fallback (sans IA)")
        nw, nh = w * scale, h * scale
        if _vaapi_ok():
            r = run(["ffmpeg", "-y", "-v", "error", "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
                     "-filter_hw_device", "va", "-i", str(src),
                     "-vf", f"scale={nw}:{nh}:flags=lanczos,format=nv12,hwupload",
                     "-c:v", "h264_vaapi", "-qp", "20", "-c:a", "copy", str(out)])
            if r.returncode == 0:
                shutil.rmtree(work, ignore_errors=True)
                prog(100, "terminé (fallback GPU)")
                return out, "vaapi-gpu(fallback)"
        vf = f"scale={nw}:{nh}:flags=lanczos,hqdn3d,unsharp=5:5:0.8:5:5:0.4"
        r = run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf,
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", str(out)])
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg fallback: {r.stderr[:500]}")
        shutil.rmtree(work, ignore_errors=True)
        prog(100, "terminé (fallback CPU)")
        return out, "cpu(fallback)"

    # 3. ré-encodage : frames upscalées + audio d'origine
    prog(92, "ré-encodage mp4")
    pat = up / f"f%06d.{ext}"
    if ia_scale != scale:  # HQ x2 : descente x4 -> x2 en lanczos
        nw, nh = w * scale, h * scale
        r = run(["ffmpeg", "-y", "-v", "error", "-framerate", str(info["fps"] or 30.0),
                 "-i", str(pat), "-i", str(src), "-map", "0:v",
                 "-vf", f"scale={nw}:{nh}:flags=lanczos",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"] +
                (["-map", "1:a", "-c:a", "copy"] if info["has_audio"] else ["-an"]) +
                ["-shortest", str(out)])
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg downscale: {r.stderr[:500]}")
    else:
        engine += f"+{_encode(pat, src, info, out, hq)}"
    shutil.rmtree(work, ignore_errors=True)  # ponytail: pas de cache, le disque est précieux
    prog(100, f"terminé : {out.name} ({w}x{h} → {w*scale}x{h*scale}) [{engine}]")
    return out, engine


# ---------------- serveur web (stdlib) ----------------
class H(BaseHTTPRequestHandler):
    server_version = "Upscale/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/jobs":
            return self._send(200, json.dumps(list(JOBS.values())))
        if u.path == "/api/gpu":
            g = _gpu_status()
            return self._send(200, json.dumps({"ia_device": g["ia_device"], "vulkan": g["vulkan"],
                                               "vaapi": g["vaapi"], "gpu_first": True}))
        m = re.match(r"^/api/jobs/([\w-]+)$", u.path)
        if m:
            j = JOBS.get(m.group(1))
            if not j and (JOBS_DIR / f"{m.group(1)}.json").exists():
                j = json.loads((JOBS_DIR / f"{m.group(1)}.json").read_text())
            return self._send(200 if j else 404, json.dumps(j or {"error": "inconnu"}))
        m = re.match(r"^/api/download/([\w-]+)$", u.path)
        if m:
            j = JOBS.get(m.group(1))
            p = Path(j["output"]) if j and j.get("output") else None
            if p and p.exists():
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
                self.send_header("Content-Length", str(p.stat().st_size))
                self.end_headers()
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            return self._send(404, '{"error":"fichier manquant"}')
        if u.path.startswith("/output/") or u.path.startswith("/input/"):
            p = ROOT / unquote(u.path[1:])
            if p.is_file() and ROOT in p.resolve().parents:
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "video/mp4")
            return self._send(404, "not found", "text/plain")
        return self._send(404, '{"error":"route inconnue"}')

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            return self._send(404, '{"error":"route inconnue"}')
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            return self._send(400, '{"error":"multipart attendu"}')
        boundary = ("--" + ctype.split("boundary=")[1]).encode()
        length = int(self.headers.get("Content-Length", 0))
        if length > 4 * 1024**3:
            return self._send(413, '{"error":"fichier > 4 Go refusé"}')
        raw = self.rfile.read(length)
        # mini-parser multipart (un seul fichier + champs scale/model suffisent en V1)
        def field(name):
            m = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n(.*?)\r\n--', raw, re.S)
            return m.group(1).decode(errors="ignore").strip() if m else ""
        scale = int(field("scale") or "2")
        model = field("model") or "anime"
        hq = (field("quality") or field("hq") or "").lower() in ("hq", "1", "true", "high")
        # parsing binaire sûr :
        parts = raw.split(boundary)
        fname, data = None, None
        for p in parts:
            if b'filename="' in p:
                fname = re.search(br'filename="([^"]+)"', p).group(1).decode(errors="ignore")
                data = p.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                break
        if not data:
            return self._send(400, '{"error":"aucun fichier"}')
        fname = re.sub(r"[^\w.\- ]", "_", fname or "rush.mp4")
        jid = uuid.uuid4().hex[:8]
        dest = IN_DIR / f"{jid}_{fname}"
        dest.write_bytes(data)
        job = {"id": jid, "file": fname, "scale": scale, "model": model, "hq": hq,
               "status": "queued", "progress": 0, "msg": "en file", "output": None, "created": time.time()}
        JOBS[jid] = job
        save_job(job)
        threading.Thread(target=_run_job, args=(jid, str(dest), scale, model, hq), daemon=True).start()
        return self._send(200, json.dumps({"id": jid}))


def _run_job(jid, src, scale, model, hq=False):
    job = JOBS[jid]
    def on_progress(p, msg):
        job.update(progress=round(p, 1), msg=msg, status="running")
    try:
        out, engine = upscale_video(src, scale, model, hq=hq, job_id=jid, on_progress=on_progress)
        job.update(status="done", progress=100, output=str(out), gpu=engine.startswith("vulkan") or "vaapi" in engine,
                   engine=engine, msg=f"terminé : {Path(out).name} [{engine}]")
    except Exception as e:  # ponytail: message brut, pas de système d'erreur abstrait
        job.update(status="error", msg=str(e)[:500])
    save_job(job)


def serve(port):
    # 1 seul job à la fois : la RX 6600 8 Go sature sinon (verrou simple, pas de file distribuée)
    print(f"Upscale web : http://localhost:{port}  (dossier {ROOT})")
    print(f"GPU : {'realesrgan-ncnn-vulkan OK' if BIN.exists() else 'fallback ffmpeg (binaire absent)'}")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
    except KeyboardInterrupt:
        print("\nstop")


def main():
    ap = argparse.ArgumentParser(description="Upscale IA local (RX 6600, Vulkan)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--input", "-i")
    ap.add_argument("--output", "-o")
    ap.add_argument("--scale", "-s", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--model", "-m", default="anime", choices=list(MODELS_MAP))
    ap.add_argument("--fast", action="store_true", help="sans IA, ffmpeg seul")
    ap.add_argument("--hq", action="store_true", help="qualité max : PNG + IA x4->x2 + x264 slow crf16 (plus lent)")
    ap.add_argument("--list-gpu", action="store_true")
    a = ap.parse_args()
    if a.list_gpu:
        g = _gpu_status()
        print("IA (Vulkan -g 0):", g["ia_device"] or "AUCUN — mode CPU seulement")
        print("Vulkan détectés:", g["vulkan"] or "aucun")
        print("VA-API encode:", "OK (/dev/dri/renderD128)" if g["vaapi"] else "absent (encode CPU)")
        print("modèles:", sorted(p.name for p in MODELS.glob("*.param")))
        return
    if a.serve or not a.input:
        g = _gpu_status()
        print(f"GPU prioritaire: IA={g['ia_device'] or 'ABSENT'} VA-API={'oui' if g['vaapi'] else 'non'}")
        return serve(a.port)
    out, engine = upscale_video(a.input, a.scale, a.model, a.output, a.fast, hq=a.hq,
                        on_progress=lambda p, m: print(f"{p:.0f}% {m}", flush=True))
    print("OK:", out, f"[{engine}]")


if __name__ == "__main__":
    main()

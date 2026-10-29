"""Local dashboard and SUMO worker. Launch: python scripts/dashboard.py"""
from __future__ import annotations
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import math
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind.evaluation import CONTROLLERS, ROOT, SCENARIOS, export_results, latest_comparisons, load_results, run_experiment
from flowmind.runtime import RuntimeControl, RULE_LIMITS
from flowmind.emission_rl import MODEL_DIR
from chatbot.simulation_explainer import SimulationExplainer
from chatbot.gemini_client import GeminiError, load_environment
from chatbot.simulation_context import build_context

lock = threading.Lock()
stop = threading.Event()
job = {"running": False, "status": "Prêt", "error": None, "run_ids": []}
runtime_control = None
explainer = SimulationExplainer()


def latest_live(started_at=0):
    """Use metadata so a newly starting run never displays the previous run."""
    files = sorted((ROOT / "runs").glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            if path.stat().st_mtime < started_at:
                continue
            meta = json.loads(path.read_text(encoding="utf-8"))
            live_path = path.parent / "live.json"
            if live_path.exists():
                meta.update(json.loads(live_path.read_text(encoding="utf-8")))
            return {k: v for k, v in meta.items() if k not in ("config", "protocol")}
        except (OSError, ValueError):
            continue
    return None


def read_series(run_id, start=0, end=float("inf")):
    if not math.isfinite(start) or start < 0 or end < start or math.isnan(end):
        raise ValueError("Période invalide")
    directory = (ROOT / "runs" / run_id).resolve()
    if directory.parent != (ROOT / "runs").resolve():
        raise ValueError("Identifiant d’essai invalide")
    points = []
    meta = {}
    try:
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        pass
    try:
        with (directory / "timeseries.csv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(line for line in source if line.endswith("\n")):
                try:
                    values = {k: float(row[k]) for k in (
                        "elapsed_s", "vehicles", "halting", "queue_mean_lane",
                        "speed_sum_m_s", "co2_mg_s")}
                    if not all(math.isfinite(v) for v in values.values()):
                        continue
                    values["speed_kmh"] = 3.6 * values.pop("speed_sum_m_s") / values["vehicles"] if values["vehicles"] else None
                    values["co2_g_s"] = values.pop("co2_mg_s") / 1000
                    for key in ("waiting_mean_active_s", "queue_max_lane", "nox_mg_s", "pmx_mg_s", "fuel_mg_s"):
                        value = row.get(key)
                        values[key] = float(value) if value not in (None, "") else None
                    values["controller"] = row.get("controller") or meta.get("controller", "unknown")
                    values["control_revision"] = int(row.get("control_revision", 0))
                    if start <= values["elapsed_s"] <= end:
                        points.append(values)
                except (KeyError, ValueError, TypeError):
                    continue  # The writer may still be completing the final CSV row.
    except FileNotFoundError:
        pass
    count = len(points)
    if count > 600:
        points = [points[round(i * (count - 1) / 599)] for i in range(600)]
    return {"run_id": run_id, "sample_count": count, "points": points}


def worker(options):
    try:
        for seed in options["seeds"]:
            for controller in options["controllers"]:
                if stop.is_set():
                    break
                with lock:
                    job["status"] = f"{controller} · seed {seed}"
                runtime_args = {"runtime_control": runtime_control} if options.get("interactive") else {}
                result = run_experiment(controller, options["scenario"], options["seconds"], seed,
                                        options["gui"], options["delay"], stop, **runtime_args)
                with lock:
                    job["run_ids"].append(result["run_id"])
            if stop.is_set():
                break
        with lock:
            job["status"] = "Interrompu" if stop.is_set() else "Terminé"
    except Exception as exc:
        with lock:
            job.update(status="Échec", error=str(exc))
    finally:
        try:
            export_results()
        except Exception as exc:
            with lock:
                job.update(status="Échec de l’export", error=str(exc))
        with lock:
            job["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/series":
            run_id = parse_qs(urlsplit(self.path).query).get("run_id", [""])[0]
            try:
                query = parse_qs(urlsplit(self.path).query)
                self.send_json(read_series(run_id, float(query.get("start", [0])[0]), float(query.get("end", ["inf"])[0])))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if path == "/api/state":
            rows = load_results()
            with lock:
                state = dict(job)
            live = latest_live(state.get("started_at", 0))
            model = ROOT / "runs/models/flow_ppo.zip"
            emission_manifest = None
            if (MODEL_DIR / "best.zip").is_file() and (MODEL_DIR / "manifest.json").is_file():
                emission_manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
            self.send_json({"job": state, "live": live, "results": rows,
                            "comparisons": latest_comparisons(rows), "model_available": model.is_file(),
                            "control": runtime_control.snapshot() if runtime_control else None,
                            "rule_defaults": {k:v[2] for k,v in RULE_LIMITS.items()},
                            "emission_model": emission_manifest})
            return
        if path in ("/", "/simulation", "/history"):
            target = ROOT / "dashboard/index.html"
        elif path == "/chat":
            target = ROOT / "dashboard/chat.html"
        elif path.startswith("/files/"):
            target = (ROOT / "runs" / path[len("/files/"):]).resolve()
            if not target.is_relative_to((ROOT / "runs").resolve()) or target.suffix not in (".csv", ".json", ".xml", ".png"):
                self.send_json({"error": "Accès refusé"}, 403)
                return
        else:
            self.send_json({"error": "Page introuvable"}, 404)
            return
        if not target.is_file():
            self.send_json({"error": "Fichier introuvable"}, 404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(str(target))[0] or "application/octet-stream") + ("; charset=utf-8" if target.suffix == ".html" else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        global runtime_control
        # Drain a bounded request before any early reply. Closing a Windows TCP
        # connection with an unread body can reset it and hide the JSON error.
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = 32768 if self.path == "/api/chat" else 8192
            if not 0 < size < limit:
                raise ValueError("Requête invalide")
            self.connection.settimeout(10)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Requête incomplète")
        except (ValueError, OSError):
            self.send_json({"error": "Requête invalide ou incomplète"}, 400)
            return
        # Requests must come from this local page; no remote cross-origin controls.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send_json({"error": "Origine refusée"}, 403)
            return
        if self.headers.get("X-FlowMind") != "dashboard":
            self.send_json({"error": "En-tête de contrôle requis"}, 403)
            return
        if self.path == "/api/stop":
            stop.set()
            self.send_json({"ok": True})
            return
        if self.path not in ("/api/start", "/api/control", "/api/chat"):
            self.send_json({"error": "Action inconnue"}, 404)
            return
        try:
            options = json.loads(body)
            if not isinstance(options, dict):
                raise ValueError("Objet JSON requis")
            if self.path == "/api/chat":
                # Any saved simulation can be explained; build_context checks
                # that the run exists inside runs/. Without one, use the latest.
                run_id = options.get("run_id")
                if not run_id:
                    with lock:
                        started_at = job.get("started_at", 0)
                    live = latest_live(started_at)
                    if not live:
                        self.send_json({"error": "Aucune simulation disponible : lancez une simulation et attendez les premières mesures."}, 409)
                        return
                    run_id = live.get("run_id")
                context = build_context(ROOT, run_id)
                answer = explainer.explain(context, options.get("question"), options.get("history"))
                self.send_json(answer)
                return
            if self.path == "/api/control":
                with lock:
                    if not job["running"] or runtime_control is None:
                        self.send_json({"error": "Aucune simulation interactive en cours"}, 409)
                        return
                    if options["method"] == "emission_model":
                        live = latest_live(job.get("started_at",0))
                        requested = RuntimeControl("emission_model", runtime_control.snapshot()["rules"] | options.get("rules", {})).snapshot()["rules"]
                        self.check_emission_model(live["scenario"] if live else "3x3grid", live.get("seed") if live else None, requested)
                    control = runtime_control.update(options["method"], options.get("rules", {}))
                self.send_json({"ok": True, "control": control})
                return
            interactive = options.get("interactive", False)
            next_control = None
            if interactive:
                next_control = RuntimeControl(options["method"], options.get("rules", {}))
                options["controllers"] = [options["method"]]
                options["seeds"] = [int(options["seed"])]
            controllers = list(dict.fromkeys(options["controllers"]))
            seeds = list(dict.fromkeys(int(s) for s in options["seeds"]))
            seconds = int(options["seconds"])
            if not controllers or any(c not in CONTROLLERS for c in controllers):
                raise ValueError("Choisissez au moins un contrôleur disponible")
            if not seeds or len(seeds) > 20 or any(s < 0 or s > 2147483647 for s in seeds):
                raise ValueError("Choisissez de 1 à 20 seeds positives")
            if options["scenario"] not in SCENARIOS or not 10 <= seconds <= 86400:
                raise ValueError("Scénario ou durée invalide (10 à 86400 s)")
            if "flow_model" in controllers and not (ROOT / "runs/models/flow_ppo.zip").is_file():
                raise ValueError("Modèle entraîné absent")
            if "emission_model" in controllers:
                for test_seed in seeds:
                    self.check_emission_model(options["scenario"], test_seed, next_control.snapshot()["rules"] if next_control else None)
            delay = int(options.get("delay", 100))
            if not 0 <= delay <= 1000:
                raise ValueError("Délai invalide")
            options.update(controllers=controllers, seeds=seeds, seconds=seconds, delay=delay, gui=bool(options.get("gui", False)))
            with lock:
                if job["running"]:
                    self.send_json({"error": "Une simulation est déjà en cours"}, 409)
                    return
                job.update(running=True, status="Démarrage…", error=None, run_ids=[], started_at=time.time())
                runtime_control = next_control
                stop.clear()
            threading.Thread(target=worker, args=(options,), daemon=True).start()
            self.send_json({"ok": True})
        except GeminiError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except OSError:
            self.send_json({"error": "Les données de simulation sont temporairement indisponibles."}, 503)
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def check_emission_model(self, scenario, seed=None, rules=None):
        if not (MODEL_DIR/"best.zip").is_file() or not (MODEL_DIR/"manifest.json").is_file():
            raise ValueError("Modèle émissions absent : lancez scripts/train_emissions.py")
        manifest=json.loads((MODEL_DIR/"manifest.json").read_text(encoding="utf-8"))
        if manifest["scenario"]!=scenario:
            raise ValueError("Le modèle émissions n’est pas compatible avec ce scénario")
        if seed in manifest["train_seeds"] + manifest["validation_seeds"]:
            raise ValueError("Choisir une seed de test distincte de l’entraînement/validation")
        if rules is not None:
            from flowmind.emission_rl import inference_safety
            inference_safety(manifest, rules)


def main():
    load_environment(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"FlowMind dashboard: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop.set()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

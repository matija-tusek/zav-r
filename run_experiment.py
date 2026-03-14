"""
run_experiment.py — pokreće N ponavljanja GA.py paralelno s istim seedom

Sve instance koriste iste parametre (seed, GA postavke) a razlike u
rezultatima dolaze od inherentne stohastičnosti (PyBullet, GA operatori).

Struktura outputa:
    results/
    eksperiment_1/
        run_1/
            run_1.json
            run_1_summary.png
            run_1_best_creature.json
            run_1_progress.png
        run_2/
            run_2.json
            run_2_summary.png
            run_2_best_creature.json
            run_2_progress.png
        ...
    logs/
        <experiment_name>/
            run_1.log
            run_2.log
            ...
    best_creature_<experiment_name>_run_1.json
    best_creature_<experiment_name>_run_2.json
    ...

Pokretanje:
    python run_experiment.py
    python run_experiment.py --name moj_eksperiment --runs 5 --seed 0
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# ── Konfiguracija ─────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "rewardBased500Steps1Seed"   # naziv eksperimenta (= naziv foldera)
N_RUNS          = 5                 # broj ponavljanja
SEED            = 0                 # isti seed za sva ponavljanja
GA_SCRIPT       = "GA.py"

# ── CLI override (opcionalno) ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name",  type=str, default=EXPERIMENT_NAME)
parser.add_argument("--runs",  type=int, default=N_RUNS)
parser.add_argument("--seed",  type=int, default=SEED)
args = parser.parse_args()

EXPERIMENT_NAME = args.name
N_RUNS          = args.runs
SEED            = args.seed


def launch(run_number: int) -> subprocess.Popen:
    """Pokreni jednu instancu GA.py kao zaseban proces."""
    run_name = f"run_{run_number}"

    cmd = [
        sys.executable, GA_SCRIPT,
        "--experiment", EXPERIMENT_NAME,
        "--run",        run_name,
        "--seed",       str(SEED),
    ]

    log_dir = os.path.join("logs", EXPERIMENT_NAME)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{run_number}.log")
    log_file = open(log_path, "w", buffering=1)

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    print(f"  [run_{run_number}]  PID={proc.pid}  log → {log_path}")
    return proc, log_path


def main():
    print("=" * 60)
    print(f"Eksperiment:  {EXPERIMENT_NAME}")
    print(f"Ponavljanja:  {N_RUNS}")
    print(f"Seed:         {SEED}  (isti za sva ponavljanja)")
    print(f"Pokrenuto:    {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    # Osiguraj da results/<experiment_name>/ folder postoji
    results_dir = os.path.join("results", EXPERIMENT_NAME)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nRezultati → {results_dir}/")
    print(f"Pokrećem {N_RUNS} instanci...\n")

    procs = {}  # run_number -> (proc, log_path)

    try:
        for run in range(1, N_RUNS + 1):
            proc, log_path = launch(run)
            procs[run] = (proc, log_path)

        print(f"\nSvi procesi pokrenuti. Čekam da završe...\n")
        finished_count = 0

        while procs:
            time.sleep(10)
            done = [r for r, (p, _) in procs.items() if p.poll() is not None]
            for run in done:
                proc, log_path = procs.pop(run)
                finished_count += 1
                rc = proc.returncode
                status = "✓" if rc == 0 else f"✗ (exit {rc})"
                print(f"  {status} [run_{run}]  završen  "
                      f"{finished_count}/{N_RUNS}  "
                      f"{datetime.now().strftime('%H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\nCtrl+C — zaustavljam sve procese...")
        for run, (proc, _) in procs.items():
            try:
                proc.terminate()
                print(f"  → [run_{run}] PID={proc.pid} terminiran")
            except Exception:
                pass
        sys.exit(0)

    print("\n" + "=" * 60)
    print(f"Završeno: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Rezultati: results/{EXPERIMENT_NAME}/")
    print(f"Logovi:    logs/{EXPERIMENT_NAME}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
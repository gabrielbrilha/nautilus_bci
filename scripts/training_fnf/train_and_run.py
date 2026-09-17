import os
import sys
import subprocess
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

_current_dir = Path(__file__).resolve().parent

def find_fnf_dirs():
    candidates = []
    submod = _current_dir.parent.parent / "fnf_prot" / "python"
    if (submod / "main.py").exists():
        candidates.append(submod.resolve())
    cwd = Path.cwd().resolve()
    for c in [ cwd / "fnf_prot" / "python" ]:
        if (c / "main.py").exists():
            candidates.append(c.resolve())
    return list(set(candidates))

def get_primary_fnf_dir():
    dirs = find_fnf_dirs()
    if not dirs:
        submod = _current_dir.parent.parent / "fnf_prot" / "python"
        return submod if submod.exists() else (_current_dir.parent.parent.parent / "fnf_prot" / "python")
    return dirs[0]

def resolve_pipeline_python(td_dir: Path) -> Path:
    if "VIRTUAL_ENV" in os.environ:
        v_py = Path(os.environ["VIRTUAL_ENV"]) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if v_py.exists():
            return v_py
    return Path(sys.executable)

if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
for td_d in find_fnf_dirs():
    if str(td_d) not in sys.path:
        sys.path.insert(0, str(td_d))

import dataset
import algorithms

def train_model(
    dataset_name="bids_fnf",
    sub_id="02",
    session_ids=None,
    alg_key="riemann_logreg",
    n_splits=5,
    C_val=0.1,
    output_path=None,
    report_path=None,
    custom_tag=""
):
    if (session_ids is None or len(session_ids) == 0):
        available = dataset.get_available_sessions(dataset_name, sub_id)
        if not available:
            raise ValueError(f"No sessions found for sub-{sub_id} in {dataset_name}")
        session_ids = [s[0] for s in available]

    sub_clean = sub_id.replace("sub-", "")
    print("=" * 80)
    print(" BCI FRIDAY NIGHT FUNKIN: RHYTHM MODEL TRAINING STUDIO ".center(80, "="))
    print("=" * 80)
    print(f"[*] Dataset         : {dataset_name}")
    print(f"[*] Subject         : sub-{sub_clean}")
    print(f"[*] Game Sessions   : {session_ids}")
    print(f"[*] Algorithm       : {alg_key.upper()}")
    print(f"[*] CV Folds        : {n_splits}")
    print("=" * 80)

    print("\n---> Loading and Preprocessing EEG Sessions...")
    X_im, y_im, stats, meta_df = dataset.load_dataset_sessions(
        dataset_name,
        sub_clean,
        session_ids,
        sfreq=250.0,
        win_len_s=0.7,
        progress_callback=lambda msg, _: print(f"    {msg}")
    )

    print(f"\n[+] Total Pooled Dataset: {len(y_im)} trials | Epoch Shape: {X_im.shape}")

    candidates = list(algorithms.ALGORITHMS.keys()) if alg_key == "all" else [alg_key]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for c_key in candidates:
        c_info = algorithms.ALGORITHMS[c_key]
        print(f"\n[*] Evaluating [{c_info['name']}]...")

        accs, f1s, kappas = [], [], []
        oof_preds = np.zeros_like(y_im)

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_im, y_im)):
            if "riemann" in c_key or "fbcsp" in c_key:
                clf = algorithms.create_classifier(c_key, C=C_val) if hasattr(algorithms.ALGORITHMS[c_key]['class'], 'C') else algorithms.create_classifier(c_key)
            else:
                clf = algorithms.create_classifier(c_key)

            clf.fit(X_im[tr_idx], y_im[tr_idx])
            p_te = clf.predict(X_im[te_idx])
            oof_preds[te_idx] = p_te

            accs.append(accuracy_score(y_im[te_idx], p_te))
            f1s.append(f1_score(y_im[te_idx], p_te, average="macro"))
            kappas.append(cohen_kappa_score(y_im[te_idx], p_te))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        results[c_key] = {
            'name': c_info['name'],
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_f1': float(np.mean(f1s)),
            'mean_kappa': float(np.mean(kappas)),
            'cm': confusion_matrix(y_im, oof_preds).tolist()
        }
        print(f"    • Mean Accuracy : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}% (Chance: 25.00%)")

    best_key = max(results.keys(), key=lambda k: results[k]['mean_acc'])
    best_res = results[best_key]

    print("\n" + "=" * 80)
    print(f" SELECTED BEST ALGORITHM: {best_res['name']} ({best_res['mean_acc']*100:.2f}%) ".center(80, "="))
    print("=" * 80)

    print(f"\n[*] Fitting final model on full dataset ({len(y_im)} trials)...")
    if "riemann" in best_key or "fbcsp" in best_key:
        final_model = algorithms.create_classifier(best_key, C=C_val) if hasattr(algorithms.ALGORITHMS[best_key]['class'], 'C') else algorithms.create_classifier(best_key)
    else:
        final_model = algorithms.create_classifier(best_key)

    final_model.fit(X_im, y_im)
    self_acc = float(accuracy_score(y_im, final_model.predict(X_im)))
    print(f"[+] Final Model Fit Complete. Self-Accuracy: {self_acc*100:.2f}%")

    models_dir = _current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if custom_tag.strip():
        tag = custom_tag.strip()
    else:
        n_ses = len(session_ids) if session_ids else 0
        tag = f"sub{sub_clean}_{best_key}_{n_ses}ses"

    output_path = Path(output_path) if output_path else models_dir / f"rhythm_model_{tag}.joblib"
    report_path = Path(report_path) if report_path else models_dir / f"rhythm_report_{tag}.json"

    export_dict = {
        'model': final_model,
        'model_name': f"{best_key.upper()}_4Class_FNF",
        'algorithm_key': best_key,
        'algorithm_name': best_res['name'],
        'dataset_folder': dataset_name,
        'subject': sub_clean,
        'sessions': session_ids,
        'classes': ['Left', 'Right', 'Up', 'Down'],
        'element_mapping': algorithms.DIRECTION_NAMES,
        'sfreq': 250.0,
        'window_size_sec': 0.7,
        'n_trials': len(y_im),
        'metrics': {
            'cv_accuracy_mean': best_res['mean_acc'],
            'cv_accuracy_std': best_res['std_acc'],
            'cv_f1_macro': best_res['mean_f1'],
            'cv_cohen_kappa': best_res['mean_kappa'],
            'self_accuracy': self_acc
        },
        'confusion_matrix': best_res['cm']
    }

    joblib.dump(export_dict, output_path)
    print(f"\n[+] Exported joblib model: {output_path}")

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(export_dict, f, indent=2, default=str)
    print(f"[+] Exported JSON report: {report_path}")

    return final_model, output_path, export_dict


def launch_pipeline(
    model_path,
    source="simulator",
    mode="bids_replay",
    sub="02",
    ses="05",
    threshold=0.40,
    auto_send=True,
    interactive=False
):
    td_dir = get_primary_fnf_dir()
    main_py = td_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Could not locate main.py at {main_py}.")

    print("\n" + "=" * 80)
    print(" INITIATING REAL-TIME BCI PIPELINE (GODOT BRIDGE) ".center(80, "="))
    print(f"[*] FNF Dir           : {td_dir}")
    print("=" * 80)

    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))
    if str(td_dir) not in sys.path:
        sys.path.insert(0, str(td_dir))
    os.environ["PYTHONPATH"] = str(_current_dir) + os.pathsep + str(td_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

    venv_py = resolve_pipeline_python(td_dir)

    try:
        from main import run_pipeline
        run_pipeline(
            source=source,
            mode=mode,
            auto_send=auto_send,
            threshold=threshold,
            interactive=interactive,
            model_path=str(model_path),
            sub=sub,
            ses=ses
        )
    except Exception as e:
        import subprocess
        cmd = [
            str(venv_py), str(main_py),
            "--source", source,
            "--mode", mode,
            "--threshold", str(threshold),
            "--sub", sub,
            "--ses", ses,
            "--model", str(model_path)
        ]
        if auto_send: cmd.append("--auto-send")
        if interactive: cmd.append("--interactive")

        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(_current_dir) + os.pathsep + str(td_dir) + os.pathsep + run_env.get("PYTHONPATH", "")
        subprocess.run(cmd, cwd=str(td_dir), env=run_env)


def main():
    parser = argparse.ArgumentParser(description="BCI FNF Rhythm Training Studio & Real-Time Launcher")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--dataset", type=str, default="bids_fnf")
    parser.add_argument("--sub", type=str, default="02")
    parser.add_argument("--ses", type=str, default="all")
    parser.add_argument("--alg", type=str, default="riemann_logreg", choices=["all"] + list(algorithms.ALGORITHMS.keys()))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--source", type=str, default="simulator", choices=["simulator", "lsl"])
    parser.add_argument("--mode", type=str, default="bids_replay", choices=["bids_replay", "synthetic"])
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--no-auto-send", dest="auto_send", action="store_false", default=True)
    parser.add_argument("--interactive", action="store_true", default=False)

    if len(sys.argv) == 1:
        from gui import launch_gui
        launch_gui()
        return

    args = parser.parse_args()

    if args.gui:
        from gui import launch_gui
        launch_gui()
        return

    if args.model and args.run_pipeline:
        launch_pipeline(args.model, args.source, args.mode, args.sub, args.ses if args.ses != "all" else "05", args.threshold, args.auto_send, args.interactive)
        return

    if args.ses.lower() == "all":
        available = dataset.get_available_sessions(args.dataset, args.sub)
        sessions = [s[0] for s in available]
    else:
        sessions = [s.strip().replace("ses-", "") for s in args.ses.split(",") if s.strip()]

    final_model, model_path, report = train_model(
        dataset_name=args.dataset, sub_id=args.sub, session_ids=sessions,
        alg_key=args.alg, n_splits=args.cv, C_val=args.C,
        output_path=args.output, report_path=args.report, custom_tag=args.tag
    )

    if args.run_pipeline:
        launch_pipeline(model_path, args.source, args.mode, args.sub, sessions[-1] if sessions else "05", args.threshold, args.auto_send, args.interactive)

if __name__ == "__main__":
    main()

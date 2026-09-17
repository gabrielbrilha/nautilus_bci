import os
import sys
import json
import time
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

import shutil
import signal

_current_dir = Path(__file__).resolve().parent

def find_fnf_dirs():
    candidates = []
    submod = _current_dir.parent.parent / "fnf_prot" / "python"
    if (submod / "main.py").exists():
        candidates.append(submod.resolve())
    cwd = Path.cwd().resolve()
    for c in [cwd / "fnf_prot" / "python"]:
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

def find_terminal_command(workdir: Path, cmd: list) -> tuple:
    if sys.platform == "win32":
        return ("windows_console", cmd)
    return (None, cmd)

_td_python_dir = get_primary_fnf_dir()
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
for td_d in find_fnf_dirs():
    if str(td_d) not in sys.path:
        sys.path.insert(0, str(td_d))

import dataset
import algorithms

class TrainingStudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI Friday Night Funkin — Rhythm Training & Real-Time Studio")
        self.root.geometry("1180x820")

        self.bg_color = "#0d1117"
        self.panel_bg = "#161b22"
        self.card_bg = "#21262d"
        self.card_border = "#30363d"
        self.accent_color = "#58a6ff"
        self.fg_color = "#ffffff"
        self.muted_fg = "#c9d1d9"
        self.success_color = "#3fb950"
        self.metric_title_color = "#79c0ff"
        self.input_bg = "#0d1117"

        self.root.configure(bg=self.bg_color)
        self._setup_styles()

        self.available_datasets = dataset.get_available_datasets()
        self.selected_dataset = tk.StringVar()
        self.selected_subject = tk.StringVar()
        self.session_vars = {}
        self.selected_algorithm = tk.StringVar()
        self.cv_folds = tk.IntVar(value=5)
        self.reg_c = tk.DoubleVar(value=0.1)
        self.model_name_var = tk.StringVar()

        self.rt_source = tk.StringVar(value="simulator")
        self.rt_mode = tk.StringVar(value="bids_replay")
        self.rt_threshold = tk.DoubleVar(value=0.40)
        self.rt_autosend = tk.BooleanVar(value=True)
        self.rt_interactive = tk.BooleanVar(value=True)
        self.last_trained_model_path = None
        self.realtime_process = None

        self.available_models = {}
        self.selected_model_var = tk.StringVar()
        self.log_queue = queue.Queue()

        self._build_ui()
        self._init_defaults()
        self._refresh_available_models()
        self._start_log_consumer()

    def _setup_styles(self):
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=self.panel_bg, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#238636", foreground="#ffffff")
        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background="#1f6feb", foreground="#ffffff")
        self.style.configure("Danger.TButton", font=("Segoe UI", 9, "bold"), background="#da3633", foreground="#ffffff")

    def _build_ui(self):
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        left_frame = tk.Frame(main_container, bg=self.panel_bg, bd=1, relief=tk.SOLID)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6), pady=0)
        self._build_left_controls(left_frame)

        right_frame = tk.Frame(main_container, bg=self.panel_bg, bd=1, relief=tk.SOLID)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=0)
        self._build_right_dashboard(right_frame)

    def _build_left_controls(self, parent):
        ds_group = ttk.LabelFrame(parent, text=" 1. Dataset Selection ", padding=10)
        ds_group.pack(fill=tk.X, padx=8, pady=6)

        self.ds_combo = ttk.Combobox(ds_group, textvariable=self.selected_dataset, values=list(self.available_datasets.keys()), state="readonly")
        self.ds_combo.pack(fill=tk.X, pady=(2, 6))
        self.ds_combo.bind("<<ComboboxSelected>>", self._on_dataset_change)

        sub_group = ttk.LabelFrame(parent, text=" 2. Subject & Session ", padding=10)
        sub_group.pack(fill=tk.X, padx=8, pady=6)

        self.sub_combo = ttk.Combobox(sub_group, textvariable=self.selected_subject, state="readonly")
        self.sub_combo.pack(fill=tk.X)
        self.sub_combo.bind("<<ComboboxSelected>>", self._on_subject_change)

        self.sessions_box = tk.Frame(sub_group, bg=self.card_bg)
        self.sessions_box.pack(fill=tk.X, pady=4)

        btn_row = ttk.Frame(sub_group)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Select All", command=self._select_all_sessions).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Clear All", command=self._clear_all_sessions).pack(side=tk.LEFT)

        alg_group = ttk.LabelFrame(parent, text=" 3. Algorithm ", padding=10)
        alg_group.pack(fill=tk.X, padx=8, pady=6)

        self.alg_display_map = {"all": "Benchmark All"}
        self.alg_display_map.update({k: v['name'] for k, v in algorithms.ALGORITHMS.items()})
        self.display_to_key = {v: k for k, v in self.alg_display_map.items()}

        self.alg_combo = ttk.Combobox(alg_group, values=list(self.alg_display_map.values()), state="readonly")
        self.alg_combo.pack(fill=tk.X)

        self.train_btn = ttk.Button(parent, text="TRAIN MODEL", style="Accent.TButton", command=self._start_training_thread)
        self.train_btn.pack(fill=tk.X, padx=8, pady=10)

    def _build_right_dashboard(self, parent):
        metrics_group = ttk.LabelFrame(parent, text=" Evaluation Metrics ")
        metrics_group.pack(fill=tk.X, padx=10, pady=8)
        self.progress_lbl = ttk.Label(metrics_group, text="Ready")
        self.progress_lbl.pack(anchor="w")

        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, bg="#0d1117", fg="#f0f6fc", height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10)

        rt_group = ttk.LabelFrame(parent, text=" Real-Time Bridge ")
        rt_group.pack(fill=tk.X, padx=10, pady=10)

        self.model_combo = ttk.Combobox(rt_group, textvariable=self.selected_model_var, state="readonly")
        self.model_combo.pack(fill=tk.X)

        self.launch_rt_btn = ttk.Button(rt_group, text="LAUNCH PIPELINE", style="Success.TButton", command=self._launch_realtime_pipeline)
        self.launch_rt_btn.pack(fill=tk.X, pady=5)

    def _init_defaults(self):
        datasets = list(self.available_datasets.keys())
        if datasets:
            self.selected_dataset.set(datasets[0])
            self._on_dataset_change()
        self.alg_combo.current(0)

    def _on_dataset_change(self, event=None):
        ds_key = self.selected_dataset.get()
        subjects = dataset.get_available_subjects(ds_key)
        self.sub_combo['values'] = subjects
        if subjects:
            self.selected_subject.set(subjects[0])
            self._on_subject_change()

    def _on_subject_change(self, event=None):
        ds_key = self.selected_dataset.get()
        sub_id = self.selected_subject.get()
        sessions = dataset.get_available_sessions(ds_key, sub_id)

        for widget in self.sessions_box.winfo_children():
            widget.destroy()
        self.session_vars.clear()

        for ses, mode in sessions:
            var = tk.BooleanVar(value=True)
            self.session_vars[ses] = var
            cb = tk.Checkbutton(self.sessions_box, text=f"ses-{ses} [{mode}]", variable=var, bg=self.card_bg, fg="#ffffff")
            cb.pack(anchor="w")

    def _select_all_sessions(self):
        for v in self.session_vars.values(): v.set(True)

    def _clear_all_sessions(self):
        for v in self.session_vars.values(): v.set(False)

    def _log(self, text, end="\n"):
        self.log_queue.put(text + end)

    def _start_log_consumer(self):
        def check_logs():
            while not self.log_queue.empty():
                self.log_text.insert(tk.END, self.log_queue.get_nowait())
                self.log_text.see(tk.END)
            self.root.after(80, check_logs)
        self.root.after(80, check_logs)

    def _start_training_thread(self):
        selected_sessions = [ses for ses, var in self.session_vars.items() if var.get()]
        if not selected_sessions: return
        worker = threading.Thread(
            target=self._run_training_worker,
            args=(self.selected_dataset.get(), self.selected_subject.get(), selected_sessions, self.display_to_key.get(self.alg_combo.get(), "riemann_logreg")),
            daemon=True
        )
        worker.start()

    def _run_training_worker(self, ds_key, sub_id, session_ids, alg_key):
        self._log(f"[*] Training {alg_key} on {sub_id} sessions {session_ids}")
        X_im, y_im, stats, meta_df = dataset.load_dataset_sessions(ds_key, sub_id, session_ids, progress_callback=lambda m, f: self._log(m))
        clf = algorithms.create_classifier(alg_key) if alg_key != "all" else algorithms.create_classifier("riemann_logreg")
        clf.fit(X_im, y_im)

        models_dir = _current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / f"rhythm_model_sub{sub_id}_{alg_key}.joblib"
        joblib.dump({'model': clf}, model_file)
        self._log(f"[+] Saved to {model_file}")
        self._refresh_available_models()

    def _refresh_available_models(self):
        self.available_models.clear()
        models_dir = _current_dir / "models"
        if models_dir.exists():
            for f in sorted(models_dir.glob("*.joblib")):
                self.available_models[f.name] = f
        self.model_combo['values'] = list(self.available_models.keys())
        if self.available_models: self.selected_model_var.set(list(self.available_models.keys())[-1])

    def _launch_realtime_pipeline(self):
        model_path = self.available_models.get(self.selected_model_var.get())
        if not model_path: return

        td_dir = get_primary_fnf_dir()
        main_py = td_dir / "main.py"
        venv_py = resolve_pipeline_python(td_dir)

        cmd = [str(venv_py), str(main_py), "--model", str(model_path)]
        self._log(f"[*] Launching {cmd}")
        self.realtime_process = subprocess.Popen(cmd, cwd=str(td_dir), stdout=subprocess.PIPE, text=True)
        threading.Thread(target=lambda p: [self._log(f"[RT] {l.rstrip()}") for l in iter(p.stdout.readline, '')], args=(self.realtime_process,), daemon=True).start()

def launch_gui():
    root = tk.Tk()
    app = TrainingStudioGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()

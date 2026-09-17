import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids

_curr_dir = Path(__file__).resolve().parent
_bids_root = _curr_dir.parent / "bids" / "bids_fnf"

STANDARD_32 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2',
    'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]

DIRECTIONS = ['Left', 'Right', 'Up', 'Down']
DIRECTION_MAP = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}


def resolve_dataset_path(dataset_name_or_path):
    p = Path(dataset_name_or_path)
    if p.exists():
        return p
    cand = _curr_dir.parent / "bids" / dataset_name_or_path
    if cand.exists():
        return cand
    cand = _curr_dir.parent.parent / "bids" / dataset_name_or_path
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Could not resolve BIDS directory: {dataset_name_or_path}")


def get_available_datasets():
    bids_dir = _curr_dir.parent / "bids"
    if not bids_dir.exists():
        return {"bids_fnf": "Default FNF BIDS"}

    found = {}
    for d in bids_dir.iterdir():
        if d.is_dir() and "fnf" in d.name:
            found[d.name] = d.name

    if not found:
        return {"bids_fnf": "Default FNF BIDS"}
    return found


def get_available_subjects(dataset_path_str):
    ds_path = resolve_dataset_path(dataset_path_str)
    subjects = []
    for d in sorted(ds_path.glob("sub-*")):
        if d.is_dir():
            subjects.append(d.name)
    return subjects


def get_available_sessions(dataset_path_str, sub_id):
    ds_path = resolve_dataset_path(dataset_path_str)
    sub_clean = sub_id.replace("sub-", "")
    sub_dir = ds_path / f"sub-{sub_clean}"
    if not sub_dir.exists():
        return []

    sessions = []
    for d in sorted(sub_dir.glob("ses-*")):
        if d.is_dir():
            ses_clean = d.name.replace("ses-", "")

            eeg_dir = d / "eeg"
            if eeg_dir.exists():
                mode = "unknown"
                for f in eeg_dir.glob("*.vhdr"):
                    if "task-leftright" in f.name:
                        mode = "Mind"
                    elif "task-me" in f.name:
                        mode = "Movement"
                    break
                sessions.append((ses_clean, mode))
            else:
                sessions.append((ses_clean, "unknown"))
    return sessions


def get_session_trial_preview(dataset_path_str, sub_id, ses_id):
    try:
        ds_path = resolve_dataset_path(dataset_path_str)
        sub_clean = sub_id.replace("sub-", "")

        sub_dir = ds_path / f"sub-{sub_clean}" / f"ses-{ses_id}" / "eeg"
        task = "leftright"
        for f in sub_dir.glob("*.vhdr"):
            if "task-me" in f.name:
                task = "me"
            break

        bp = BIDSPath(subject=sub_clean, session=ses_id, task=task, datatype="eeg", root=ds_path)
        raw = read_raw_bids(bp, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        counts = {}
        total = 0
        for direction in DIRECTIONS:
            target_name = f'Arrow_{direction}_HitZone'
            code = None
            for k, v in event_id.items():
                if target_name in k:
                    code = v
                    break
            if code is not None:
                d_events = events[events[:, 2] == code]
                counts[direction] = len(d_events)
                total += len(d_events)

        mode = "Mind" if task == "leftright" else "Movement"
        return total, counts, mode
    except Exception as e:
        return 0, {}, "unknown"


def load_single_session_raw(bids_root, sub, ses, l_freq=8.0, h_freq=30.0):
    bids_root = str(bids_root)
    sub_clean = sub.replace("sub-", "")
    ses_clean = ses.replace("ses-", "")

    ds_path = resolve_dataset_path(bids_root)
    sub_dir = ds_path / f"sub-{sub_clean}" / f"ses-{ses_clean}" / "eeg"
    task = "leftright"
    if sub_dir.exists():
        for f in sub_dir.glob("*.vhdr"):
            if "task-me" in f.name:
                task = "me"
            break

    bp = BIDSPath(subject=sub_clean, session=ses_clean, task=task, datatype="eeg", root=ds_path)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()

    mapping = {raw.ch_names[i]: STANDARD_32[i] for i in range(min(32, len(raw.ch_names)))}
    if len(raw.ch_names) > 32:
        raw.set_channel_types({raw.ch_names[32]: 'misc'})
    raw.rename_channels(mapping)
    raw.pick('eeg')

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    raw_filt = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw_filt.notch_filter(freqs=50.0, verbose=False)
    raw_filt.set_eeg_reference('average', projection=False, verbose=False)

    mode = "Mind" if task == "leftright" else "Movement"
    return raw_filt, mode


def load_dataset_sessions(
    dataset_name_or_path,
    sub_id,
    session_ids,
    sfreq=250.0,
    win_len_s=0.7,
    progress_callback=None
):
    sub_clean = sub_id.replace("sub-", "")

    all_X = []
    all_y = []
    all_meta = []
    session_stats = {}

    ds_path = resolve_dataset_path(dataset_name_or_path)
    total_ses = len(session_ids)

    tmin, tmax = -0.1, 0.6

    for idx, ses in enumerate(session_ids):
        ses_clean = ses.replace("ses-", "")
        msg = f"Loading session ses-{ses_clean} ({idx + 1}/{total_ses})..."
        if progress_callback:
            progress_callback(msg, (idx / total_ses) * 0.9)

        raw_filt, mode = load_single_session_raw(
            str(ds_path), sub_clean, ses_clean
        )

        events, event_id = mne.events_from_annotations(raw_filt, verbose=False)

        X_sess = []
        y_sess = []
        counts = {d: 0 for d in DIRECTIONS}

        for direction in DIRECTIONS:
            target_name = f'Arrow_{direction}_HitZone'
            code = None
            for k, v in event_id.items():
                if target_name in k:
                    code = v
                    break
            if code is not None:
                d_events = events[events[:, 2] == code]
                if len(d_events) > 0:
                    ep = mne.Epochs(
                        raw_filt,
                        d_events,
                        event_id={f'Target_{direction}': code},
                        tmin=tmin,
                        tmax=tmax,
                        baseline=(-0.1, 0.0),
                        preload=True,
                        verbose=False
                    )
                    X_data = ep.get_data()
                    for i in range(len(X_data)):
                        X_sess.append(X_data[i])
                        y_sess.append(DIRECTION_MAP[direction])
                        all_meta.append({
                            'session': ses_clean,
                            'direction': direction,
                            'class_id': DIRECTION_MAP[direction],
                            'mode': mode
                        })
                    counts[direction] = len(d_events)

        if len(y_sess) == 0:
            print(f"[Warning] No mental rhythm trials extracted from ses-{ses_clean}.")
            continue

        session_stats[ses_clean] = {
            'trials': len(y_sess),
            'breakdown': counts
        }

        all_X.append(np.array(X_sess))
        all_y.append(np.array(y_sess))

    if not all_X:
        raise RuntimeError(f"No valid trials found across sessions {session_ids} for sub-{sub_clean}.")

    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.concatenate(all_y, axis=0)
    meta_df = pd.DataFrame(all_meta)

    if progress_callback:
        progress_callback(f"Successfully pooled {len(y_pooled)} total trials across {len(session_stats)} source blocks.", 1.0)

    return X_pooled, y_pooled, session_stats, meta_df

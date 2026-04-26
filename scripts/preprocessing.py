import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d


# ============================================================
# Basic utilities
# ============================================================

def read_real_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def read_vr_csv(csv_path):
    """
    VR csv:
    line 1: ArmSpan,149.15
    line 2: header
    line 3+: data
    """
    with open(csv_path, "r") as f:
        first_line = f.readline().strip().split(",")

    meta = {}
    if len(first_line) >= 2 and first_line[0].lower() == "armspan":
        meta["ArmSpan"] = float(first_line[1])

    df = pd.read_csv(csv_path, skiprows=1)
    return df, meta


def wrap_deg_to_rad(x):
    """
    Convert degree angle in [-180, 180] to rad in [-pi, pi].
    """
    return np.deg2rad(((x + 180) % 360) - 180)


def range_norm_minus_pi_pi(x):
    """
    Normalize angle already in [-pi, pi] to approximately [-1, 1].
    """
    return x / np.pi


def zscore(x, eps=1e-8):
    return (x - np.nanmean(x)) / (np.nanstd(x) + eps)


def butter_bandpass_filter(x, fs, lowcut=0.2, highcut=8.0, order=4):
    """
    x: (T,) or (T, D)
    """
    x = np.asarray(x, dtype=float)

    nyq = 0.5 * fs
    low = lowcut / nyq if lowcut is not None else None
    high = highcut / nyq if highcut is not None else None

    if low is not None and high is not None:
        b, a = butter(order, [low, high], btype="band")
    elif low is not None:
        b, a = butter(order, low, btype="high")
    elif high is not None:
        b, a = butter(order, high, btype="low")
    else:
        return x

    if len(x) < max(len(a), len(b)) * 3:
        return x

    return filtfilt(b, a, x, axis=0)


def interpolate_to_fs(df, time_col, feature_cols, fs):
    """
    Linear interpolation to ideal sampling rate.
    """
    t = df[time_col].values.astype(float)
    X = df[feature_cols].values.astype(float)

    valid = np.isfinite(t)
    valid &= np.all(np.isfinite(X), axis=1)

    t = t[valid]
    X = X[valid]

    order = np.argsort(t)
    t = t[order]
    X = X[order]

    # remove duplicate timestamps
    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]
    X = X[unique_idx]

    if len(t) < 3:
        return None, None

    t_new = np.arange(t[0], t[-1], 1.0 / fs)

    f = interp1d(
        t,
        X,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    X_new = f(t_new)
    return t_new, X_new


def add_velocity_features(t, X, feature_cols, vel_mode_dict=None):
    """
    Construct velocity by numerical derivative.

    Returns:
        X_aug: original + velocity
        cols_aug
    """
    if vel_mode_dict is None:
        vel_mode_dict = {}

    dt = np.gradient(t)
    vel = np.gradient(X, axis=0) / dt[:, None]

    vel_cols = []
    for col in feature_cols:
        vel_cols.append(f"{col}_vel")

    X_aug = np.concatenate([X, vel], axis=1)
    cols_aug = list(feature_cols) + vel_cols

    return X_aug, cols_aug


def apply_norm(X, cols, norm_dict):
    """
    norm_dict example:
    {
        "pitch": "range_pi",
        "roll": "range_pi",
        "yaw": "range_pi",
        "HeadX": "zscore",
        "HeadPitch": "range_pi",
    }
    """
    X = X.copy()

    for j, col in enumerate(cols):
        base_col = col.replace("_vel", "")

        method = norm_dict.get(col, norm_dict.get(base_col, None))

        if method is None:
            continue

        if method in ["range_pi", "range_norm_pi", "minus_pi_pi"]:
            X[:, j] = range_norm_minus_pi_pi(X[:, j])

        elif method == "zscore":
            X[:, j] = zscore(X[:, j])

        elif method == "none":
            pass

        else:
            raise ValueError(f"Unknown norm method: {method} for column {col}")

    return X


def bin_one_rep(t, X, cols, start_t, end_t, n_bins=60):
    """
    Bin one rep into fixed T bins.

    pos/rot columns: mean
    velocity columns: last by default
    """
    mask = (t >= start_t) & (t <= end_t)
    tt = t[mask]
    XX = X[mask]

    if len(tt) < 2:
        return None

    edges = np.linspace(start_t, end_t, n_bins + 1)
    out = np.full((n_bins, X.shape[1]), np.nan)

    for b in range(n_bins):
        m = (tt >= edges[b]) & (tt < edges[b + 1])
        if b == n_bins - 1:
            m = (tt >= edges[b]) & (tt <= edges[b + 1])

        if not np.any(m):
            continue

        xb = XX[m]

        for d, col in enumerate(cols):
            if col.endswith("_vel"):
                out[b, d] = xb[-1, d]
            else:
                out[b, d] = np.nanmean(xb[:, d])

    # fill empty bins by interpolation
    for d in range(out.shape[1]):
        y = out[:, d]
        good = np.isfinite(y)
        if good.sum() >= 2:
            out[:, d] = np.interp(np.arange(n_bins), np.where(good)[0], y[good])
        elif good.sum() == 1:
            out[:, d] = y[good][0]
        else:
            return None

    return out

def find_matching_head_stream(
    head_streams,
    start_t,
    end_t,
    min_overlap_ratio=0.8,
):
    """
    Find the head stream that best covers the hand-detected rep interval.

    Returns None if no head stream sufficiently overlaps.
    """
    if len(head_streams) == 0:
        return None

    rep_dur = end_t - start_t
    if rep_dur <= 0:
        return None

    best_stream = None
    best_overlap = 0.0

    for hs in head_streams:
        ht0 = hs["t"][0]
        ht1 = hs["t"][-1]

        overlap_start = max(start_t, ht0)
        overlap_end = min(end_t, ht1)
        overlap = max(0.0, overlap_end - overlap_start)

        overlap_ratio = overlap / rep_dur

        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_stream = hs

    if best_overlap >= min_overlap_ratio:
        return best_stream

    return None

def get_time_seconds(df, source):
    if source == "real":
        # Apple Watch UTC timestamp, usually nanoseconds
        t = df["time"].values.astype(float)

        # convert ns -> seconds if needed
        if np.nanmedian(t) > 1e12:
            t = t / 1e9

        return t

    elif source == "vr":
        return df["Timestamp"].values.astype(float)

    else:
        raise ValueError(source)


# ============================================================
# Rep segmentation: ONLY rest-gated swim segment method
# ============================================================

def get_envelope_from_left_hand(t, X, cols, hand_feature_priority=None, smooth_sec=0.15):
    """
    Build left-hand motion envelope.

    Default:
    - prefer velocity columns
    - envelope = L2 norm of selected columns
    - then rolling mean smoothing
    """
    if hand_feature_priority is None:
        hand_feature_priority = [
            "pitch_vel", "roll_vel", "yaw_vel",
            "LeftX_norm_vel", "LeftY_norm_vel", "LeftZ_norm_vel",
            "accelerationX", "accelerationY", "accelerationZ",
            "pitch", "roll", "yaw",
            "LeftX_norm", "LeftY_norm", "LeftZ_norm",
        ]

    use_cols = [c for c in hand_feature_priority if c in cols]
    if len(use_cols) == 0:
        use_cols = cols

    idx = [cols.index(c) for c in use_cols]
    env = np.linalg.norm(X[:, idx], axis=1)

    fs_est = 1.0 / np.nanmedian(np.diff(t))
    win = max(3, int(smooth_sec * fs_est))
    if win % 2 == 0:
        win += 1

    env_s = (
        pd.Series(env)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )

    return env_s


def detect_swim_segments_by_rest(
    t,
    x,
    fs,
    win_sec=1.0,
    rest_std_thr=5.0,
    min_rest_sec=5.0,
    min_swim_sec=8.0,
):
    """
    Detect swim segments using rest periods.

    Logic:
    - compute 1s-window std of rest signal
    - std < rest_std_thr => rest
    - consecutive rest >= min_rest_sec => true rest block
    - gaps between rest blocks >= min_swim_sec => swim segment

    Parameters
    ----------
    t : array
        Time in seconds. Can be Unix seconds.
    x : array
        Signal for rest detection, e.g. roll in degrees.
    fs : float
        Sampling rate.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]

    w = max(1, int(win_sec * fs))

    if len(x) < w * 2:
        return [], np.array([]), []

    stds = np.array([
        np.nanstd(x[i:i + w])
        for i in range(0, len(x) - w, w)
    ])

    is_rest = stds < rest_std_thr

    if len(is_rest) == 0:
        return [], stds, []

    changes = np.diff(is_rest.astype(int))

    r_starts = np.concatenate([
        [0] if is_rest[0] else [],
        np.where(changes == 1)[0] + 1
    ])

    r_ends = np.concatenate([
        np.where(changes == -1)[0] + 1,
        [len(is_rest)] if is_rest[-1] else []
    ])

    min_rest_bins = int(np.ceil(min_rest_sec / win_sec))

    rests = [
        (int(s), int(e))
        for s, e in zip(r_starts, r_ends)
        if (e - s) >= min_rest_bins
    ]

    swim_segs = []
    prev = 0

    for rs, re in rests:
        dur_sec = (rs - prev) * win_sec

        if dur_sec >= min_swim_sec:
            start_idx = int(prev * w)
            end_idx = min(int(rs * w), len(t) - 1)
            swim_segs.append((t[start_idx], t[end_idx]))

        prev = re

    if prev * w < len(t):
        dur_sec = (len(stds) - prev) * win_sec

        if dur_sec >= min_swim_sec:
            start_idx = int(prev * w)
            end_idx = len(t) - 1
            swim_segs.append((t[start_idx], t[end_idx]))

    return swim_segs, stds, rests


def segment_reps_in_swim_segments(
    t,
    env,
    swim_segs,
    fs,
    valid_rep_time_range=(0.5, 2.0),
    peak_height=None,
    peak_height_k=1.5,
    peak_prominence_k=0.8,
    smooth_sec=0.35,
    n_continuous_windows=2,
):
    """
    Detect stroke reps only inside detected swim segments.

    Stroke peak rule:
    - peak distance >= valid_rep_time_range[0]
    - peak height must exceed absolute peak_height if provided
    - otherwise use robust threshold: median + k * MAD
    """

    t = np.asarray(t, dtype=float)
    env = np.asarray(env, dtype=float)

    win = max(3, int(smooth_sec * fs))
    if win % 2 == 0:
        win += 1

    env_s = (
        pd.Series(env)
        .rolling(win, center=True, min_periods=1)
        .median()
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )

    rep_intervals = []

    for seg_st, seg_ed in swim_segs:
        mask = (t >= seg_st) & (t <= seg_ed)

        if mask.sum() < int(valid_rep_time_range[0] * fs * 2):
            continue

        t_seg = t[mask]
        env_seg = env_s[mask]

        med = np.nanmedian(env_seg)
        mad = np.nanmedian(np.abs(env_seg - med)) + 1e-8
        robust_sd = 1.4826 * mad

        if peak_height is None:
            height = med + peak_height_k * robust_sd
        else:
            height = peak_height

        prominence = max(peak_prominence_k * robust_sd, 1e-8)
        min_dist = max(1, int(valid_rep_time_range[0] * fs))

        peaks, _ = find_peaks(
            env_seg,
            height=height,
            prominence=prominence,
            distance=min_dist,
        )

        peaks = merge_close_peaks(
            peaks,
            env_seg,
            fs=fs,
            min_peak_gap_sec=1.0,
        )

        if len(peaks) < 2:
            continue

        peak_t = t_seg[peaks]

        boundaries = []
        boundaries.append(max(seg_st, peak_t[0] - (peak_t[1] - peak_t[0]) / 2))

        for i in range(len(peak_t) - 1):
            boundaries.append((peak_t[i] + peak_t[i + 1]) / 2)

        boundaries.append(min(seg_ed, peak_t[-1] + (peak_t[-1] - peak_t[-2]) / 2))

        for i in range(len(boundaries) - 1):
            st, ed = boundaries[i], boundaries[i + 1]
            dur = ed - st

            if valid_rep_time_range[0] <= dur <= valid_rep_time_range[1]:
                rep_intervals.append((st, ed))

    rep_intervals = filter_reps_by_continuity(
        rep_intervals,
        valid_rep_time_range=valid_rep_time_range,
        n_continuous_windows=n_continuous_windows,
        max_gap_sec=valid_rep_time_range[1],
    )

    return rep_intervals, env_s

def filter_reps_by_continuity(
    rep_intervals,
    valid_rep_time_range=(0.5, 2.0),
    n_continuous_windows=3,
    max_gap_sec=None,
):
    """
    Drop isolated rep intervals.

    Keep a rep only if it belongs to a continuous cluster with at least
    n_continuous_windows reps.

    max_gap_sec:
        max allowed gap between consecutive reps.
        If None, use valid_rep_time_range[1].
    """
    if len(rep_intervals) == 0:
        return []

    if max_gap_sec is None:
        max_gap_sec = valid_rep_time_range[1]

    reps = sorted(rep_intervals, key=lambda x: x[0])

    clusters = []
    cur = [reps[0]]

    for rep in reps[1:]:
        prev_st, prev_ed = cur[-1]
        st, ed = rep

        gap = st - prev_ed

        if gap <= max_gap_sec:
            cur.append(rep)
        else:
            clusters.append(cur)
            cur = [rep]

    clusters.append(cur)

    kept = []
    for c in clusters:
        if len(c) >= n_continuous_windows:
            kept.extend(c)

    return kept


def plot_segmentation_debug(
    t,
    env,
    rep_intervals,
    swim_segs=None,
    fs=100,
    xlim=None,
    title=None,
):
    import matplotlib.pyplot as plt

    t = np.asarray(t)
    t0 = t[0]
    frame = ((t - t0) * fs).astype(int)

    plt.figure(figsize=(10, 3))
    plt.plot(frame, env, label="envelope")

    if swim_segs is not None:
        for st, ed in swim_segs:
            plt.axvspan(
                int((st - t0) * fs),
                int((ed - t0) * fs),
                alpha=0.12,
                color="lightblue"
            )

    for st, ed in rep_intervals:
        plt.axvspan(
            int((st - t0) * fs),
            int((ed - t0) * fs),
            alpha=0.25,
            color="orange"
        )

    if title is not None:
        plt.title(title)

    plt.xlabel("frame")
    plt.ylabel("envelope")

    if xlim is not None:
        plt.xlim(int(xlim[0] * fs), int(xlim[1] * fs))

    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Modality preprocessing
# ============================================================
def preprocess_one_stream(
    df,
    source,
    joint,
    feature_cols,
    fs_target,
    norm_dict,
    filter_cfg,
    construct_vel=True,
):
    df = df.copy()

    if source == "real":
        t_raw = get_time_seconds(df, source)

    elif source == "vr":
        t_raw = get_time_seconds(df, source)

        for col in ["HeadPitch", "HeadYaw", "HeadRoll"]:
            if col in df.columns:
                df[col] = wrap_deg_to_rad(df[col].values)

    else:
        raise ValueError(source)

    missing = [c for c in feature_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns in {source}-{joint}: {missing}")

    X_raw = df[feature_cols].values.astype(float)

    t, X = interpolate_array_to_fs(t_raw, X_raw, fs_target)

    if t is None:
        return None

    X = butter_bandpass_filter(
        X,
        fs=fs_target,
        lowcut=filter_cfg.get("lowcut", 0.2),
        highcut=filter_cfg.get("highcut", 8.0),
        order=filter_cfg.get("order", 4),
    )

    cols = list(feature_cols)

    if construct_vel:
        X, cols = add_velocity_features(t, X, cols)

    X = apply_norm(X, cols, norm_dict)

    return {
        "t": t,
        "X": X,
        "cols": cols,
        "source": source,
        "joint": joint,
    }


def concatenate_streams(stream_dict, order, fs=None):
    """
    Align all streams to a common overlapping time grid.
    """
    t_start = max(stream_dict[j]["t"][0] for j in order)
    t_end   = min(stream_dict[j]["t"][-1] for j in order)

    if t_end <= t_start:
        return None

    if fs is None:
        t0 = stream_dict[order[0]]["t"]
        fs = 1.0 / np.median(np.diff(t0))

    t_ref = np.arange(t_start, t_end, 1.0 / fs)

    X_all = []
    cols_all = []

    for joint in order:
        s = stream_dict[joint]
        t = s["t"]
        X = s["X"]
        cols = s["cols"]

        X_interp = np.zeros((len(t_ref), X.shape[1]))

        for d in range(X.shape[1]):
            f = interp1d(
                t,
                X[:, d],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            X_interp[:, d] = f(t_ref)

        X_all.append(X_interp)
        cols_all.extend([f"{joint}_{c}" for c in cols])

    return {
        "t": t_ref,
        "X": np.concatenate(X_all, axis=1),
        "cols": cols_all,
    }

def interpolate_array_to_fs(t, X, fs):
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)

    valid = np.isfinite(t)
    valid &= np.all(np.isfinite(X), axis=1)

    t = t[valid]
    X = X[valid]

    order = np.argsort(t)
    t = t[order]
    X = X[order]

    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]
    X = X[unique_idx]

    if len(t) < 3:
        return None, None

    t_new = np.arange(t[0], t[-1], 1.0 / fs)

    f = interp1d(
        t,
        X,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return t_new, f(t_new)

def merge_close_peaks(peaks, env_seg, fs, min_peak_gap_sec=0.9):
    if len(peaks) <= 1:
        return peaks

    min_gap = int(min_peak_gap_sec * fs)

    merged = []
    group = [peaks[0]]

    for p in peaks[1:]:
        if p - group[-1] < min_gap:
            group.append(p)
        else:
            best = group[np.argmax(env_seg[group])]
            merged.append(best)
            group = [p]

    best = group[np.argmax(env_seg[group])]
    merged.append(best)

    return np.asarray(merged, dtype=int)
# ============================================================
# File discovery
# ============================================================

def find_real_motion_files(real_dir):
    """
    real_dir has:
        hand / multiple folders / WristMotion.csv
        head / multiple folders / WristMotion.csv
    """
    hand_files = glob.glob(
        os.path.join(real_dir, "hand", "**", "WristMotion.csv"),
        recursive=True
    )
    head_files = glob.glob(
        os.path.join(real_dir, "head", "**", "WristMotion.csv"),
        recursive=True
    )

    return sorted(hand_files), sorted(head_files)


def find_baseline_real_dir(phase_dir, pid):
    return os.path.join(phase_dir, "baseline", f"sub_{pid}", "real")


def find_block_dirs(phase_dir, pid):
    pattern = os.path.join(phase_dir, "block_*", f"sub_{pid}")
    return sorted(glob.glob(pattern))


def find_vr_csvs(block_sub_dir):
    return sorted(glob.glob(os.path.join(block_sub_dir, "vr_*", "*.csv")))


def find_real_dir_in_block(block_sub_dir):
    return os.path.join(block_sub_dir, "real")


# ============================================================
# Main processing functions
# ============================================================

def process_real_session(
    real_dir,
    feature_dict,
    fs_target=100,
    norm_dict=None,
    filter_cfg=None,
    n_bins=60,
    valid_rep_time_range=(1.0, 2.5),
    debug_plot=True,
    debug_xlim=(100, 200),
    fs=100,
    peak_height=None,
):
    """
    Process one real session.

    Logic:
    - hand files are used for rep segmentation
    - head files are preprocessed independently
    - each hand-detected rep is matched to any head stream with overlapping time range
    """
    if norm_dict is None:
        norm_dict = {}
    if filter_cfg is None:
        filter_cfg = {}

    hand_files, head_files = find_real_motion_files(real_dir)

    if len(hand_files) == 0:
        return None, None

    all_reps = []
    final_cols = None

    # =====================================================
    # 1) Preprocess all head streams first
    # =====================================================
    head_streams = []

    if "head" in feature_dict["real"]:
        for head_path in head_files:
            try:
                head_df = read_real_csv(head_path)

                head_stream = preprocess_one_stream(
                    head_df,
                    source="real",
                    joint="head",
                    feature_cols=feature_dict["real"]["head"],
                    fs_target=fs_target,
                    norm_dict=norm_dict.get("real", {}).get("head", {}),
                    filter_cfg=filter_cfg,
                    construct_vel=True,
                )

                if head_stream is not None:
                    head_stream["path"] = head_path
                    head_streams.append(head_stream)

            except Exception as e:
                print(f"[WARN] Failed to preprocess head file: {head_path}")
                print(e)

    # =====================================================
    # 2) Process each hand file independently
    # =====================================================
    for hand_path in hand_files:

        try:
            hand_df = read_real_csv(hand_path)

            hand_stream = preprocess_one_stream(
                hand_df,
                source="real",
                joint="left_hand",
                feature_cols=feature_dict["real"]["left_hand"],
                fs_target=fs_target,
                norm_dict=norm_dict.get("real", {}).get("left_hand", {}),
                filter_cfg=filter_cfg,
                construct_vel=True,
            )

        except Exception as e:
            print(f"[WARN] Failed to preprocess hand file: {hand_path}")
            print(e)
            continue

        if hand_stream is None:
            continue

        hand_stream["path"] = hand_path

        # -----------------------------
        # segment reps from this hand file
        # -----------------------------
        env = get_envelope_from_left_hand(
            hand_stream["t"],
            hand_stream["X"],
            hand_stream["cols"],
        )

        # Use original roll for rest detection.
        # real roll is rad, so convert to degree.
        rest_signal = np.rad2deg(hand_df["roll"].values)

        # Align rest_signal to hand_stream time grid
        t_raw = get_time_seconds(hand_df, source="real")
        _, rest_signal_interp = interpolate_array_to_fs(
            t_raw,
            rest_signal.reshape(-1, 1),
            fs_target
        )
        rest_signal_interp = rest_signal_interp[:, 0]

        swim_segs, rest_stds, rests = detect_swim_segments_by_rest(
            t=hand_stream["t"],
            x=rest_signal_interp,
            fs=fs_target,
            win_sec=1.0,
            rest_std_thr=5.0,
            min_rest_sec=5.0,
            min_swim_sec=8.0,
        )

        rep_intervals, env_s = segment_reps_in_swim_segments(
            t=hand_stream["t"],
            env=env,
            swim_segs=swim_segs,
            fs=fs_target,
            valid_rep_time_range=valid_rep_time_range,
            peak_height=peak_height,        # or set e.g. 3.0
            peak_height_k=1.5,
            peak_prominence_k=0.8,
            smooth_sec=0.35,
        )

        if debug_plot:
            plot_segmentation_debug(
                t=hand_stream["t"],
                env=env_s,
                rep_intervals=rep_intervals,
                swim_segs=swim_segs,
                fs=fs_target,
                xlim=debug_xlim,
                title=f"Real segmentation | {hand_path}",
            )

        # =====================================================
        # 3) For each hand rep, find matching head stream by time
        # =====================================================
        for st, ed in rep_intervals:

            streams = {
                "left_hand": hand_stream
            }

            matched_head = find_matching_head_stream(
                head_streams,
                st,
                ed,
                min_overlap_ratio=0.8,
            )

            if matched_head is None:
                continue

            streams["head"] = matched_head

            order = []
            if "head" in streams:
                order.append("head")
            order.append("left_hand")

            merged = concatenate_streams(
                streams,
                order=order,
                fs=fs_target,
            )

            if merged is None:
                continue

            rep = bin_one_rep(
                merged["t"],
                merged["X"],
                merged["cols"],
                st,
                ed,
                n_bins=n_bins,
            )

            if rep is not None:
                all_reps.append(rep)
                final_cols = merged["cols"]

    if len(all_reps) == 0:
        return None, final_cols

    return np.stack(all_reps, axis=0), final_cols


def process_vr_file(
    vr_csv,
    feature_dict,
    fs_target=24,
    norm_dict=None,
    filter_cfg=None,
    n_bins=60,
    valid_rep_time_range=(1.0, 2.5),
    peak_height=None,
):
    if norm_dict is None:
        norm_dict = {}
    if filter_cfg is None:
        filter_cfg = {}

    df, meta = read_vr_csv(vr_csv)

    streams = {}

    for joint in ["head", "left_hand", "right_hand"]:
        if joint not in feature_dict["vr"]:
            continue

        stream = preprocess_one_stream(
            df,
            source="vr",
            joint=joint,
            feature_cols=feature_dict["vr"][joint],
            fs_target=fs_target,
            norm_dict=norm_dict.get("vr", {}).get(joint, {}),
            filter_cfg=filter_cfg,
            construct_vel=True,
        )

        if stream is not None:
            streams[joint] = stream

    if "left_hand" not in streams:
        return None, None, meta

    env = get_envelope_from_left_hand(
        streams["left_hand"]["t"],
        streams["left_hand"]["X"],
        streams["left_hand"]["cols"]
    )

    # For VR, use left-hand X/Y/Z movement magnitude as rest signal
    lh = streams["left_hand"]
    lh_cols = lh["cols"]

    rest_use_cols = [
        c for c in ["LeftX_norm", "LeftY_norm", "LeftZ_norm"]
        if c in lh_cols
    ]

    if len(rest_use_cols) == 0:
        rest_use_cols = [
            c for c in ["LeftX_norm_vel", "LeftY_norm_vel", "LeftZ_norm_vel"]
            if c in lh_cols
        ]

    rest_idx = [lh_cols.index(c) for c in rest_use_cols]
    rest_signal = np.linalg.norm(lh["X"][:, rest_idx], axis=1)

    swim_segs, rest_stds, rests = detect_swim_segments_by_rest(
        t=lh["t"],
        x=rest_signal,
        fs=fs_target,
        win_sec=1.0,
        rest_std_thr=0.02,       # VR normalized position threshold, may tune
        min_rest_sec=5.0,
        min_swim_sec=8.0,
    )

    rep_intervals, env_s = segment_reps_in_swim_segments(
        t=lh["t"],
        env=env,
        swim_segs=swim_segs,
        fs=fs_target,
        valid_rep_time_range=valid_rep_time_range,
        peak_height=peak_height,
        peak_height_k=1.5,
        peak_prominence_k=0.8,
        smooth_sec=0.35,
    )

    order = [j for j in ["head", "left_hand", "right_hand"] if j in streams]
    merged = concatenate_streams(streams, order=order, fs=fs_target)

    if merged is None:
        return None, None, meta

    reps = []
    for st, ed in rep_intervals:
        rep = bin_one_rep(
            merged["t"],
            merged["X"],
            merged["cols"],
            st,
            ed,
            n_bins=n_bins
        )
        if rep is not None:
            reps.append(rep)

    if len(reps) == 0:
        return None, merged["cols"], meta

    return np.stack(reps, axis=0), merged["cols"], meta


def process_swim_phase(
    data_root,
    phase_j,
    PIDS,
    feature_dict,
    norm_dict,
    fs_dict=None,
    filter_cfg=None,
    n_bins=60,
    valid_rep_time_range=(1.0, 2.5),
    peak_height = None,
):
    """
    Final returned structure:

    {
        "sub_2": {
            "meta": {"ArmSpan": ...},
            "feature_cols": {...},
            "baseline": (N,T,D),
            "block_1": {
                "vr": (N,T,D),
                "real": (N,T,D),
            },
            ...
        }
    }
    """
    if fs_dict is None:
        fs_dict = {
            "real": 100,
            "vr": 24,
        }

    if filter_cfg is None:
        filter_cfg = {
            "lowcut": 0.2,
            "highcut": 8.0,
            "order": 4,
        }

    phase_dir = os.path.join(data_root, f"phase_{phase_j}")

    out = {}

    for pid in PIDS:
        sub_key = f"sub_{pid}"

        out[sub_key] = {
            "meta": {},
            "feature_cols": {},
            "baseline": None,
        }

        # -------------------------
        # Baseline real
        # -------------------------
        baseline_real_dir = find_baseline_real_dir(phase_dir, pid)

        if os.path.exists(baseline_real_dir):
            print(baseline_real_dir)
            baseline_arr, baseline_cols = process_real_session(
                baseline_real_dir,
                feature_dict=feature_dict,
                fs_target=fs_dict["real"],
                norm_dict=norm_dict,
                filter_cfg=filter_cfg,
                n_bins=n_bins,
                valid_rep_time_range=valid_rep_time_range,
                debug_plot=True,
                peak_height=peak_height,
            )
            out[sub_key]["baseline"] = baseline_arr
            out[sub_key]["feature_cols"]["baseline"] = baseline_cols
        else:
            raise ValueError("can't find baseline folder")

        # -------------------------
        # Blocks
        # -------------------------
        block_sub_dirs = find_block_dirs(phase_dir, pid)

        for block_sub_dir in block_sub_dirs:
            block_name = os.path.basename(os.path.dirname(block_sub_dir))

            out[sub_key][block_name] = {
                "vr": None,
                "real": None,
            }

            # real
            real_dir = find_real_dir_in_block(block_sub_dir)
            if os.path.exists(real_dir):
                real_arr, real_cols = process_real_session(
                    real_dir,
                    feature_dict=feature_dict,
                    fs_target=fs_dict["real"],
                    norm_dict=norm_dict,
                    filter_cfg=filter_cfg,
                    n_bins=n_bins,
                    valid_rep_time_range=valid_rep_time_range,
                    debug_plot=True,
                    peak_height=peak_height,
                )
                out[sub_key][block_name]["real"] = real_arr
                out[sub_key]["feature_cols"][f"{block_name}_real"] = real_cols

            # vr
            vr_csvs = find_vr_csvs(block_sub_dir)

            vr_arrs = []
            vr_cols_final = None

            for vr_csv in vr_csvs:
                vr_arr, vr_cols, meta = process_vr_file(
                    vr_csv,
                    feature_dict=feature_dict,
                    fs_target=fs_dict["vr"],
                    norm_dict=norm_dict,
                    filter_cfg=filter_cfg,
                    n_bins=n_bins,
                    valid_rep_time_range=valid_rep_time_range,
                    peak_height=peak_height,
                )

                out[sub_key]["meta"].update(meta)

                if vr_arr is not None:
                    vr_arrs.append(vr_arr)
                    vr_cols_final = vr_cols

            if len(vr_arrs) > 0:
                out[sub_key][block_name]["vr"] = np.concatenate(vr_arrs, axis=0)
                out[sub_key]["feature_cols"][f"{block_name}_vr"] = vr_cols_final

    return out
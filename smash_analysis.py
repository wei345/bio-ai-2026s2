# region Import
# %%
import os
import warnings

# Suppress TensorFlow and C++ absl logging (removes the W0000 and INFO logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'

# Suppress the specific Protobuf deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
if 'HEADLESS_MODE' in os.environ:
    # For a web server, Matplotlib must be configured to run in the
    # background without trying to open GUI windows
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import IntEnum
from typing import List
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
# endregion

# region Kinematic Metrics Extraction
# %%
@dataclass
class FrameMetrics:
    trunk_speed: float = 0.0
    trunk_v: Tuple[float, float] = (0.0, 0.0) # (vx, vy)
    shoulder_speed: float = 0.0
    shoulder_v: Tuple[float, float] = (0.0, 0.0)
    elbow_speed: float = 0.0
    elbow_v: Tuple[float, float] = (0.0, 0.0)
    wrist_speed: float = 0.0
    wrist_v: Tuple[float, float] = (0.0, 0.0)
    grip_speed: float = 0.0
    grip_v: Tuple[float, float] = (0.0, 0.0)

    upper_arm_angular_speed: float = 0.0
    upper_arm_angular_deceleration: float = 0.0

    trunk_coord: Optional[Tuple[int, int]] = None
    shoulder_coord: Optional[Tuple[int, int]] = None
    elbow_coord: Optional[Tuple[int, int]] = None
    wrist_coord: Optional[Tuple[int, int]] = None
    grip_coord: Optional[Tuple[int, int]] = None
    head_coord: Optional[Tuple[int, int]] = None

def _calculate_angle(a, b, c):
    """Calculates the interior angle (in degrees) at vertex b formed by points a, b, and c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def extract_kinematic_metrics(
        input_video_path: str,
        trunk_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        shoulder_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        elbow_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        wrist_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        grip_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
        head_lm: mp.solutions.pose.PoseLandmark = mp.solutions.pose.PoseLandmark.NOSE,
        alpha: float = 0.35) -> tuple[list[FrameMetrics], float]:

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 120

    dt = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 1 for high-speed motion robustness
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    metrics = []

    trunk_state     = {"prev_pos": None, "vx": 0.0, "vy": 0.0}
    shoulder_state  = {"prev_pos": None, "vx": 0.0, "vy": 0.0}
    elbow_state     = {"prev_pos": None, "vx": 0.0, "vy": 0.0, "prev_angle": None, "omega": 0.0, "prev_omega": 0.0, "ang_accel": 0.0}
    wrist_state     = {"prev_pos": None, "vx": 0.0, "vy": 0.0}
    grip_state      = {"prev_pos": None, "vx": 0.0, "vy": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        m = FrameMetrics()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            hip = np.array([landmarks[trunk_lm.value].x * width, landmarks[trunk_lm.value].y * height])
            shoulder = np.array([landmarks[shoulder_lm.value].x * width, landmarks[shoulder_lm.value].y * height])
            elbow = np.array([landmarks[elbow_lm.value].x * width, landmarks[elbow_lm.value].y * height])
            wrist = np.array([landmarks[wrist_lm.value].x * width, landmarks[wrist_lm.value].y * height])
            index_finger = np.array([landmarks[grip_lm.value].x * width, landmarks[grip_lm.value].y * height])
            nose = np.array([landmarks[head_lm.value].x * width, landmarks[head_lm.value].y * height])

            m.trunk_coord = (int(hip[0]), int(hip[1]))
            m.shoulder_coord = (int(shoulder[0]), int(shoulder[1]))
            m.elbow_coord = (int(elbow[0]), int(elbow[1]))
            m.wrist_coord = (int(wrist[0]), int(wrist[1]))
            m.grip_coord = (int(index_finger[0]), int(index_finger[1]))
            m.head_coord = (int(nose[0]), int(nose[1]))

            # 1. TRUNK
            if trunk_state["prev_pos"] is not None:
                delta = hip - trunk_state["prev_pos"]
                trunk_state["vx"] = alpha * (delta[0] / dt) + (1 - alpha) * trunk_state["vx"]
                trunk_state["vy"] = alpha * (delta[1] / dt) + (1 - alpha) * trunk_state["vy"]
                m.trunk_speed = float(np.sqrt(trunk_state["vx"]**2 + trunk_state["vy"]**2))
                m.trunk_v = (trunk_state["vx"], trunk_state["vy"])
            trunk_state["prev_pos"] = hip

            # 2. SHOULDER
            if shoulder_state["prev_pos"] is not None:
                delta = shoulder - shoulder_state["prev_pos"]
                shoulder_state["vx"] = alpha * (delta[0] / dt) + (1 - alpha) * shoulder_state["vx"]
                shoulder_state["vy"] = alpha * (delta[1] / dt) + (1 - alpha) * shoulder_state["vy"]
                m.shoulder_speed = float(np.sqrt(shoulder_state["vx"]**2 + shoulder_state["vy"]**2))
                m.shoulder_v = (shoulder_state["vx"], shoulder_state["vy"])
            shoulder_state["prev_pos"] = shoulder

            # 3. ELBOW (Upper Arm) & Angular Kinematics
            curr_angle = _calculate_angle(hip, shoulder, elbow)
            if elbow_state["prev_pos"] is not None and elbow_state["prev_angle"] is not None:
                delta = elbow - elbow_state["prev_pos"]
                elbow_state["vx"] = alpha * (delta[0] / dt) + (1 - alpha) * elbow_state["vx"]
                elbow_state["vy"] = alpha * (delta[1] / dt) + (1 - alpha) * elbow_state["vy"]
                m.elbow_speed = float(np.sqrt(elbow_state["vx"]**2 + elbow_state["vy"]**2))
                m.elbow_v = (elbow_state["vx"], elbow_state["vy"])

                raw_omega = (curr_angle - elbow_state["prev_angle"]) / dt
                elbow_state["omega"] = alpha * raw_omega + (1 - alpha) * elbow_state["omega"]
                m.upper_arm_angular_speed = float(elbow_state["omega"])

                raw_ang_accel = (elbow_state["omega"] - elbow_state["prev_omega"]) / dt
                elbow_state["ang_accel"] = alpha * raw_ang_accel + (1 - alpha) * elbow_state["ang_accel"]
                m.upper_arm_angular_deceleration = float(elbow_state["ang_accel"])

                elbow_state["prev_omega"] = elbow_state["omega"]

            elbow_state["prev_pos"] = elbow
            elbow_state["prev_angle"] = curr_angle

            # 4. WRIST (Forearm)
            if wrist_state["prev_pos"] is not None:
                delta = wrist - wrist_state["prev_pos"]
                wrist_state["vx"] = alpha * (delta[0] / dt) + (1 - alpha) * wrist_state["vx"]
                wrist_state["vy"] = alpha * (delta[1] / dt) + (1 - alpha) * wrist_state["vy"]
                m.wrist_speed = float(np.sqrt(wrist_state["vx"]**2 + wrist_state["vy"]**2))
                m.wrist_v = (wrist_state["vx"], wrist_state["vy"])
            wrist_state["prev_pos"] = wrist

            # 5. GRIP
            if grip_state["prev_pos"] is not None:
                delta = index_finger - grip_state["prev_pos"]
                grip_state["vx"] = alpha * (delta[0] / dt) + (1 - alpha) * grip_state["vx"]
                grip_state["vy"] = alpha * (delta[1] / dt) + (1 - alpha) * grip_state["vy"]
                m.grip_speed = float(np.sqrt(grip_state["vx"]**2 + grip_state["vy"]**2))
                m.grip_v = (grip_state["vx"], grip_state["vy"])
            grip_state["prev_pos"] = index_finger

        metrics.append(m)

    cap.release()
    pose.close()

    # ---------------------------------------------------------
    # Post-Processing: Outlier Rejection & Kinematic Smoothing
    # ---------------------------------------------------------
    # Applied to remove impossible human 50ms deceleration V-dips
    # caused by 2D camera foreshortening and tracking artifacts.
    if len(metrics) > 11:
        window_len = 11
        poly_order = 3
        med_size = 5 # Median filter size for spike removal

        def smooth_trajectory(data, is_speed=True):
            # Step 1: Reject sudden outlier dips/spikes
            despiked = median_filter(data, size=med_size)
            # Step 2: Smooth the continuous trajectory
            smoothed = savgol_filter(despiked, window_len, poly_order)
            if is_speed:
                smoothed = np.maximum(smoothed, 0.0) # Speed cannot be negative
            return smoothed

        trunk_s = smooth_trajectory([m.trunk_speed for m in metrics])
        shoulder_s = smooth_trajectory([m.shoulder_speed for m in metrics])
        elbow_s = smooth_trajectory([m.elbow_speed for m in metrics])
        wrist_s = smooth_trajectory([m.wrist_speed for m in metrics])
        grip_s = smooth_trajectory([m.grip_speed for m in metrics])

        ang_s = smooth_trajectory([m.upper_arm_angular_speed for m in metrics], is_speed=False)
        ang_a = smooth_trajectory([m.upper_arm_angular_deceleration for m in metrics], is_speed=False)

        for i, m in enumerate(metrics):
            m.trunk_speed = float(trunk_s[i])
            m.shoulder_speed = float(shoulder_s[i])
            m.elbow_speed = float(elbow_s[i])
            m.wrist_speed = float(wrist_s[i])
            m.grip_speed = float(grip_s[i])
            m.upper_arm_angular_speed = float(ang_s[i])
            m.upper_arm_angular_deceleration = float(ang_a[i])

    return metrics, fps
# endregion

# region Smash Assessment
# %%
@dataclass
class SmashEvent:
    peak_frame_idx: int
    peak_time_str: str
    critical_end_frame_idx: int
    critical_end_time_str: str
    start_frame_idx: int
    start_time_str: str
    end_frame_idx: int
    end_time_str: str
    overhead_start_frame_idx: int
    overhead_start_time_str: str
    overhead_end_frame_idx: int
    overhead_end_time_str: str
    critical_deceleration: float

def _format_timestamp(frame_idx: int, fps: float) -> str:
    """Helper to convert frame indices to MM:SS.mmm format."""
    total_seconds = frame_idx / fps
    mins = int(total_seconds // 60)
    secs = total_seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def find_smashes(kin_metrics: list[FrameMetrics],
                 fps: float,
                 time_spans_included: Optional[list[tuple[float, float]]] = None,
                 time_spans_excluded: Optional[list[tuple[float, float]]] = None,
                 max_returning_time_ms: int = 500,
                 pre_smash_buffer_ms: int = 500,
                 post_smash_buffer_ms: int = 300,
                 critical_deceleration_window_ms: int = 50,
                 min_overhead_time_ms: int = 50) -> list[SmashEvent]:
    """
    Identifies badminton smashes by finding overhead swing blocks lasting at least min_overhead_time_ms,
    locating the peak grip velocity, and extracting critical deceleration metrics.
    """
    smashes = []

    pre_frames = int((pre_smash_buffer_ms / 1000.0) * fps)
    post_frames = int((post_smash_buffer_ms / 1000.0) * fps)
    crit_dec_frames = max(1, int((critical_deceleration_window_ms / 1000.0) * fps))
    max_ret_frames = int((max_returning_time_ms / 1000.0) * fps)
    min_overhead_frames = max(1, int((min_overhead_time_ms / 1000.0) * fps))

    swing_blocks = []
    in_swing = False
    start_idx = 0

    # 1. Identify all overhead swing blocks meeting the minimum duration
    for i, m in enumerate(kin_metrics):
        if m.grip_coord and m.head_coord:
            grip_y = m.grip_coord[1]
            head_y = m.head_coord[1]

            if grip_y < head_y:  # Grip is physically higher than the head
                if not in_swing:
                    in_swing = True
                    start_idx = i
            else:  # Grip dropped below the head
                if in_swing:
                    end_idx = i
                    if (end_idx - start_idx) >= min_overhead_frames:
                        swing_blocks.append((start_idx, end_idx))
                    in_swing = False

    if in_swing:
        end_idx = len(kin_metrics)
        if (end_idx - start_idx) >= min_overhead_frames:
            swing_blocks.append((start_idx, end_idx))

    # 2. Filter blocks and extract metrics anchored to the kinematic peak
    for start_idx, end_idx in swing_blocks:
        max_grip_speed = -float('inf')
        peak_idx = start_idx
        for j in range(start_idx, end_idx):
            if kin_metrics[j].grip_speed > max_grip_speed:
                max_grip_speed = kin_metrics[j].grip_speed
                peak_idx = j

        peak_time_sec = peak_idx / fps

        if time_spans_included:
            if not any(start <= peak_time_sec <= end for start, end in time_spans_included):
                continue
        if time_spans_excluded:
            if any(start <= peak_time_sec <= end for start, end in time_spans_excluded):
                continue

        frames_to_return = end_idx - peak_idx
        if frames_to_return > max_ret_frames:
            continue

        dec_start = peak_idx
        dec_end = min(len(kin_metrics), peak_idx + crit_dec_frames)

        if dec_end > dec_start:
            avg_dec = sum(kin_metrics[j].upper_arm_angular_deceleration for j in range(dec_start, dec_end)) / (dec_end - dec_start)
        else:
            avg_dec = 0.0

        s_start = max(0, peak_idx - pre_frames)
        s_end = min(len(kin_metrics), peak_idx + post_frames)

        smashes.append(SmashEvent(
            peak_frame_idx=peak_idx,
            peak_time_str=_format_timestamp(peak_idx, fps),
            critical_end_frame_idx=dec_end,
            critical_end_time_str=_format_timestamp(dec_end, fps),
            start_frame_idx=s_start,
            start_time_str=_format_timestamp(s_start, fps),
            end_frame_idx=s_end,
            end_time_str=_format_timestamp(s_end, fps),
            overhead_start_frame_idx=start_idx,
            overhead_start_time_str=_format_timestamp(start_idx, fps),
            overhead_end_frame_idx=end_idx,
            overhead_end_time_str=_format_timestamp(end_idx, fps),
            critical_deceleration=avg_dec
        ))

    return smashes


class RiskLevel(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

@dataclass
class SmashAssessment:
    start_time_str: str
    peak_time_str: str
    end_time_str: str

    p_d_sequence: List[str]
    p_d_sequence_risk: RiskLevel

    velocity_amplification: List[str]
    velocity_amplification_risk: RiskLevel

    peak_angular_speed: float
    critical_deceleration: float
    critical_deceleration_risk: RiskLevel

    overall_risk: RiskLevel

def assess_smashes(kin_metrics: list[FrameMetrics],
                   smashes: list[SmashEvent],
                   max_critical_deceleration: float,
                   high_risk_dec_threshold_pct: float = 0.85,
                   mod_risk_dec_threshold_pct: float = 0.7) -> list[SmashAssessment]:
    """
    Evaluates kinematic characteristics of identified badminton smashes to assess shoulder injury risk.

    Checks three primary characteristics:
    1. Proximal-to-distal (P-D) Sequence:
       - Low Risk: Exact expected sequence (Trunk -> Upper Arm -> Forearm -> Grip).
       - High Risk: Proximal segments (Trunk/Upper Arm) peak AFTER distal segments (Forearm/Grip).
       - Moderate Risk: Minor sequencing errors (e.g., Trunk/Upper Arm swap, or Forearm/Grip swap).

    2. Velocity Amplification:
       - Low Risk: Speeds increase purely distally (Trunk < Upper Arm < Forearm < Grip).
       - High Risk: Proximal segments are faster than distal segments.
       - Moderate Risk: Minor amplification errors.

    3. Upper-Arm Deceleration Rate:
       - Low Risk: < max_critical_deceleration * mod_risk_dec_threshold_pct
       - High Risk: >= max_critical_deceleration * high_risk_dec_threshold_pct
       - Moderate Risk: Between the two thresholds.

    Overall Risk Logic:
    - Dictated primarily by deceleration risk if deceleration risk > LOW.
    - Otherwise, if sequencing AND amplification are perfect, overall risk is LOW.
    - If sequencing OR amplification are flawed but deceleration is safe, overall risk is MODERATE.
    """

    assessments = []

    # Ensure our max deceleration baseline is a positive magnitude for safe percentage math
    baseline_max_dec = abs(max_critical_deceleration)

    for smash in smashes:
        # 1. Extract peak values and their frame indices within the smash window
        peaks = {
            'Trunk': {'val': -1.0, 'idx': -1},
            'Upper Arm': {'val': -1.0, 'idx': -1},
            'Forearm': {'val': -1.0, 'idx': -1},
            'Grip': {'val': -1.0, 'idx': -1}
        }

        peak_ang_speed = -1.0

        for i in range(smash.start_frame_idx, smash.end_frame_idx):
            m = kin_metrics[i]

            if m.trunk_speed > peaks['Trunk']['val']:
                peaks['Trunk'] = {'val': m.trunk_speed, 'idx': i}
            # Upper arm speed is represented by shoulder/elbow mechanics;
            # using elbow speed here based on standard distal definitions
            if m.shoulder_speed > peaks['Upper Arm']['val']:
                pass
            # Using elbow_speed for upper arm, wrist for forearm, grip for hand
            if m.elbow_speed > peaks['Upper Arm']['val']:
                peaks['Upper Arm'] = {'val': m.elbow_speed, 'idx': i}
            if m.wrist_speed > peaks['Forearm']['val']:
                peaks['Forearm'] = {'val': m.wrist_speed, 'idx': i}
            if m.grip_speed > peaks['Grip']['val']:
                peaks['Grip'] = {'val': m.grip_speed, 'idx': i}

            if m.upper_arm_angular_speed > peak_ang_speed:
                peak_ang_speed = m.upper_arm_angular_speed

        # 2. Evaluate Proximal-to-Distal Sequence
        # Sort segment names by the frame index they peaked at (ascending time)
        seq_sorted = sorted(peaks.keys(), key=lambda k: peaks[k]['idx'])

        # Check High Risk Sequence: Proximal peaks AFTER Distal
        if (peaks['Trunk']['idx'] > peaks['Forearm']['idx'] or
            peaks['Trunk']['idx'] > peaks['Grip']['idx'] or
            peaks['Upper Arm']['idx'] > peaks['Forearm']['idx'] or
            peaks['Upper Arm']['idx'] > peaks['Grip']['idx']):
            seq_risk = RiskLevel.HIGH
        elif seq_sorted == ['Trunk', 'Upper Arm', 'Forearm', 'Grip']:
            seq_risk = RiskLevel.LOW
        else:
            seq_risk = RiskLevel.MODERATE

        # 3. Evaluate Velocity Amplification
        # Sort segment names by their peak velocity magnitude (ascending speed)
        amp_sorted = sorted(peaks.keys(), key=lambda k: peaks[k]['val'])

        # Check High Risk Amplification: Proximal is FASTER than Distal
        if (peaks['Trunk']['val'] > peaks['Forearm']['val'] or
            peaks['Trunk']['val'] > peaks['Grip']['val'] or
            peaks['Upper Arm']['val'] > peaks['Forearm']['val'] or
            peaks['Upper Arm']['val'] > peaks['Grip']['val']):
            amp_risk = RiskLevel.HIGH
        elif amp_sorted == ['Trunk', 'Upper Arm', 'Forearm', 'Grip']:
            amp_risk = RiskLevel.LOW
        else:
            amp_risk = RiskLevel.MODERATE

        # 4. Evaluate Critical Deceleration Risk
        # Use absolute value to ensure negative acceleration (deceleration) scales correctly
        crit_dec_mag = abs(smash.critical_deceleration)

        if crit_dec_mag >= (baseline_max_dec * high_risk_dec_threshold_pct):
            dec_risk = RiskLevel.HIGH
        elif crit_dec_mag < (baseline_max_dec * mod_risk_dec_threshold_pct):
            dec_risk = RiskLevel.LOW
        else:
            dec_risk = RiskLevel.MODERATE

        # 5. Determine Overall Risk
        if dec_risk > RiskLevel.LOW:
            overall_risk = dec_risk
        else:
            if seq_risk == RiskLevel.LOW and amp_risk == RiskLevel.LOW:
                overall_risk = RiskLevel.LOW
            else:
                overall_risk = RiskLevel.MODERATE

        # 6. Construct Final Object
        assessments.append(SmashAssessment(
            start_time_str=smash.start_time_str,
            peak_time_str=smash.peak_time_str,
            end_time_str=smash.end_time_str,
            p_d_sequence=seq_sorted,
            p_d_sequence_risk=seq_risk,
            velocity_amplification=amp_sorted,
            velocity_amplification_risk=amp_risk,
            peak_angular_speed=peak_ang_speed,
            critical_deceleration=smash.critical_deceleration, # preserve original sign for logging/output
            critical_deceleration_risk=dec_risk,
            overall_risk=overall_risk
        ))

    return assessments
# endregion

# region Visualization:Graph
# %%
def plot_smashes_kinematics(metrics: list[FrameMetrics], fps: float,
                            smashes: list[SmashEvent],
                            output_dir: str):
    """
    Generates a 3-panel plot for EACH smash individually to prevent data compression.
    The timeline is dynamically sliced to [smash.start - 0.25s, smash.end + 0.25s].
    """
    buffer_frames = int(0.25 * fps)

    for i, smash in enumerate(smashes):
        # Calculate strict frame boundaries for this specific smash plot
        start_idx = max(0, smash.start_frame_idx - buffer_frames)
        end_idx = min(len(metrics), smash.end_frame_idx + buffer_frames)

        sliced_metrics = metrics[start_idx:end_idx]
        times = np.arange(start_idx, end_idx) / fps

        # Extract data for the sliced window
        trunk_v = [m.trunk_speed for m in sliced_metrics]
        shoulder_v = [m.shoulder_speed for m in sliced_metrics]
        elbow_v = [m.elbow_speed for m in sliced_metrics]
        wrist_v = [m.wrist_speed for m in sliced_metrics]
        grip_v = [m.grip_speed for m in sliced_metrics]

        ang_accel = [m.upper_arm_angular_deceleration for m in sliced_metrics]
        grip_y = [m.grip_coord[1] if m.grip_coord else np.nan for m in sliced_metrics]
        head_y = [m.head_coord[1] if m.head_coord else np.nan for m in sliced_metrics]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Kinematic Plot: Smash {i+1}', fontsize=16, fontweight='bold')

        # --- Panel 1: Linear Speeds ---
        ax1.plot(times, trunk_v, label='Trunk', color='green', linewidth=1.5)
        ax1.plot(times, elbow_v, label='Upper Arm', color='blue', linewidth=1.5)
        ax1.plot(times, wrist_v, label='Forearm', color='orange', linewidth=1.5)
        ax1.plot(times, grip_v, label='Grip', color='red', linewidth=2)
        ax1.set_ylabel('Speed (px/s)')
        ax1.set_title('Proximal-to-Distal Kinetic Chain & Velocity Amplification')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Panel 2: Upper Arm Angular Deceleration ---
        ax2.plot(times, ang_accel, label='Upper Arm Ang Accel/Decel', color='orange', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(times, 0, ang_accel, where=(np.array(ang_accel) < 0), color='red', alpha=0.3, label='Deceleration Phase')
        ax2.set_ylabel('Acceleration (deg/s²)')
        ax2.set_title('Upper Arm Angular Deceleration Profile')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- Panel 3: Position Validation ---
        ax3.plot(times, grip_y, label='Grip Level', color='red', linewidth=2)
        ax3.plot(times, head_y, label='Head Level', color='blue', linestyle='--', linewidth=2)
        ax3.invert_yaxis()
        ax3.set_ylabel('Vertical Position (px)')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Stroke Elevation (Grip vs Head)')
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Draw regions and boundary lines relative to the actual video timeline
        start_t = smash.start_frame_idx / fps
        end_t = smash.end_frame_idx / fps
        peak_t = smash.peak_frame_idx / fps
        crit_end_t = smash.critical_end_frame_idx / fps

        for ax in (ax1, ax2, ax3):
            ax.axvspan(start_t, end_t, color='gray', alpha=0.15, zorder=0)
            ax.axvspan(peak_t, crit_end_t, color='red', alpha=0.15, zorder=0)
            ax.axvline(x=peak_t, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Kinematic Peak')
            ax.axvline(x=crit_end_t, color='darkred', linestyle=':', linewidth=1.5, alpha=0.8, label='Crit. Window End')

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper right')

        plt.tight_layout()

        # Save individual file per smash
        fig_name = os.path.join(output_dir, f'kinematic_plot_{i}.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig) # Free memory for the web server
# endregion

# region Visualization: Overlaying Video
# %% [markdown]
# * Display speeds
# * Display a detail assessment table at bottom-right of the screen during each smash
#   Smash 1
#
#   | Feature   | Result                   | Risk    |
#   |-----------|--------------------------|---------|
#   | P-D Seq   | Core, Arm, Forearm, Hand | 🟢 Low  |
#   | V-Amp     | Core, Arm, Hand, Forearm | 🟡 Mod  |
#   | UA Decel  | 6.2k / 3.0k              | 🔴 High |
# * Display a summary table at bottom-right of the screen after all smashes
#
#   | # | Time          | Risk    |
#   |---|---------------|---------|
#   | 1 | 02:25 - 02:26 | 🟢 Low  |
#   | 2 | 03:10 - 03:11 | 🔴 High |
# %%
def _draw_double_arrow(img, start_pt, end_pt, color, thickness=2):
    """
    Draws an arrow with a double-line shaft, mathematically matching
    the 'implies' symbol from the reference image.
    """
    p1 = np.array(start_pt, dtype=float)
    p2 = np.array(end_pt, dtype=float)

    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-3:
        return

    u = vec / length
    n = np.array([-u[1], u[0]])

    offset = 3.0
    tip_length = min(length * 0.35, 18.0)
    tip_width = 8.0

    if tip_width <= offset:
        tip_width = offset + 2.0

    intersect_dist = (offset * tip_length) / tip_width

    shaft_end_center = p2 - u * intersect_dist
    l1_end = shaft_end_center + n * offset
    l2_end = shaft_end_center - n * offset

    l1_start = p1 + n * offset
    l2_start = p1 - n * offset

    base_pt = p2 - u * tip_length
    wing1 = base_pt + n * tip_width
    wing2 = base_pt - n * tip_width

    cv2.line(img, (int(l1_start[0]), int(l1_start[1])), (int(l1_end[0]), int(l1_end[1])), color, thickness, cv2.LINE_AA)
    cv2.line(img, (int(l2_start[0]), int(l2_start[1])), (int(l2_end[0]), int(l2_end[1])), color, thickness, cv2.LINE_AA)

    cv2.line(img, (int(p2[0]), int(p2[1])), (int(wing1[0]), int(wing1[1])), color, thickness, cv2.LINE_AA)
    cv2.line(img, (int(p2[0]), int(p2[1])), (int(wing2[0]), int(wing2[1])), color, thickness, cv2.LINE_AA)

def _draw_table_overlay(frame, x, y, w, h):
    """Helper to draw a semi-transparent background for tables."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def _get_risk_visuals(risk_level):
    """Maps the RiskLevel enum to BGR colors and string labels."""
    if risk_level == 0:  # LOW
        return (0, 255, 0), "Low"     # Green
    elif risk_level == 1:  # MODERATE
        return (0, 255, 255), "Mod"   # Yellow
    else:  # HIGH
        return (0, 0, 255), "High"    # Red

def format_sequence(seq):
    """Shortens joint names to fit neatly in the table columns."""
    return ", ".join([s.replace("Upper Arm", "Arm").replace("Trunk", "Core").replace("Grip", "Hand") for s in seq])


def overlay_kinematic_assessment(input_video_path,
                                 output_video_path,
                                 metrics:list[FrameMetrics],
                                 smashes:list[SmashEvent] | None = None,
                                 assessments:list[SmashAssessment] | None = None,
                                 scale_linear=0.08,
                                 scale_ang_accel=0.01,
                                 show_labels=False,
                                 pixels_per_meter=None,
                                 overwrite=True):

    if smashes is None: smashes = []
    if assessments is None: assessments = []

    if os.path.exists(output_video_path):
        if not overwrite:
            print(f"Skipping: Output file '{output_video_path}' already exists.")
            return
        else:
            print(f"Overwriting: Deleting existing file '{output_video_path}'...")
            os.remove(output_video_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 120.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (width, height))

    divisor = 1.0 if pixels_per_meter is None else pixels_per_meter
    unit_v = "px/s" if pixels_per_meter is None else "m/s"
    fmt_v = "{:.0f}" if pixels_per_meter is None else "{:.2f}"

    # Arrow Colors (BGR)
    c_trunk = (0, 255, 0)
    c_upper_arm = (255, 0, 0)
    c_accel = (0, 165, 255)
    c_forearm = (0, 165, 255)
    c_grip = (0, 0, 255)

    # --- DYNAMIC SCALING SETUP ---
    size_scale = width / 1024.0

    # Fonts
    font_title = max(0.4, 0.7 * size_scale)
    font_header = max(0.3, 0.6 * size_scale)
    font_base = max(0.3, 0.5 * size_scale)
    font_small = max(0.2, 0.4 * size_scale)

    # Thicknesses (Must be integers >= 1)
    thick_thin = max(1, int(1 * size_scale))
    thick_med = max(1, int(2 * size_scale))
    thick_thick = max(1, int(3 * size_scale))

    # Radii and Offsets
    circle_radius = max(2, int(5 * size_scale))
    off_x = int(15 * size_scale)

    print("Rendering kinematic assessment video...")
    frame_idx = 0

    # Determine the frame index after the final smash ends to trigger the summary table
    summary_start_frame = smashes[-1].end_frame_idx if smashes else float('inf')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(metrics):
            break

        m = metrics[frame_idx]

        # --- 1. DRAW KINEMATIC ARROWS ---

        if m.trunk_coord and m.trunk_speed > 20:
            start_pt = m.trunk_coord
            end_pt = (int(start_pt[0] + m.trunk_v[0] * scale_linear),
                      int(start_pt[1] + m.trunk_v[1] * scale_linear))
            cv2.arrowedLine(frame, start_pt, end_pt, c_trunk, thick_thick, tipLength=0.25)
            cv2.circle(frame, start_pt, circle_radius, (0, 0, 255), -1)
            if show_labels:
                cv2.putText(frame, f"Trunk: {fmt_v.format(m.trunk_speed / divisor)} {unit_v}",
                            (start_pt[0] + off_x, start_pt[1] + int(20 * size_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_trunk, thick_med, cv2.LINE_AA)

        if m.elbow_coord and m.shoulder_coord:
            start_pt = m.elbow_coord
            if m.elbow_speed > 30:
                end_pt_vel = (int(start_pt[0] + m.elbow_v[0] * scale_linear),
                              int(start_pt[1] + m.elbow_v[1] * scale_linear))
                cv2.arrowedLine(frame, start_pt, end_pt_vel, c_upper_arm, thick_thick, tipLength=0.25)

                ang_accel = m.upper_arm_angular_deceleration
                dx = m.elbow_coord[0] - m.shoulder_coord[0]
                dy = m.elbow_coord[1] - m.shoulder_coord[1]
                arm_length = math.hypot(dx, dy)

                if arm_length > 0:
                    perp_dx, perp_dy = -dy / arm_length, dx / arm_length
                    visual_ax = perp_dx * ang_accel * scale_ang_accel
                    visual_ay = perp_dy * ang_accel * scale_ang_accel
                    end_pt_accel = (int(start_pt[0] + visual_ax), int(start_pt[1] + visual_ay))
                    _draw_double_arrow(frame, start_pt, end_pt_accel, c_accel, thickness=thick_med)

                cv2.circle(frame, start_pt, circle_radius, (0, 0, 255), -1)
                if show_labels:
                    cv2.putText(frame, f"Upper Arm: {fmt_v.format(m.elbow_speed / divisor)} {unit_v}",
                                (start_pt[0] + off_x, start_pt[1] - int(25 * size_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_upper_arm, thick_med, cv2.LINE_AA)
                    cv2.putText(frame, f"Accel: {int(ang_accel)} deg/s^2",
                                (start_pt[0] + off_x, start_pt[1] - int(8 * size_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_small, c_accel, thick_thin, cv2.LINE_AA)

        if m.wrist_coord and m.wrist_speed > 40:
            start_pt = m.wrist_coord
            end_pt = (int(start_pt[0] + m.wrist_v[0] * scale_linear),
                      int(start_pt[1] + m.wrist_v[1] * scale_linear))
            cv2.arrowedLine(frame, start_pt, end_pt, c_forearm, thick_thick, tipLength=0.25)
            cv2.circle(frame, start_pt, circle_radius, (0, 0, 255), -1)
            if show_labels:
                cv2.putText(frame, f"Forearm: {fmt_v.format(m.wrist_speed / divisor)} {unit_v}",
                            (start_pt[0] + off_x, start_pt[1] - int(20 * size_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_forearm, thick_med, cv2.LINE_AA)

        if m.grip_coord and m.grip_speed > 50:
            start_pt = m.grip_coord
            end_pt = (int(start_pt[0] + m.grip_v[0] * scale_linear),
                      int(start_pt[1] + m.grip_v[1] * scale_linear))
            cv2.arrowedLine(frame, start_pt, end_pt, c_grip, thick_thick, tipLength=0.25)
            cv2.circle(frame, start_pt, circle_radius, (0, 0, 255), -1)
            if show_labels:
                cv2.putText(frame, f"Racket Grip: {fmt_v.format(m.grip_speed / divisor)} {unit_v}",
                            (start_pt[0] + off_x, start_pt[1] - int(20 * size_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_grip, thick_med, cv2.LINE_AA)

        # --- 2. DRAW HUD TABLES ---

        active_smash_idx = -1
        for i, s in enumerate(smashes):
            if s.start_frame_idx <= frame_idx <= s.end_frame_idx:
                active_smash_idx = i
                break

        # Render Table 1: Detailed Smash Assessment (During Smash)
        if active_smash_idx != -1 and active_smash_idx < len(assessments):
            a = assessments[active_smash_idx]

            # ==========================================
            # TABLE 1 LAYOUT PARAMETERS
            # ==========================================
            t1_margin_right = int(15 * size_scale)
            t1_y_from_top = height - int(260 * size_scale)

            t1_pad_x = int(15 * size_scale)
            t1_pad_y_top = int(30 * size_scale)
            t1_row_h = int(30 * size_scale)
            t1_bottom_pad = int(15 * size_scale)

            t1_col1_w = int(80 * size_scale)
            t1_col2_w = int(200 * size_scale)
            t1_col3_w = int(40 * size_scale)
            # ==========================================

            tw = t1_pad_x + t1_col1_w + t1_col2_w + t1_col3_w + t1_pad_x
            th = t1_pad_y_top + int(10 * size_scale) + t1_row_h + (3 * t1_row_h) + t1_bottom_pad

            tx = width - tw - t1_margin_right
            ty = t1_y_from_top

            _draw_table_overlay(frame, tx, ty, tw, th)

            # Titles & Headers
            cv2.putText(frame, f"Smash {active_smash_idx + 1}", (tx + t1_pad_x, ty + t1_pad_y_top), cv2.FONT_HERSHEY_SIMPLEX, font_title, (255, 255, 255), thick_med, cv2.LINE_AA)
            cv2.line(frame, (tx + t1_pad_x, ty + t1_pad_y_top + int(10 * size_scale)), (tx + tw - t1_pad_x, ty + t1_pad_y_top + int(10 * size_scale)), (255, 255, 255), thick_thin)

            # Calculated Column Offsets
            col1 = tx + t1_pad_x
            col2 = col1 + t1_col1_w
            col3 = col2 + t1_col2_w
            y_base = ty + t1_pad_y_top + int(40 * size_scale)

            # Headers
            cv2.putText(frame, "Feature", (col1, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, "Result", (col2, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, "Risk", (col3, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)

            # Row 1: P-D Sequence
            c_pd, txt_pd = _get_risk_visuals(a.p_d_sequence_risk)
            cv2.putText(frame, "P-D Seq", (col1, y_base + t1_row_h), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, format_sequence(a.p_d_sequence), (col2, y_base + t1_row_h), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, txt_pd, (col3, y_base + t1_row_h), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_pd, thick_med, cv2.LINE_AA)

            # Row 2: Velocity Amplification
            c_amp, txt_amp = _get_risk_visuals(a.velocity_amplification_risk)
            cv2.putText(frame, "V-Amp", (col1, y_base + t1_row_h * 2), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, format_sequence(a.velocity_amplification), (col2, y_base + t1_row_h * 2), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, txt_amp, (col3, y_base + t1_row_h * 2), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_amp, thick_med, cv2.LINE_AA)

            # Row 3: UA Deceleration
            c_dec, txt_dec = _get_risk_visuals(a.critical_deceleration_risk)
            cv2.putText(frame, "UA Decel", (col1, y_base + t1_row_h * 3), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, f"{abs(a.critical_deceleration):.0f} deg/s²", (col2, y_base + t1_row_h * 3), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, txt_dec, (col3, y_base + t1_row_h * 3), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_dec, thick_med, cv2.LINE_AA)

        # Render Table 2: Overall Summary (After the last smash completes)
        elif frame_idx > summary_start_frame:

            # ==========================================
            # TABLE 2 LAYOUT PARAMETERS
            # ==========================================
            t2_margin_right = int(30 * size_scale)
            t2_margin_bottom = int(90 * size_scale)

            t2_pad_x = int(15 * size_scale)
            t2_pad_y_top = int(30 * size_scale)
            t2_row_h = int(30 * size_scale)

            t2_col1_w = int(80 * size_scale)
            t2_col2_w = int(200 * size_scale)
            t2_col3_w = int(40 * size_scale)
            # ==========================================

            tw = t2_pad_x + t2_col1_w + t2_col2_w + t2_col3_w + t2_pad_x
            th = int(80 * size_scale) + (len(assessments) * t2_row_h)

            tx = width - tw - t2_margin_right
            ty = height - th - t2_margin_bottom

            _draw_table_overlay(frame, tx, ty, tw, th)

            # Title
            cv2.putText(frame, "Smash Assessment Summary", (tx + t2_pad_x, ty + t2_pad_y_top), cv2.FONT_HERSHEY_SIMPLEX, font_header, (255, 255, 255), thick_med, cv2.LINE_AA)
            cv2.line(frame, (tx + t2_pad_x, ty + t2_pad_y_top + int(10 * size_scale)), (tx + tw - t2_pad_x, ty + t2_pad_y_top + int(10 * size_scale)), (255, 255, 255), thick_thin)

            # Calculated Column Offsets
            col1 = tx + t2_pad_x
            col2 = col1 + t2_col1_w
            col3 = col2 + t2_col2_w
            y_base = ty + t2_pad_y_top + int(35 * size_scale)

            # Headers
            cv2.putText(frame, "#", (col1, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, "Time", (col2, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)
            cv2.putText(frame, "Risk", (col3, y_base), cv2.FONT_HERSHEY_SIMPLEX, font_base, (200, 200, 200), thick_thin, cv2.LINE_AA)

            # Populate Smash Rows
            for i, a in enumerate(assessments):
                row_y = y_base + (i + 1) * t2_row_h
                c_risk, txt_risk = _get_risk_visuals(a.overall_risk)

                time_span = f"{a.start_time_str[:5]} - {a.end_time_str[:5]}"

                cv2.putText(frame, str(i + 1), (col1, row_y), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
                cv2.putText(frame, time_span, (col2, row_y), cv2.FONT_HERSHEY_SIMPLEX, font_base, (255, 255, 255), thick_thin, cv2.LINE_AA)
                cv2.putText(frame, txt_risk, (col3, row_y), cv2.FONT_HERSHEY_SIMPLEX, font_base, c_risk, thick_med, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Success! Saved to {output_video_path}")
# endregion
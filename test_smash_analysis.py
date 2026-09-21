# %%
import smash_analysis as sa
from pathlib import Path

input_video = 'badminton-smash-1.mp4'

file_path = Path(input_video)
file_stem = file_path.stem

output_video = f"{file_stem}_analyzed{file_path.suffix}"
output_kin_image = f"{file_stem}_kinematic.png"

# %%
kin_metrics, fps = sa.extract_kinematic_metrics(input_video)

# %%
smashes = sa.find_smashes(kin_metrics, fps)
max_critical_dec = max([abs(s.critical_deceleration) for s in smashes] + [0])

# %%
assessments = sa.assess_smashes(kin_metrics=kin_metrics,
               smashes=smashes,
               max_critical_deceleration=max_critical_dec)

# %%
sa.plot_smash_validation(kin_metrics, fps,
                         smashes=smashes,
                         assessments=assessments,
                         fig_name=output_kin_image)

sa.overlay_kinematic_assessment(
    input_video_path=input_video,
    output_video_path=output_video,
    metrics=kin_metrics,
    smashes=smashes,
    assessments=assessments,
    scale_linear=0.2,
    scale_ang_accel=0.05,
    show_labels=True,
    # pixels_per_meter=120
)
"""
Configuration for the pneumatic chicken-wing tip-deflection benchmark.

Edit GROUPS to add/move videos between experimental conditions.
Paths are relative to this file's directory (the experiment root).

Excluded by design: any "*dont-use*" / "blooper*" files, and the
2.5%_Formalin .../sped_up/*.mp4 clips (they are 10x duplicates of the originals).
"""
import os

# Real-world size of ONE small graph-paper square. Used to convert px -> mm.
GRID_MM_PER_SQUARE = 1.0

# Frames at the very start of each clip assumed to be at rest (unactuated).
# The "rest" tip position is the median of these frames.
REST_FRAMES = 12

# Hard override of px-per-square per clip (skips calibration entirely).
# Key = video rel_path, value = px per 1 grid square.
# These were auto-measured from the dark grid-line spacing (wd.measure_grid),
# median over 400+ clean line spacings per clip -- the grid is ~1mm and the rig
# is nearly constant, so values cluster at 45-49 px. Re-measure or set to None
# to fall back to interactive calibration.
GRID_OVERRIDE = {
    "pneumatics_first_test_5.20.26/fresh_chicken_bicep_1.MOV": 47.0,
    "pneumatics_first_test_5.20.26/fresh_chicken_bicep_2.MOV": 46.0,
    "pneumatics_first_test_5.20.26/fresh_chicken_bicep_3.MOV": 45.0,
    "pneumatics_first_test_5.20.26/fresh_chicken_bicep_4.MOV": 45.0,
    "pneumatics_first_test_5.20.26/fresh_chicken_double_actuator.MOV": 46.0,
    "pneumatics_first_test_5.20.26/fresh_chicken_double_actuator_2.MOV": 48.0,
    "pneumatics_first_test_5.20.26/5%_formalin_tricep_1.MOV": 46.0,
    "pneumatics_first_test_5.20.26/5%_formalin_tricep_2.MOV": 46.0,
    "pneumatics_first_test_5.20.26/5%_formalin_tricep_3.MOV": 46.0,
    "pneumatics_first_test_5.20.26/5%_formalin_tricep_4.MOV": 46.0,
    "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_1.MOV": 47.0,
    "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_2.MOV": 47.0,
    "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_3.MOV": 47.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_3.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_3.1.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1_forearm_core_taken_out.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1_forearm_core_taken_out_2.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_2_forearm_core_taken_out.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_2_forearm_core_taken_out_2.MOV": 49.0,
    "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_4days_1.MOV": 49.0,
}

# Experimental groups -> list of video paths (relative to experiment root).
GROUPS = {
    "fresh": [
        "pneumatics_first_test_5.20.26/fresh_chicken_bicep_1.MOV",
        "pneumatics_first_test_5.20.26/fresh_chicken_bicep_2.MOV",
        "pneumatics_first_test_5.20.26/fresh_chicken_bicep_3.MOV",
        "pneumatics_first_test_5.20.26/fresh_chicken_bicep_4.MOV",
    ],
    "fresh_double_actuator": [
        "pneumatics_first_test_5.20.26/fresh_chicken_double_actuator.MOV",
        "pneumatics_first_test_5.20.26/fresh_chicken_double_actuator_2.MOV",
    ],
    "5%_formalin_2day": [
        "pneumatics_first_test_5.20.26/5%_formalin_tricep_1.MOV",
        "pneumatics_first_test_5.20.26/5%_formalin_tricep_2.MOV",
        "pneumatics_first_test_5.20.26/5%_formalin_tricep_3.MOV",
        "pneumatics_first_test_5.20.26/5%_formalin_tricep_4.MOV",
    ],
    "2.5%_formalin_1day": [
        "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_1.MOV",
        "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_2.MOV",
        "2.5%_Formalin_30%_glycerin_chicken_5.25.26/2.5%_formalin_1day_3.MOV",
    ],
    "70%_EtOH_1day": [
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1.MOV",
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_3.MOV",
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_3.1.MOV",
    ],
    "70%_EtOH_1day_core_removed": [
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1_forearm_core_taken_out.MOV",
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_1_forearm_core_taken_out_2.MOV",
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_2_forearm_core_taken_out.MOV",
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_1days_2_forearm_core_taken_out_2.MOV",
    ],
    "70%_EtOH_4day": [
        "EtOH_fixed_chicken_pneumatic_experiments_6.11.26/70%_EtOH_4days_1.MOV",
    ],
}

# Order (and thus color/x-position) of groups in the plots: fresh -> most fixed.
GROUP_ORDER = [
    "fresh",
    "fresh_double_actuator",
    "5%_formalin_2day",
    "2.5%_formalin_1day",
    "70%_EtOH_1day",
    "70%_EtOH_1day_core_removed",
    "70%_EtOH_4day",
]

# Human-readable labels for plotting.
GROUP_LABELS = {
    "fresh": "Fresh\n(unfixed)",
    "fresh_double_actuator": "Fresh\n(2 actuators)",
    "5%_formalin_2day": "5% formalin 2 day\n(core removed)",
    "2.5%_formalin_1day": "2.5% formalin 1 day\n(core removed)",
    "70%_EtOH_1day": "70% EtOH\n1 day",
    "70%_EtOH_1day_core_removed": "70% EtOH 1 day\n(core removed)",
    "70%_EtOH_4day": "70% EtOH\n4 day",
}

# --- derived paths -------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "deflection_analysis")
TRACKS_DIR = os.path.join(RESULTS_DIR, "tracks")       # per-video CSV + JSON
OVERLAYS_DIR = os.path.join(RESULTS_DIR, "overlays")   # verification mp4s
WING_SIZE_DIR = os.path.join(RESULTS_DIR, "wing_size")  # wing-size masks + JSON
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def all_videos():
    """Yield (group, abs_path, rel_path) for every configured video."""
    for group in GROUP_ORDER:
        for rel in GROUPS[group]:
            yield group, os.path.join(ROOT, rel), rel


def result_stem(rel_path):
    """A filesystem-safe stem unique per video, used for output filenames."""
    return rel_path.replace("/", "__").rsplit(".", 1)[0]

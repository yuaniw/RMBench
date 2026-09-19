"""Identity of the three-camera bimanual September 17 recording."""

REPO_ID = "press_button-memory-260917-franka-3view-v1"
TRAIN_CONFIG = "pi0_base_franka_press_button_260917_anchor_adaln_bz32_h50_30k"
PROMPT_TRAIN_CONFIG = "pi0_base_franka_press_button_260917_left3_right1_anchor_adaln_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_press_button_260917_precompute_h50"
DEDUP_REPO_ID = "press_button-memory-260917-franka-3view-dedup-v1"
DEDUP_TRAIN_CONFIG = "pi0_base_franka_press_button_260917_dedup_anchor_adaln_bz32_h50_30k"
DEDUP_PRECOMPUTE_CONFIG = "pi0_base_franka_press_button_260917_dedup_precompute_h50"
PROMPT = "Press the left button three times, then press the right button once"
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
    "observation.images.cam_right": "observation.images.cam_right_wrist",
}
JOINT_NAMES = tuple(
    name for arm in ("left", "right")
    for name in (*(f"{arm}_joint_{i}" for i in range(1, 8)), f"{arm}_gripper")
)

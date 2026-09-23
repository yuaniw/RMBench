"""Single-left-arm, front/left-wrist September 19 press-button dataset."""

REPO_ID = "memory_0919-press-button-franka-left-2view-v1"
TRAIN_CONFIG = "pi0_base_franka_left_press_button_260919_anchor_adaln_bz32_h50_30k"
BASELINE_CONFIG = "pi0_base_franka_left_press_button_260919_lora_baseline_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_left_press_button_260919_precompute_h50"
PROMPT = "press the button three times"
FPS = 15  # Declared export rate; processed source may have a different physical time basis.
HORIZON = 50
JOINT_NAMES = (*tuple(f"left_joint_{i}" for i in range(1, 8)), "left_gripper")
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
}

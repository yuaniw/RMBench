"""September 23 right-arm putback-block memory task."""

REPO_ID = "memory_0923-putback-block-franka-right-2view-v1"
TRAIN_CONFIG = "pi0_base_franka_right_putback_block_260923_anchor_adaln_bz32_h50_30k"
BASELINE_CONFIG = "pi0_base_franka_right_putback_block_260923_lora_baseline_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_right_putback_block_260923_precompute_h50"
PROMPT = "Move the block from the left or right side to the center, then return it to its original position."
FPS = 15
HORIZON = 50
ARM_MODE = "right"
GRIPPER_STATE_CLAMP_ZERO = True
JOINT_NAMES = (*tuple(f"right_joint_{i}" for i in range(1, 8)), "right_gripper")
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_right": "observation.images.cam_right_wrist",
}

"""September 24 right-arm swap-block-through-center memory task."""

REPO_ID = "memory_0924-swap-block-franka-right-2view-v1"
TRAIN_CONFIG = "pi0_base_franka_right_swap_block_260924_anchor_adaln_bz32_h50_30k"
BASELINE_CONFIG = "pi0_base_franka_right_swap_block_260924_lora_baseline_bz32_h50_30k"
FULL_CONFIG = "pi0_base_franka_right_swap_block_260924_full_finetune_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_right_swap_block_260924_precompute_h50"
PROMPT = "Swap the left and right blocks by moving them through the center."
FPS = 15
HORIZON = 50
ARM_MODE = "right"
GRIPPER_STATE_CLAMP_ZERO = True
JOINT_NAMES = (*tuple(f"right_joint_{i}" for i in range(1, 8)), "right_gripper")
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_right": "observation.images.cam_right_wrist",
}

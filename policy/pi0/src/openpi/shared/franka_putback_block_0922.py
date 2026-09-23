"""September 22 left-arm block-to-center-and-back memory task."""

REPO_ID = "memory_0922-putback-block-franka-left-2view-v1"
TRAIN_CONFIG = "pi0_base_franka_left_putback_block_260922_anchor_adaln_bz32_h50_30k"
BASELINE_CONFIG = "pi0_base_franka_left_putback_block_260922_lora_baseline_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_left_putback_block_260922_precompute_h50"
PROMPT = "Move the block from the left or right side to the center, then return it to its original position."
FPS = 15
HORIZON = 50
# Source observation.state clamps tiny negative measured gripper widths to zero.
GRIPPER_STATE_CLAMP_ZERO = True
JOINT_NAMES = (*tuple(f"left_joint_{i}" for i in range(1, 8)), "left_gripper")
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
}

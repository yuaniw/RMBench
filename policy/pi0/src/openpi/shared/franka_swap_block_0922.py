"""September 22 left-arm swap-block-through-center memory task."""

REPO_ID = "memory_0922-swap-block-franka-left-2view-v1"
TRAIN_CONFIG = "pi0_base_franka_left_swap_block_260922_anchor_adaln_bz32_h50_30k"
PRECOMPUTE_CONFIG = "pi0_base_franka_left_swap_block_260922_precompute_h50"
PROMPT = "Swap the left and right blocks by moving them through the center."
FPS = 15
HORIZON = 50
GRIPPER_STATE_CLAMP_ZERO = True
JOINT_NAMES = (*tuple(f"left_joint_{i}" for i in range(1, 8)), "left_gripper")
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
}

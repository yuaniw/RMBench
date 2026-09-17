"""Dataset identity shared by the memory conversion and training entrypoints."""

REPO_ID = "memory_260915-franka-left-2view-v1"
SINGLE_REPO_ID = "memory_260915-franka-left-front-v1"
PRECOMPUTE_CONFIG = "pi0_base_franka_left_memory_260915_precompute_h50_v1"
TRAIN_CONFIG = "pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1"
SINGLE_PRECOMPUTE_CONFIG = "pi0_base_franka_left_memory_260915_front_precompute_h50_v1"
SINGLE_TRAIN_CONFIG = "pi0_base_franka_left_memory_260915_front_anchor_adaln_h50_30k_v1"
PROMPT = (
    "There are four mats, one block, and a button on the table. "
    "One block is on one of the mats. First, put the block to the center, "
    "then press the button. Then, put the block back in its original position."
)
FPS = 15
HORIZON = 50
JOINT_NAMES = tuple(f"left_joint_{i}" for i in range(1, 8)) + ("left_gripper",)
CAMERAS = {
    "observation.images.cam_front": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
}
SINGLE_CAMERAS = {"observation.images.cam_front": "observation.images.cam_high"}

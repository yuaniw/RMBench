from .pi_model import PI0


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    return PI0(
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["asset_id"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
        usr_args["eval_save_dir"],
        usr_args["history_overflow"],
    )


def eval(TASK_ENV, model, observation):

    first_observation = model.observation_window is None
    if first_observation:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state, advance_history=first_observation)

    # ======== Get Action ========

    actions = model.get_action(TASK_ENV.take_action_cnt)[:model.pi0_step]

    for action in actions:
        previous_action_count = TASK_ENV.take_action_cnt
        TASK_ENV.take_action(action)
        if TASK_ENV.take_action_cnt == previous_action_count:
            break
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()

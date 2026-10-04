"""UniGoal agent shim: do not auto-reset Habitat when an intermediate goal ends."""
import numpy as np

from src_unigoal.agent.unigoal.agent import UniGoal_Agent


class SequentialUniGoalAgent(UniGoal_Agent):
    def reset(self):
        obs, rgbd, info = super().reset()
        # The source agent attempts goal-image category recognition here.
        # Multi-IIN uses the task-provided category while retaining predicted
        # Mask R-CNN semantic channels in `rgbd`.
        self.envs.set_goal_cat_id(self.envs.task_goal_idx)
        info["goal_cat_id"] = self.envs.task_goal_idx
        info["goal_name"] = self.envs.goal_name
        return obs, rgbd, info

    def step(self, agent_input):
        if agent_input["wait"]:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.rgbd.shape), False, self.envs.info
        target_boxes = [box for box in self.pred_box if box[0] == self.envs.gt_goal_idx]
        agent_input["found_goal"] = bool(target_boxes)
        self.instance_discriminator(agent_input, target_boxes)
        action = self.get_action(agent_input)
        if self.args.visualize:
            self.visualize(agent_input)
        if action < 0:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.obs_shape), False, self.envs.info
        obs, done, info = self.envs.step({"action": action})
        raw = np.concatenate((obs["rgb"].astype(np.uint8), obs["depth"]), axis=2).transpose(2, 0, 1)
        self.raw_obs, self.raw_depth = raw[:3].transpose(1, 2, 0), raw[3:4]
        self.rgbd, _ = self.preprocess_obs(raw)
        self.last_action = action
        return obs, self.rgbd, done, info

    def switch_target(self, observations, info):
        """Reset goal-specific matching/planning state, retaining map and pose."""
        self.instance_imagegoal = info["instance_imagegoal"]
        self.envs.set_goal_cat_id(self.envs.task_goal_idx)
        raw = np.concatenate((observations["rgb"].astype(np.uint8), observations["depth"]), axis=2).transpose(2, 0, 1)
        self.raw_obs, self.raw_depth = raw[:3].transpose(1, 2, 0), raw[3:4]
        self.rgbd, _ = self.preprocess_obs(raw)
        self.global_goal = self.temp_goal = self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask.fill(1)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.last_action = None
        # Display-only trajectory: target2 figures must not overlay target1.
        # Planning memory (`visited`, collision map and BEV map) is retained.
        if self.visited_vis is not None:
            self.visited_vis.fill(0)
        return observations, self.rgbd, info

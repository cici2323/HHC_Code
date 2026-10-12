"""Task profiles for the teacher -> student -> residual pipeline."""

from .g1_mimic_priv_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicPrivDistillCfgPPO
from rsl_rl.residual.motion_rewards import PAPER_MOTION_WEIGHTS, PAPER_MOTION_SCALES


class G1PaperCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        num_envs = 512

    class rewards(G1MimicPrivCfg.rewards):
        only_positive_rewards = False
        clip_rewards = False

        class scales:
            # Upstream non-tracking terms, retained and separately accounted for.
            alive = 3.0
            orientation = -4.0
            dof_acc = -3e-7
            action_rate = -0.1
            dof_error = -0.1
            feet_stumble = -2.0
            dof_pos_limits = -10.0
            feet_air_time = 10.0
            feet_force = -3e-3
            ankle_action = -0.1
            waist_roll_pitch_error = -0.5

    class reward_profile:
        stage = "teacher"
        tracking_weight = 1.0
        motion_weights = dict(PAPER_MOTION_WEIGHTS)
        motion_scales = dict(PAPER_MOTION_SCALES)
        collaboration = {
            "support_weight": 5.0, "workspace_weight": 3.0,
            "stability_weight": 1.5, "slip_weight": 0.05,
            "support_tolerance": 0.02, "support_scale": 0.05,
        }

    class construction:
        enabled = False
        data_path = ""
        geometry_path = ""


class G1PaperTeacherCfgPPO(G1MimicPrivCfgPPO):
    seed = 42

    class runner(G1MimicPrivCfgPPO.runner):
        save_interval = 500
        experiment_name = "paper_teacher"

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        learning_rate = 1e-3
        entropy_coef = 0.005


class G1PaperStudentCfg(G1PaperCfg):
    class reward_profile(G1PaperCfg.reward_profile):
        stage = "student"


class G1PaperAdaptationCfg(G1PaperCfg):
    class reward_profile(G1PaperCfg.reward_profile):
        stage = "adaptation"
        tracking_weight = 0.2

    class construction(G1PaperCfg.construction):
        enabled = True


class G1PaperStudentCfgPPO(G1MimicPrivDistillCfgPPO):
    seed = 42

    class runner(G1MimicPrivDistillCfgPPO.runner):
        save_interval = 500
        experiment_name = "paper_student"

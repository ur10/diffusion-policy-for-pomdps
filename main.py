import os, time

t0 = time.time()
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import numpy as np
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
from absl import app, flags
from ml_collections import config_flags
import pickle
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    "configs/seq_models/mlp_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl", "config_env"])

# shared encoder settings
flags.DEFINE_boolean("shared_encoder", False, "share encoder in actor-critic or not")
flags.DEFINE_boolean(
    "freeze_critic", False, "in shared encoder, freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 1000, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.25, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("submit_time", None, "used in sbatch")


def main(argv):
    if FLAGS.seed < 0:
        seed = int(pid)  # to avoid conflict within a job which has same datetime
    else:
        seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    print(f"[1/4] Creating train env: {env_name} ...")
    env = make_env(env_name, seed)
    print(f"[1/4] Creating eval env ...")
    eval_env = make_env(env_name, seed + 42)
    print(f"[1/4] Envs ready.")

    system.reproduce(seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    set_gpu_mode(torch.cuda.is_available())

    ## now only use env and time as directory name
    run_name = f"{config_env.env_type}/{config_env.env_name}/"
    config_seq, _ = config_seq.name_fn(config_seq, env.max_episode_steps)
    max_training_steps = int(FLAGS.train_episodes * env.max_episode_steps)
    config_rl, _ = config_rl.name_fn(
        config_rl, env.max_episode_steps, max_training_steps
    )
    uid = f"{system.now_str()}+{jobid}-{pid}"
    run_name += uid

    format_strs = ["csv"]
    if FLAGS.debug:
        FLAGS.save_dir = "debug"
        format_strs.extend(["stdout", "log"])  # logger.log

    log_path = os.path.join(FLAGS.save_dir, run_name)

    if not config_rl.debug:
        import wandb
        print("[2/4] Initialising wandb ...")
        wandb.init(
            project=config_rl.wandb_project,
            name=config_rl.wandb_tag,
            config={
                "env": config_env.to_dict(),
                "rl": config_rl.to_dict(),
                "seq": config_seq.to_dict(),
                "seed": seed,
            },
        )
        format_strs.append("wandb")
        print("[2/4] wandb ready.")
    logger.configure(dir=log_path, format_strs=format_strs)

    # write flags to a txt
    key_flags = FLAGS.get_key_flags_for_module(argv[0])
    with open(os.path.join(log_path, "flags.txt"), "w") as text_file:
        text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")
    # write flags to pkl
    with open(os.path.join(log_path, "flags.pkl"), "wb") as f:
        pickle.dump(FLAGS.flag_values_dict(), f)

    # start training
    print("[3/4] Building agent ...")
    learner = Learner(env, eval_env, FLAGS, config_rl, config_seq, config_env)
    print("[4/4] Starting training ...")
    learner.train()


if __name__ == "__main__":
    app.run(main)

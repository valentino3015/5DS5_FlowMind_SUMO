"""PPO fine-tuning of a saved policy: replaced hyperparameters and critic warm-up."""
import gymnasium as gym
import numpy as np


class Toy(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (4,), np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    def __init__(self):
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.full(4, .5, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.full(4, .5, np.float32), float(action[0] == 1), self.t >= 8, False, {}


def test_resume_replaces_ppo_settings_and_rebuilds_the_buffer(tmp_path):
    from stable_baselines3 import PPO
    from scripts.train_emissions import load_for_finetuning
    PPO("MlpPolicy", Toy(), n_steps=2048, batch_size=64, device="cpu").save(tmp_path / "parent.zip")
    model = load_for_finetuning(tmp_path / "parent.zip", Toy(), {
        "n_steps": 64, "batch_size": 32, "learning_rate": 5e-5, "clip_range": .1, "target_kl": .01})
    assert (model.n_steps, model.batch_size, model.target_kl) == (64, 32, .01)
    assert model.rollout_buffer.buffer_size == 64
    assert model.clip_range(1.) == .1 and model.lr_schedule(1.) == 5e-5


def test_critic_warmup_keeps_the_decisions_unchanged():
    import torch
    from stable_baselines3 import PPO
    from scripts.train_emissions import set_actor_trainable
    model = PPO("MlpPolicy", Toy(), n_steps=64, batch_size=32, n_epochs=4, learning_rate=1e-2, device="cpu", seed=0)
    actor = [p.detach().clone() for m in (model.policy.mlp_extractor.policy_net, model.policy.action_net)
             for p in m.parameters()]
    critic = [p.detach().clone() for p in model.policy.value_net.parameters()]
    set_actor_trainable(model.policy, False)
    model.learn(128)
    after = [p for m in (model.policy.mlp_extractor.policy_net, model.policy.action_net) for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(actor, after))
    assert not all(torch.equal(a, b) for a, b in zip(critic, model.policy.value_net.parameters()))
    set_actor_trainable(model.policy, True)
    assert all(p.requires_grad for p in model.policy.action_net.parameters())

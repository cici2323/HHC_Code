"""PPO for a frozen-student residual policy, without simulator dependencies."""

import math

import torch


class ResidualPPO:
    """Update only the residual actor, critic and log standard deviation.

    Rollout fields ``residual_obs`` and ``latent_actions`` have shape [T, N, D]
    and [T, N, A]. Required scalar fields ``old_log_prob``, ``values``,
    ``rewards`` and ``dones`` have shape [T, N] (a trailing singleton dimension
    is also accepted). ``last_values`` evaluates the final successor states.

    For auto-reset environments, a truncated transition must be marked in both
    ``dones`` and ``time_outs``. Supply ``terminal_values`` evaluated from the
    actual pre-reset terminal observation, never from the reset observation.
    An ambiguous timeout is rejected rather than silently biased.
    """

    def __init__(
        self,
        policy,
        learning_rate=1e-3,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        value_loss_coef=1.0,
        entropy_coef=0.005,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        desired_kl=0.01,
        schedule="adaptive",
    ):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < clip_param < 1:
            raise ValueError("clip_param must lie strictly between 0 and 1")
        if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("gamma and lam must lie between 0 and 1")
        if not isinstance(num_learning_epochs, int) or num_learning_epochs <= 0:
            raise ValueError("num_learning_epochs must be a positive integer")
        if not isinstance(num_mini_batches, int) or num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be a positive integer")
        for name, value in (("value_loss_coef", value_loss_coef), ("entropy_coef", entropy_coef)):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if desired_kl is not None and (not math.isfinite(desired_kl) or desired_kl <= 0):
            raise ValueError("desired_kl must be None or finite and positive")
        if schedule not in ("adaptive", "fixed"):
            raise ValueError("schedule must be 'adaptive' or 'fixed'")
        self.policy = policy
        self.learning_rate = float(learning_rate)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.parameters = list(policy.trainable_parameters())
        student_parameter_ids = {id(parameter) for parameter in policy.student.parameters()}
        if any(id(parameter) in student_parameter_ids for parameter in self.parameters):
            raise ValueError("Residual optimizer must not contain student parameters")
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def _scalar(self, tensor, name, shape):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("{} must be a tensor".format(name))
        if tensor.shape == tuple(shape) + (1,):
            tensor = tensor.squeeze(-1)
        if tensor.shape != tuple(shape):
            raise ValueError("{} must have shape {}".format(name, tuple(shape)))
        if tensor.device != self.policy.log_std.device:
            raise ValueError("{} must be on the policy device".format(name))
        if not torch.isfinite(tensor).all():
            raise ValueError("{} contains nonfinite values".format(name))
        return tensor.detach()

    @torch.no_grad()
    def compute_returns(self, rollout, last_values):
        """Return [T, N] value targets and advantages using timeout-safe GAE."""
        if "rewards" not in rollout:
            raise ValueError("Rollout is missing rewards")
        rewards = rollout["rewards"]
        if not isinstance(rewards, torch.Tensor) or rewards.ndim not in (2, 3):
            raise ValueError("rewards must have shape [T, N]")
        horizon, num_envs = rewards.shape[:2]
        if horizon <= 0 or num_envs <= 0:
            raise ValueError("Rollout horizon and environment count must be positive")
        shape = (horizon, num_envs)
        rewards = self._scalar(rewards, "rewards", shape).to(dtype=self.policy.log_std.dtype)
        for name in ("values", "dones"):
            if name not in rollout:
                raise ValueError("Rollout is missing {}".format(name))
        values = self._scalar(rollout["values"], "values", shape).to(dtype=rewards.dtype)
        dones = self._scalar(rollout["dones"], "dones", shape)
        if not ((dones == 0) | (dones == 1)).all():
            raise ValueError("dones must be binary")
        dones = dones.bool()
        last_values = self._scalar(last_values, "last_values", (num_envs,)).to(dtype=rewards.dtype)
        time_outs = torch.zeros_like(dones)
        if "time_outs" in rollout:
            time_outs = self._scalar(rollout["time_outs"], "time_outs", shape)
            if not ((time_outs == 0) | (time_outs == 1)).all():
                raise ValueError("time_outs must be binary")
            time_outs = time_outs.bool()
            if (time_outs & ~dones).any():
                raise ValueError("Every timeout must also be marked done")
        terminal_values = torch.zeros_like(values)
        if time_outs.any():
            if "terminal_values" not in rollout:
                raise ValueError("Timeouts require terminal_values from pre-reset terminal observations")
            terminal_values = self._scalar(rollout["terminal_values"], "terminal_values", shape).to(dtype=rewards.dtype)

        advantages = torch.empty_like(values)
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(horizon)):
            next_value = last_values if step == horizon - 1 else values[step + 1]
            bootstrap = torch.where(dones[step], torch.zeros_like(next_value), next_value)
            bootstrap = torch.where(time_outs[step], terminal_values[step], bootstrap)
            delta = rewards[step] + self.gamma * bootstrap - values[step]
            # A timeout bootstraps value, but must not cross into a reset episode.
            advantage = delta + self.gamma * self.lam * (~dones[step]).to(values.dtype) * advantage
            advantages[step] = advantage
        returns = advantages + values
        if not torch.isfinite(returns).all() or not torch.isfinite(advantages).all():
            raise FloatingPointError("GAE produced nonfinite values")
        return returns, advantages

    def _set_learning_rate(self, value):
        self.learning_rate = float(value)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self, rollout, last_values):
        """Run PPO over every transition, including uneven final minibatches.

        Optional ``old_mean`` and ``old_std`` enable exact normal-distribution
        KL diagnostics/adaptation. Otherwise a nonnegative sampled KL estimate
        is used. Return sample-weighted losses and update counts.
        """
        returns, advantages = self.compute_returns(rollout, last_values)
        horizon, num_envs = returns.shape
        shape = (horizon, num_envs)
        batch_size = horizon * num_envs
        tensors = {}
        for name, width in (("residual_obs", self.policy.residual_obs_dim), ("latent_actions", self.policy.num_actions)):
            if name not in rollout:
                raise ValueError("Rollout is missing {}".format(name))
            tensor = rollout[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape + (width,):
                raise ValueError("{} must have shape [T, N, {}]".format(name, width))
            if tensor.device != self.policy.log_std.device or tensor.dtype != self.policy.log_std.dtype:
                raise ValueError("{} must have the policy's device and dtype".format(name))
            if not torch.isfinite(tensor).all():
                raise ValueError("{} contains nonfinite values".format(name))
            tensors[name] = tensor.detach().reshape(batch_size, width)
        if "old_log_prob" not in rollout:
            raise ValueError("Rollout is missing old_log_prob")
        old_log_prob = self._scalar(rollout["old_log_prob"], "old_log_prob", shape).reshape(-1)
        old_values = self._scalar(rollout["values"], "values", shape).reshape(-1)
        if ("old_mean" in rollout) != ("old_std" in rollout):
            raise ValueError("old_mean and old_std must be supplied together")
        for name in ("old_mean", "old_std"):
            if name in rollout:
                tensor = rollout[name]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != shape + (self.policy.num_actions,):
                    raise ValueError("{} must have shape [T, N, num_actions]".format(name))
                if tensor.device != self.policy.log_std.device or tensor.dtype != self.policy.log_std.dtype:
                    raise ValueError("{} must have the policy's device and dtype".format(name))
                if not torch.isfinite(tensor).all() or (name == "old_std" and not (tensor > 0).all()):
                    raise ValueError("{} must be finite (and standard deviations positive)".format(name))
                tensors[name] = tensor.detach().reshape(batch_size, self.policy.num_actions)

        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std(unbiased=False)
        if not torch.isfinite(advantage_mean) or not torch.isfinite(advantage_std):
            raise FloatingPointError("Advantage normalization statistics became nonfinite")
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        self.policy.train()
        totals = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        samples = 0
        updates = 0
        for _ in range(self.num_learning_epochs):
            permutation = torch.randperm(batch_size, device=returns.device)
            # tensor_split includes remainder and allows exactly the requested
            # count (or one nonempty minibatch per sample for tiny rollouts).
            for indices in torch.tensor_split(permutation, min(self.num_mini_batches, batch_size)):
                result = self.policy.evaluate_latent(tensors["residual_obs"][indices], tensors["latent_actions"][indices])
                log_ratio = result["log_prob"] - old_log_prob[indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    if "old_mean" in tensors:
                        old_mean, old_std = tensors["old_mean"][indices], tensors["old_std"][indices]
                        kl = (torch.log(result["std"] / old_std) +
                              (old_std.square() + (old_mean - result["mean"]).square()) /
                              (2.0 * result["std"].square()) - 0.5).sum(-1).mean().clamp_min(0)
                    else:
                        kl = (ratio - 1.0 - log_ratio).mean().clamp_min(0)
                    if not torch.isfinite(kl):
                        raise FloatingPointError("PPO KL became nonfinite")
                    if self.schedule == "adaptive" and self.desired_kl is not None:
                        if kl > 2.0 * self.desired_kl:
                            self._set_learning_rate(max(1e-5, self.learning_rate / 1.5))
                        elif 0 < kl < 0.5 * self.desired_kl:
                            self._set_learning_rate(min(1e-2, self.learning_rate * 1.5))
                surrogate = -advantages[indices] * ratio
                clipped = -advantages[indices] * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.maximum(surrogate, clipped).mean()
                value_error = (result["value"] - returns[indices]).square()
                if self.use_clipped_value_loss:
                    clipped_value = old_values[indices] + (result["value"] - old_values[indices]).clamp(-self.clip_param, self.clip_param)
                    value_error = torch.maximum(value_error, (clipped_value - returns[indices]).square())
                value_loss = value_error.mean()
                entropy = result["entropy"].mean()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became nonfinite")
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                if not torch.isfinite(gradient_norm):
                    self.optimizer.zero_grad()
                    raise FloatingPointError("PPO gradients became nonfinite")
                self.optimizer.step()
                for parameter in self.parameters:
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError("PPO optimizer produced nonfinite parameters")
                count = indices.numel()
                samples += count
                updates += 1
                for name, value in (("value_loss", value_loss), ("surrogate_loss", surrogate_loss), ("entropy", entropy), ("kl", kl)):
                    totals[name] += value.item() * count
        metrics = {name: value / samples for name, value in totals.items()}
        metrics.update(learning_rate=self.learning_rate, samples=samples, updates=updates)
        return metrics

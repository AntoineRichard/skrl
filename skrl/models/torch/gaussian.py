from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Normal


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)
EPS = 1e-6

class GaussianMixin:
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model)

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: ``True``)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True (default: ``-20``)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True (default: ``2``)
        :type max_log_std: float, optional
        :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                          Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                          function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
        :type reduction: str, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises ValueError: If the reduction method is not valid

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from skrl.models.torch import Model, GaussianMixin
            >>>
            >>> class Policy(GaussianMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0",
            ...                  clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            ...
            ...         self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, self.num_actions))
            ...         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
            ...
            ...     def compute(self, inputs, role):
            ...         return self.net(inputs["states"]), self.log_std_parameter, {}
            ...
            >>> # given an observation_space: gymnasium.spaces.Box with shape (60,)
            >>> # and an action_space: gymnasium.spaces.Box with shape (8,)
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
              (net): Sequential(
                (0): Linear(in_features=60, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=8, bias=True)
              )
            )
        """
        self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._g_clip_actions:
            # Ensure that the action space is a Bounded or the rescaling will explode
            # Preven infinity values in action space and replace them with -1.0 and 1.0
            for i, _ in enumerate(self.action_space.low):
                if self.action_space.low[i] == -float("inf"):
                    self.action_space.low[i] = -1.0
            for i, _ in enumerate(self.action_space.high):
                if self.action_space.high[i] == float("inf"):
                    self.action_space.high[i] = 1.0
            self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

            print("Clip actions:", self._g_clip_actions)
            print("Clip actions min:", self._g_clip_actions_min)
            print("Clip actions max:", self._g_clip_actions_max)


        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std
        self._g_log_std_max = max_log_std

        self._g_log_std = None
        self._g_num_samples = None
        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states/observations to mean actions and log standard deviations
        print("Getting mean actions and log std")
        mean_actions, log_std, outputs = self.compute(inputs, role)
        print("Mean actions:", mean_actions[:5])
        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]

        # If the actions are clipped, apply the tanh function to the mean actions (ensuring that the gaussian
        # distribution is centered between -1 and 1)
        if self._g_clip_actions:
            mean_actions = torch.tanh(mean_actions)

        # distribution
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._g_distribution.rsample()
        print("Actions sampled:", actions[:5])

        # clip actions
        if self._g_clip_actions:
            scaled_actions = inputs.get("taken_actions", None)
            # if the actions are coming from the buffer, we need to transform them
            if scaled_actions is not None:
                print("Scaled actions:", scaled_actions[:5])
                # rescale actions to be in the range [-1, 1]
                unscaled_actions = ((scaled_actions - self._g_clip_actions_min) / (self._g_clip_actions_max - self._g_clip_actions_min)) * 2.0 - 1.0
                print("Unscaled actions:", unscaled_actions[:5])
                gaussian_actions = self.inverse_tanh(unscaled_actions)
                print("Gaussian actions:", gaussian_actions[:5])
                log_prob = self._g_distribution.log_prob(gaussian_actions) - torch.log(1 - unscaled_actions*unscaled_actions + EPS)
                print("log_prob:", log_prob[:5])
            else:
                log_prob = self._g_distribution.log_prob(actions)
                print("log_prob:", log_prob[:5])
            # Scale the actions to the desired range
            actions = ((torch.tanh(actions) + 1) / 2.0) * (self._g_clip_actions_max - self._g_clip_actions_min) + self._g_clip_actions_min
            print("Actions after scaling:", actions[:5])
            mean_actions = ((mean_actions + 1) / 2.0) * (self._g_clip_actions_max - self._g_clip_actions_min) + self._g_clip_actions_min   
            print("Mean actions after scaling:", mean_actions[:5])
        else:
            # log of the probability density function
            log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions

        # Look for NaN values in the actions and log probability
        if torch.isnan(actions).any():
            raise ValueError("NaN values found in actions")
        print("Exiting act method")
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._g_log_std.repeat(self._g_num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._g_distribution

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """Inverse hyperbolic tangent function
        
        :param x: Input tensor
        :type x: torch.Tensor"
        :return: Inverse hyperbolic tangent of the input tensor"
        :rtype: torch.Tensor"

        Example::

            >>> x = torch.tensor([0.5, 0.8])
            >>> result = GaussianMixin.atanh(x)
            >>> print(result)
            tensor([0.5493, 1.0986])
        """
        return 0.5 * (x.log1p() - (-x).log1p())
    
    @staticmethod
    def inverse_tanh(y: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse hyperbolic tangent function

        :param y: Input tensor
        :type y: torch.Tensor
        :return: Inverse hyperbolic tangent of the input tensor
        :rtype: torch.Tensor

        Example::
            >>> y = torch.tensor([0.5, 0.8])
            >>> result = GaussianMixin.inverse_tanh(y)
            >>> print(result)
            tensor([0.5493, 1.0986])
        """
        eps = torch.finfo(y.dtype).eps
        return GaussianMixin.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))
from typing import Optional, Union, Mapping, Sequence, Tuple, Any, Callable

import gym
import gymnasium
import collections
import numpy as np
import contextlib

import jax
import jaxlib
import jax.numpy as jnp
import flax

from skrl import logger


class StateDict(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, **kwargs):
        return cls(apply_fn=apply_fn, params=params, **kwargs)


class Model(flax.linen.Module):
    observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space]
    action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space]
    device: Optional[Union[str, jaxlib.xla_extension.Device]] = None

    def __init__(self,
                 observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 device: Optional[Union[str, jaxlib.xla_extension.Device]] = None,
                 parent: Optional[Any] = None,
                 name: Optional[str] = None) -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (jaxlib.xla_extension.Device): Device to be used for the computations
        - ``observation_space`` (int, sequence of int, gym.Space, gymnasium.Space): Observation/state space
        - ``action_space`` (int, sequence of int, gym.Space, gymnasium.Space): Action space
        - ``num_observations`` (int): Number of elements in the observation/state space
        - ``num_actions`` (int): Number of elements in the action space

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space, gymnasium.Space
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        :param parent: The parent Module of this Module (default: ``None``).
                       It is a Flax reserved attribute
        :type parent: str, optional
        :param name: The name of this Module (default: ``None``).
                     It is a Flax reserved attribute
        :type name: str, optional

        Custom models should override the ``act`` method::

            import flax.linen as nn
            from skrl.models.jax import Model

            class CustomModel(Model):
                def __init__(self, observation_space, action_space, device=None, **kwargs):
                    super().__init__(observation_space, action_space, device, **kwargs)

                @nn.compact
                def __call__(self, inputs, role):
                    x = nn.relu(nn.Dense(64)(inputs["states"]))
                    x = nn.relu(nn.Dense(self.num_actions)(x))
                    return x, None, {}
        """
        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jaxlib.xla_extension.Device) else jax.devices(device)[0]

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else self._get_space_size(observation_space)
        self.num_actions = None if action_space is None else self._get_space_size(action_space)

        self.state_dict: StateDict
        self.training = False

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ReservedModuleAttributeError
        self.parent = parent
        self.name = name

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        super().__post_init__()

    def init_state_dict(self, key, inputs, role):
        self.state_dict = StateDict.create(apply_fn=self.apply,
                                           params=self.init(key, inputs, role),)

    def _get_space_size(self,
                        space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                        number_of_elements: bool = True) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, sequence of int, gym.Space, or gymnasium.Space
        :param number_of_elements: Whether the number of elements occupied by the space is returned (default: ``True``).
                                   If ``False``, the shape of the space is returned. It only affects Discrete spaces
        :type number_of_elements: bool, optional

        :raises ValueError: If the space is not supported

        :return: Size of the space (number of elements)
        :rtype: int

        Example::

            # from int
            >>> model._get_space_size(2)
            2

            # from sequence of int
            >>> model._get_space_size([2, 3])
            6

            # Box space
            >>> space = gym.spaces.Box(low=-1, high=1, shape=(2, 3))
            >>> model._get_space_size(space)
            6

            # Discrete space
            >>> space = gym.spaces.Discrete(4)
            >>> model._get_space_size(space)
            4
            >>> model._get_space_size(space, number_of_elements=False)
            1

            # Dict space
            >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)),
            ...                          'b': gym.spaces.Discrete(4)})
            >>> model._get_space_size(space)
            10
            >>> model._get_space_size(space, number_of_elements=False)
            7
        """
        size = None
        if type(space) in [int, float]:
            size = space
        elif type(space) in [tuple, list]:
            size = np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                if number_of_elements:
                    size = space.n
                else:
                    size = 1
            elif issubclass(type(space), gym.spaces.Box):
                size = np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
        elif issubclass(type(space), gymnasium.Space):
            if issubclass(type(space), gymnasium.spaces.Discrete):
                if number_of_elements:
                    size = space.n
                else:
                    size = 1
            elif issubclass(type(space), gymnasium.spaces.Box):
                size = np.prod(space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
        if size is None:
            raise ValueError("Space type {} not supported".format(type(space)))
        return int(size)

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.training = True
        elif mode == "eval":
            self.training = False
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def act(self,
            inputs: Mapping[str, Union[jnp.ndarray, Any]],
            role: str = "") -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Mapping[str, Union[jnp.ndarray, Any]]]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        This method is currently implemented by the helper models (**GaussianModel**, etc.).
        The classes that inherit from the latter must only implement the ``.compute()`` method

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically jnp.ndarray
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Child class must implement this method

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of jnp.ndarray, jnp.ndarray or None, and dictionary
        """
        return self.apply(self.state_dict.params, inputs, role)

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (``parameters.requires_grad = False``)
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``)

        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them (default: ``True``)
        :type freeze: bool, optional

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        pass

    def update_parameters(self, model: flax.linen.Module, polyak: float = 1) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: Model used to update the internal parameters
        :type model: flax.linen.Module (skrl.models.jax.Model)
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        :type polyak: float, optional

        Example::

            # hard update (from source model)
            >>> model.update_parameters(source_model)

            # soft update (from source model)
            >>> model.update_parameters(source_model, polyak=0.005)
        """
        with contextlib.nullcontext():
            # hard update
            if polyak == 1:
                self.state_dict = self.state_dict.replace(params=model.state_dict.params)
            # soft update
            else:
                # params = optax.incremental_update(model.state_dict.params, self.state_dict.params, polyak)
                params = jax.tree_util.tree_map(lambda params, model_params: polyak * model_params + (1 - polyak) * params,
                                                self.state_dict.params, model.state_dict.params)
                self.state_dict = self.state_dict.replace(params=params)

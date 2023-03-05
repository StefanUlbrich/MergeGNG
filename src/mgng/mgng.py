# -*- coding: utf-8 -*-
r"""
Implementation of the Merged Growing Neural Gas for temporal data.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2020, All rights reserved."
# __credits__ = []
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Stefan Ulbrich"
__status__ = "alpha"
__date__ = "2020-01-27"
__all__ = ["MergeGNG"]

import logging
from typing import Any, List, Tuple

import numpy as np
from attr import define, field
from numpy.linalg import norm
from numpy.typing import NDArray

from mgng.helpers import get_dymmy_2d_data
from mgng.validators import is_greater_zero, is_weight_factor, repr_ndarray

logger = logging.getLogger(__name__)


@define
class MergeGNG:
    r"""
    Class that represents a Merge Growing Neural Gas.

    Differences to default implementation

    * Neurons all kept in memory to allow numpy operations
    * Introduce half life and threshold for connections (planned). For now, only decrease.
    * Adaptation rate should depend on connection strength
    * Introduce method (half-life?, decay on all synapses) to remove very old movements (I am sure that the original
      implementation allows for orphans)
    * Compare with an approach of a regular neural gas with a refactory time
    * Add threshold for an activity to trigger a new neuron (hey, make a fifo). I really want to enforce this. If a
      neuron gets activated 3 times in a row it's tiem for a new neuron!
    * REALLY CONSIDER REMOVING THE DIOGONAL ELEMENTS! Implement neighbor learn rate ..
      maybe weighted by synapse strength
    * Activity is never 0 unless it is a never used neuron or one removed because it had no connections

    * Todo: did we remove neurons without connections?

    Parameters
    ----------
    n_neurons: int
        Max. number of neurons
    n_dim: int
        Output dimension (of the feature space).
    connection_decay: float
        Hyper parameter for influencing the decay of neuron connections. NOT USED RIGHT NOW
    temporal_influence: float
        The influence of the temporal memory on finding the winning neuron
    memory_weight: float
        Determines the influence of past samples in the sequence. (Kinda "how long" it looks back into the past).
    life_span: int
        How many iterations until a synapse is deleted.
        Note .. synapses of the winning neuron only are decayed (it forgets "wrong" neighbors)
    max_activity:
        Maximal activity allowed for a neuron (cf. refractory period). If a neuron is more active than this threshold,
        a new neuron is inserted between it and the second most active neuron. Note that each time the neuron is the
        winning neuron, it's activity level is increased by 1.0 and then continuously decreases continuously in
        each iteration (c.f. `decrease_activity`)
        This is different to the reference paper where the network gros in regular intervals.
        Our approach reflects a more "on demand" approach and prevents the network from growing unnecessarily.
    decrease_activity: float
        Less important .. the activity decreases exponentially ... only interesting if there are only few iterations
        between reccuring sequences

    learn_rate: float
        lorem ipsum
    learn_rate_neighbors: float
        lorem ipsum
    delta: float
        Need a better name, right? It's a parameter that decides the neuron's activity if a new neuron is added.
    allow_removal: float
        lorem ipsum

    Attributes
    ----------

    _weights: np.ndarray, :math:`n_{\text{neurons}} \times n_{\text{dim}}`
        The amount of neurons is constant in this implementation for simplicity
        reasons and speed (block operations).

    """
    # FIXME: Until pylint + attrs work nicely together (or pylint and typehints)
    # pylint: disable=unsubscriptable-object,unsupported-assignment-operation,no-member

    # Note that comment type hints are used to ensure Python 3.5 support _and_ VSCode autocomplete
    # See https://www.attrs.org/en/stable/types.html#mypy

    n_neurons: int = 100
    n_dim: int = 3
    connection_decay: float = 0.1
    temporal_influence: float = field(default=0.5, validator=[is_weight_factor])
    memory_weight: float = field(default=0.5, validator=[is_weight_factor])
    life_span: int = 10
    learn_rate: float = field(default=0.2, validator=[is_weight_factor])
    learn_rate_neighbors: float = field(default=0.2, validator=[is_weight_factor])
    decrease_activity: float = field(default=0.8, validator=[is_weight_factor])
    # TODO find a goood name for delta (in fact, update all names)
    delta: float = field(default=0.8, validator=[is_weight_factor])

    max_activity: float = field(default=2.0, validator=[is_greater_zero])
    allow_removal: bool = True

    # I don't want this parameter truth to be told
    creation_frequency: int = 5

    # Private variables. Default initializers depend on n_neurons and n_dim. The order matters!
    _weights: NDArray[np.floating[Any]] = field(init=False, repr=repr_ndarray)
    _context: NDArray[np.floating[Any]] = field(init=False, repr=repr_ndarray)
    _connections: NDArray[np.floating[Any]] = field(init=False, repr=repr_ndarray)
    _counter: NDArray[np.floating[Any]] = field(init=False, repr=False)
    _global_context: NDArray[np.floating[Any]] = field(init=False, repr=repr_ndarray)

    debug: bool = False
    past: List[List[NDArray[np.floating[Any]]]] = field(init=False, factory=list, repr=False)

    @_weights.default
    def _default_weights(self) -> NDArray[np.floating[Any]]:
        return np.random.rand(self.n_neurons, self.n_dim)

    @_context.default
    def _default_context(self) -> NDArray[np.floating[Any]]:
        return np.random.rand(self.n_neurons, self.n_dim)

    @_global_context.default
    def _default_global_context(self) -> NDArray[np.floating[Any]]:
        return np.random.rand(self.n_dim)

    @_connections.default
    def _default_connections(self) -> NDArray[np.floating[Any]]:
        # XXX: we keep all neurons in memory such that we can do block operations
        return np.zeros((self.n_neurons, self.n_neurons))

    @_counter.default
    def _default_counter(self) -> NDArray[np.floating[Any]]:
        return np.zeros(self.n_neurons)

    def _decay(self, first: int) -> None:
        """
        Decrease all synapses of a neuron but don't allow negative synampses.

        Parameters
        ----------
        first : int
            Index of the neuron
        """

        def decay_vector(vector: NDArray[np.floating[Any]]) -> None:
            vector -= 1.0 / self.life_span
            vector[vector < 0] = 0

        logger.debug("Decaying connections before:\n%s", self._connections)
        decay_vector(self._connections[first, :])
        decay_vector(self._connections[:, first])
        logger.debug("after \n%s", self._connections)

    def kill_orphans(self) -> None:
        """Remove orphaned neurons."""
        # argwhere not suitable for indexing

        orphans = np.nonzero(np.sum(self._connections, axis=1) == 0)

        logger.debug("Orphans: %s", orphans)
        logger.debug("counter before: %s", self._counter)
        self._counter[orphans] = 0
        logger.debug("counter after: %s", self._counter)

    def adapt(self, sample: NDArray[np.floating[Any]]) -> Tuple[int, int]:
        """
        Single adaptation step

        Parameters
        ----------
        sample : NDArray[np.floating[Any]], shape: :math:`(n_{\text{dim}},)`
            A single sample.

        Returns
        -------
        Tuple[int,int]
            Optionally returns the first and second winning
            neurons used for Hebbian learning.
        """

        if self.debug:
            self.past.append(
                [
                    self._weights.copy(),
                    self._context.copy(),
                    self._connections.copy(),
                    self._counter.copy(),
                    self.get_active_weights().copy(),
                    sample.copy(),
                ]
            )

        dist_weights = norm(self._weights - sample[np.newaxis, :], axis=-1)  # type: ignore
        dist_context = norm(self._context - self._global_context, axis=-1)  # type: ignore

        logger.debug("|weights| = %s, |context| = %s", dist_weights, dist_context)
        logger.debug("connections at beginning:\n%s", self._connections)

        # Todo remove this variable
        distance = (1 - self.temporal_influence) * dist_weights + self.temporal_influence * dist_context

        winners = np.argsort(
            distance
            # (1 - self.temporal_influence) * dist_weights + self.temporal_influence * dist_context
        )

        logger.debug("winners %s, %s", winners, distance[winners])

        first = winners[0]
        second = winners[1]

        assert distance[first] <= distance[second]

        old_global_context = self._global_context.copy()

        # fmt: off
        self._global_context = (
            (1 - self.memory_weight) * self._weights[first, :]
                + self.memory_weight * self._context[first, :]
        )
        # fmt: on

        self._decay(first)  # Let's decay first so that the new new connection has maximal value

        logger.debug("Adding edge to:\n%s", self._connections)

        # Symmetric connection matrix
        self._connections[first, second] = self._connections[second, first] = 1.0
        # Diagonal only needed when the connection values are used in the update rule below.
        # then it should probably not be 1.0
        # self._connections[first, first] = 1.0

        logger.debug("after\n%s", self._connections)

        # Needs to be after new connections are created. otherwise the counter of first might be reset
        self.kill_orphans()

        self._weights[first, :] += self.learn_rate * (sample - self._weights[first, :])
        self._context[first, :] += self.learn_rate * (old_global_context - self._context[first, :])

        neighbors = np.nonzero(self._connections[first, :])  # == non-zeros in the row

        logger.debug("winning neuron's neighbors %s", neighbors)

        # Suggestion: weight adaptation by age of synapse
        # self._weights[neighbors, :] += self.learn_rate * self._connections[first, neighbors] *\
        # (sample - self._weights[neighbors, :])
        # self._context[neighbors, :] += self.learn_rate * self._connections[first, neighbors] *\
        #  (old_global_context - self._context[neighbors, :])

        self._weights[neighbors, :] += self.learn_rate_neighbors * (sample - self._weights[neighbors, :])
        self._context[neighbors, :] += self.learn_rate_neighbors * (old_global_context - self._context[neighbors, :])

        self._counter[first] += 1
        logger.debug("New counter: %s, \n%s", self._counter, self._connections)

        return first, second

    def grow(self) -> None:
        """
        Entropy maximization by adding neurons in regions of high activity.

        Note: this picks the weakest neuron. TODO this needs to be implemented too!
        """

        # Warning .. error when the max neuron does not have neighbors (pretty much impossible)

        most = np.argmax(self._counter)
        its_neighbors = np.nonzero(self._connections[most, :])  # e.g., (array([0, 2]),)
        logger.debug(its_neighbors)
        most_active_neighbors = np.argsort(self._counter[its_neighbors])  # get the activations and sort them
        logger.debug(most_active_neighbors)
        # The last entry is the winning neuron itself (WATCH OUT unless the diagonal is zero!, in that case, use -1)
        neighbor = its_neighbors[0][most_active_neighbors[-1]]

        logger.debug(
            "Most active: %d\nIts neighbors: %s, Its most active neighbor: %s",
            most,
            its_neighbors,
            neighbor,
        )

        new = self.kill_weakest()

        self.delta = 0.8  # Yet another parameter :-(

        self._weights[new, :] = 0.5 * (self._weights[most, :] + self._weights[neighbor, :])
        self._context[new, :] = 0.5 * (self._context[most, :] + self._context[neighbor, :])

        self._counter[new] = self.delta * (self._counter[most] + self._counter[neighbor])
        self._counter[most] *= 1 - self.delta
        self._counter[neighbor] *= 1 - self.delta

        self._connections[most, neighbor] = self._connections[neighbor, most] = 0.0
        self._connections[new, neighbor] = self._connections[neighbor, new] = 1.0
        self._connections[most, new] = self._connections[new, most] = 1.0

    def kill_weakest(self) -> np.signedinteger[Any]:
        """
        Finds the weakest neuron (or the first with zero activity in the list)
        and returns its index

        Returns
        -------
        int
            Index of the neuron
        """
        least = np.argmin(self._counter)  # That is a good metric? Probably yes

        logger.info("Least active neuron: %d, value: %f", least, self._counter[least])
        logger.info("Did it have conntections?\n%s", self._connections[least, :])

        if np.sum(self._connections[least, :]) > 0:
            logger.warning("Killing existing neuron. Consider larger pool! Activity: %f", self._counter[least])

        # Remove connections:
        self._connections[least, :] = 0.0
        self._connections[:, least] = 0.0
        return least

    def learn(self, samples: NDArray[np.floating[Any]], epochs: int) -> None:
        r"""
        Batch learning

        Parameters
        ----------
        samples : np.ndarray
            Row array of points. Shape :math:`n_{\text{samples}} \times n_{\text{dim}}`.
        epochs : int
            Number of repetitions.
        """

        assert samples.shape[1] == self.n_dim

        for epoch in range(epochs):
            for i, sample in enumerate(samples):
                logger.info("\n\n\n%s\nSample: %d, Epoch: %d", "*" * 24, i, epoch)
                self.adapt(sample)
                self._counter *= self.decrease_activity

                if np.max(self._counter) >= self.max_activity:  # type: ignore
                    self.grow()
                # if i % self.creation_frequency == self.creation_frequency - 1:
                #     # Make this a factor depending on the activity of neurons
                #     self.grow()

    def get_active_weights(self) -> NDArray[np.floating[Any]]:
        """Return active and inactive weights sorted."""

        # Watchout there is an argwhere not an nonzero

        # TODO check whether vstack or hstack
        return np.vstack(
            (
                self._weights[np.nonzero(self._counter > 0), :],
                self._weights[np.nonzero(self._counter <= 0), :],
            )
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    mgng = MergeGNG(connection_decay=0.1)

    print(mgng)

    temp = mgng.n_neurons

    X = get_dymmy_2d_data(20)

    print(repr_ndarray(X))

    plt.plot(X[0, :], X[1, :])
    plt.show()

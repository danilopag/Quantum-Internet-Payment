from netsquid.qubits.state_sampler import StateSampler
import numpy as np
from scipy.sparse import csr_matrix


class RepchainSampler(StateSampler):
    """
    Object holding properties state, midpoint_outcome, and generation_duration from simulation data,
    intended to be sampled from.

    This object is initialised with a dataframe and desired length. StateSampler for only the
    abovementioned properties is created. A tree structure is used for efficient extraction of relevant data
    from the dataframe and subsequently also for recursively creating the sampler. Calling the sample()
    method simply returns the state, midpoint_outcome, and generation_duration - a combination that is present
    in the simulation data.

    The state sampler created in this class can be also used for `StateDeliverySampler`, note that then only
    the created state_sampler should be passed on, not the whole RepchainSampler object. (See example.)

    Parameters
    ----------
    repchain_dataframe : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`
        This dataframe holds the data that needs to be sampled.
        Note: only dataframes that have a column with length are supported (i.e. dataframes with samples
        for one specified length with the length only in baseline parameters are not yet supported).
    length : int
        Only data with this length will be considered for sampling.

    Example
    -------

    Given a dataframe, this is how sampling would work.

    >>> repchain_sampler = RepchainSampler(placeholder_dataframe, 1)
    >>> state, midpoint_outcome, generation_duration = repchain_sampler.sample()

    If the state sampler created from within this class needs to be passed on to `StateDeliverySampler`,
    the `StateDeliverySampler` should be initialised with `repchain_sampler.state_sampler`.

    Illustration of the data structures given placeholder values:
    -------------------------------------------------------------

    Built-ins:

    .. code-block:: python

    repchain_tree = {
                'state_1': [5, {
                                'midpoint_outcome_1': [4, {
                                                            'generation_duration_1': 3,
                                                            'generation_duration_2': 1}],
                                'midpoint_outcome_2': [1, {
                                                            'generation_duration_3': 1}]}],
                'state_2': [5, {
                                'midpoint_outcome_4': [5, {
                                                            'generation_duration_4': 5}]}]}

    Tree:

    .. code-block:: text

    level_0:
                                        +--------------+                                         +--------------+
                                        | [5, state_1] |                                         | [5, state_2] |
                                        +--------------+                                         +--------------+
                                           //       \\                                                  ||
                                          //         \\                                                 ||
                                         //           \\                                                ||
    level_1:                            //             \\                                               ||
                +-------------------------+          +-------------------------+            +-------------------------+
                | [4, midpoint_outcome_1] |          | [1, midpoint_outcome_2] |            | [5, midpoint_outcome_3] |
                +-------------------------+          +-------------------------+            +-------------------------+
                        //        \\                           \\                                       ||
                       //          \\                           \\                                      ||
                      //            \\                           \\                                     ||
    level_2:         //              \\                           \\                                    ||
    +--------------------------+ +--------------------------+ +--------------------------+ +--------------------------+
    |[3, generation_duration_1]| |[1, generation_duration_2]| |[1, generation_duration_3]| |[5, generation_duration_4]|
    +--------------------------+ +--------------------------+ +--------------------------+ +--------------------------+

    StateSampler:

    .. code-block:: text

                                                StateSampler
                                                +--------------------------------+
                                                | [StateSampler1, StateSampler2] |
                                                |           [0.5, 0.5]           |
                                                |          [None, None]          |
                                                +--------------------------------+

                StateSampler1                                                   StateSampler2
                +------------------------------------------+                    +----------------------+
                |      [StateSampler3, StateSampler4]      |                    |    [StateSampler5]   |
                |                [0.8, 0.2]                |                    |          [1]         |
                | [midpoint_outcome_1, midpoint_outcome_2] |                    | [midpoint_outcome_3] |
                +------------------------------------------+                    +----------------------+

    StateSampler3                                      StateSampler4                        StateSampler5
    +------------------------------------------------+ +-------------------------+          +-------------------------+
    |            [state_1, state_1]                  | |      [state_1]          |          |      [state_2]          |
    |               [0.75, 0.25]                     | |         [1]             |          |         [1]             |
    | [generation_duration_1, generation_duration_2] | | [generation_duration_3] |          | [generation_duration_4] |
    +------------------------------------------------+ +-------------------------+          +-------------------------+
    """

    def __init__(self, repchain_dataframe, length):
        # make sure dataframe holds suitable data
        self.check_dataframe_correctness(repchain_dataframe)

        self.length = length
        self.size = 0

        # construct the tree structure
        self._states = []
        self._tree = {}
        self.construct_tree(repchain_dataframe.dataframe)

        # construct the state sampler
        self.state_sampler = self.create_state_sampler()

    def sample(self, exclude_none=False, rng=None):
        """
        Overrides the method in StateSampler such that only the required properties are returned.
        """
        if self.size == 0:
            print("There is no data to sample from.")
        state, prob, labels = self.state_sampler.sample(exclude_none, rng)
        return state, labels[0], labels[1]

    @staticmethod
    def check_dataframe_correctness(dataframe):
        """
        Checks whether the dataframe holds data suitable for sampling.

        Checks whether required properties are present and whether the dataframe
        contains data from simulation of elementary link.

        Parameters
        ----------
        :param dataframe : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`

        Note
        ----
        This method is static because it is unnecessary to store the entire dataframe as a property of this class.
        """
        columns = dataframe.dataframe.columns
        # all required properties are there
        assert ('length' in columns), "Length property is not in simulation data"
        assert ('state' in columns), "State property is not in simulation data"
        assert ('midpoint_outcome_0' in columns), "Midpoint outcome is not in simulation data"
        assert ('generation_duration' in columns), "Generation duration is not in simulation data"

        # only elementary link simulation
        assert ('swap_outcome_0' not in columns), \
            "Simulation data is not from elementary link; there should be no swap outcome"
        assert (dataframe.baseline_parameters.get('num_repeaters') == 0), \
            "Simulation data is not from elementary link; there should not be any repeaters"

    def construct_tree(self, dataframe):
        """
        Constructs the tree from simulation data in the dataframe.

        Goes through every row in the dataframe, checks if those values should be included in the sampler,
        and updates the tree with the new values if length is the same as the desired value.

        Parameters
        ----------
        :param dataframe : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`
        """
        for i in range(dataframe.shape[0]):
            # if row contains info with correct length, get relevant values
            if dataframe.at[i, 'length'] == self.length:
                state = dataframe.at[i, 'state']
                if type(state) == csr_matrix:
                    # Convert to dense matrix and then to array
                    state = np.squeeze(np.asarray(csr_matrix.todense(state)))
                values = [state,
                          dataframe.at[i, 'midpoint_outcome_0'],
                          dataframe.at[i, 'generation_duration']]
                # update the tree given new values
                self.update_tree(values)
                self.size += 1

    def update_tree(self, values):
        """
        Updates the tree with new values.

        Parameters
        ----------
        :param values : list of Any
            Values are in order: state, midpoint_outcome, generation_duration.
        """
        state_not_in_tree = True

        # check if there is a "close" state already in the tree
        for state in self._states:
            if np.isclose(state, values[0]).all():  # for now set to default values, rtol=1e-05 and atol=1e-08
                # if there is a close state, values will be updated with the state already in the tree
                values[0] = state
                state_not_in_tree = False
                break
        if state_not_in_tree:
            # if the state is not yet in tree, add to states
            self._states.append(values[0])

        values[0] = self.array_to_tuple(values[0])  # convert to tuple so it's hashable

        tree = self._tree

        if tree.get(values[0]) is None:
            # state is not in tree: create the whole path
            tree[values[0]] = [1, {}]
            level1 = tree.get(values[0])[1]
            level1[values[1]] = [1, {}]
            level2 = level1.get(values[1])[1]
            level2[values[2]] = 1

        else:
            # state is in tree: increment freq count, move on to lower level
            tree.get(values[0])[0] += 1
            level1 = tree.get(values[0])[1]

            if level1.get(values[1]) is None:
                # midpoint_outcome is not in tree: create the rest of the path
                level1[values[1]] = [1, {}]
                level2 = level1.get(values[1])[1]
                level2[values[2]] = 1

            else:
                # midpoint_outcome is in tree: increment freq count, move on to lower level
                level1.get(values[1])[0] += 1
                level2 = level1.get(values[1])[1]

                if level2.get(values[2]) is None:
                    # generation_duration is not in tree: add count
                    level2[values[2]] = 1
                else:
                    # generation_duration is in tree: increment count
                    level2[values[2]] += 1

        self._tree = tree

    def create_state_sampler(self):
        """
        Creates StateSampler from the data stored in the tree.

        This creates a StateSampler that also recursively stores information about the properties
        we want to sample as labels (midpoint_outcome, generation_duration).

        Returns
        ------
        :return :obj:`netsquid.qubits.state_sampler.StateSampler`
        """
        states = []
        probs = []
        labels = []

        # there is StateSampler for each state
        for state in self._tree.keys():
            state_states = []
            state_probs = []
            state_labels = []

            # there is StateSampler for each midpoint_outcome
            for midpoint in self._tree.get(state)[1].keys():
                midpoint_states = []
                midpoint_probs = []
                midpoint_labels = []

                for generation_duration in self._tree.get(state)[1].get(midpoint)[1].keys():
                    # append information about this leaf, including the state
                    midpoint_states.append(self.tuple_to_array(state))
                    midpoint_probs.append(self._tree.get(state)[1].get(midpoint)[1].get(generation_duration))  # freq
                    midpoint_labels.append(generation_duration)

                # convert frequencies to probabilities (because of the constructor of StateSampler)
                midpoint_probs = list(np.array(midpoint_probs) / sum(midpoint_probs))

                # append the StateSampler for this midpoint
                state_states.append(StateSampler(midpoint_states, probabilities=midpoint_probs, labels=midpoint_labels))
                state_probs.append(self._tree.get(state)[1].get(midpoint)[0])
                state_labels.append(midpoint)

            state_probs = list(np.array(state_probs) / sum(state_probs))  # convert frequencies to probabilities

            # append the StateSampler for this state
            states.append(StateSampler(state_states, probabilities=state_probs, labels=state_labels))
            probs.append(self._tree.get(state)[0])
            labels.append(None)

        probs = list(np.array(probs) / sum(probs))  # convert frequencies to probabilities

        return StateSampler(states, probabilities=probs, labels=labels)

    @staticmethod
    def array_to_tuple(state):
        """
        Converts array to tuple.
        """
        return tuple(tuple(x) for x in state)

    @staticmethod
    def tuple_to_array(state):
        """
        Converts tuple to array.
        """
        temp = []
        for row in state:
            temp.append(list(row))
        return np.array(temp)

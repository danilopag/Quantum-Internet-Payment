import pandas
import numpy as np
import pkg_resources
from netsquid.qubits.ketstates import BellIndex

from scipy import sparse


class _StringKeyTypeDict(dict):
    """
    A dictionary where the keys may only
    be strings.
    """

    def __setitem__(self, key, val):
        if not isinstance(key, str):
            raise TypeError("Key {} not a string".format(key))
        else:
            super().__setitem__(key, val)


class RepchainDataFrameHolder:
    """Object holding results from simulations, together with metadata and methods for checking and adding new data.

    Class of objects used to store results of quantum repeater chain simulations. Contains metadata about simulations
    (which cannot be done by a pure pandas.DataFrame) and provides functionality to check and combine data.
    Specifically, it contains properties indicating the the number of nodes in the simulated
    repeater chain, the baseline parameters (parameters which are the same for all simulations of which results
    are stored) and the total number of results held. It can test data it holds for validity and consistency,
    and when data is added checks are performed first. The "combine" method can be used to combine multiple
    objects of this class into one.

    Parameters
    ----------
    number_of_nodes : int
        Number of nodes used in repeater chain simulation
        (including both repeater nodes and end nodes, but not heralding stations).
    baseline_parameters : dict, optional
        Parameters that have the same value for all simulations of which data is contained in this object.
        A key-value pair consists of a parameter name and the corresponding value (must be hashable).
    additional_packages : list of str, optional
        Relevant additional packages that are used to generate the data and are added to the `packages`.
        The version is automatically looked up based on the installed packages.
        By default the netsquid and netsquid-simulationtools versions are added.
    data : pandas.DataFrame or dict, optional
        Actual simulation data. Must at least contain the following columns/keys:
        - "state" (reduced density matrix or ket vector of quantum state delivered between end nodes (numpy.ndarray)),
        - "basis_A" (measurement basis choice of Alice, must be "X", "Y" or "Z"),
        - "basis_B" (measurement basis choice of Bob, must be "X", "Y" or "Z"),
        - "outcome_A" (measurement result of Alice, must be 1 or -1),
        - "outcome_B" (measurement result of Bob, must be 1 or -1),
        - "generation_duration" (generation duration for the state, unit specified is in baseline parameters),
        - "midpoint_outcome_i" for i = (0, 1, ..., number_of_nodes - 2)
          (result of heralded entanglement generation, must be :class:`~netsquid.qubits.ketstates.BellIndex` indicating
          Bell state that was projected on),
        - "swap_outcome_i" for i = (0, 1, ..., number_of_nodes - 3),
          (result of entanglement swap in repeater, must be :class:`~netsquid.qubits.ketstates.BellIndex` indicating
          Bell state that was projected on),
        - parameter_name for all parameters which are not constant for all results
          (otherwise, they should be in baseline_parameters).
        Any other column is optional.
        Note that if any of the required columns is missing, no error or warning is raised.
        This allows users to store e.g. only QKD measurement outcomes, or only state information, instead of both.
    generation_duration_unit : str, optional
        Unit for generation duration. This value is ignored if it is already specified in baseline parameters.
        By default "seconds", i.e. `generation_duration` would hold time it takes to generate the state.
    description : str
    **kwargs : Any
        Additional arguments that will be passed on to the constructor of the :obj:`pandas.DataFrame`

    Notes
    -----
    Other functions to handle data held by objects of this class are not included as methods in order to
    keep the class simple.

    For Pauli indices, the following convention is used:
    0 : 1 (identity)
    1 : X
    2 : Y
    3 : Z

    Parameter values must be hashable objects.

    Example
    -------

    >>> data = {"outcome_A": [0, 0], "basis_A": ["X", "Z"],
    >>>         "outcome_B": [1, 0], "basis_B": ["Y", "Z"],
    >>>         "generation_duration": [12, 42],
    >>>         "midpoint_outcome_0": [1, 2],
    >>>         "midpoint_outcome_1": [3, 0],
    >>>         "swap_outcome_0": [0, 1]}
    >>>
    >>> # We also add some other information that is not required by the
    >>> # RepchainDataFrameHolder
    >>> data["myowninfo"] = ["hello", "bye"]
    >>>
    >>>
    >>> h = RepchainDataFrameHolder(data=data)
    >>>
    >>> print(h.dataframe)
    >>> # output:
    >>> #outcome_A basis_A  outcome_B basis_B  generation_duration
    >>> # midpoint_outcome_0  midpoint_outcome_1  swap_outcome_0 myowninfo
    >>> #   0          0       X          1       Y                12
    >>> # 1                   3               0         hello
    >>> #   1          0       Z          0       Z                42
    >>> # 2                   0               1         bye
    >>>

    The attribute `baseline_parameters` should be handled as a regular dictionary. Note that `generation_duration_unit`
    will be automatically added in the init function if no baseline parameters are specified.

    >>> # Initializing baseline parameters
    >>> h.baseline_parameters.update({'prob_dark_count': 0.5})
    >>> h.baseline_parameters['prob_gate_error'] = 0.1
    >>> print(h.baseline_parameters)
    >>> # output
    >>> # {'prob_dark_count': 0.5, 'prob_gate_error': 0.1, 'generation_duration_unit': 'seconds'}
    >>>
    >>> # Removing baseline parameters
    >>>
    >>> del h.baseline_parameters['prob_gate_error']
    >>> print(h.baseline_parameters)
    >>> # output
    >>> # {'prob_dark_count': 0.5}
    >>>

    If some important package is used to generate the data,
    it can be added in the constructor.
    The version is automatically looked up from the
    installed packages.

    >>> # Initialize RepChainDataFrameHolder with some additional package
    >>> # By default, `netsquid` and `netsquid-simulationtools` are added
    >>> h = RepchainDataFrameHolder(data=data, additional_packages=["netsquid-netconf"])
    >>> print(h.packages)
    >>> # output
    >>> # {'netsquid': [X.X.X], 'netsquid-simulatontools': [X.X.X], 'netsquid-netconf': [X.X.X]}

    Develop notes: why use composition rather than inheritance
    ----------------------------------------------------------
    The main differences of this class with the :obj:`pandas.DataFrame` are:
      * the metadata `baseline_parameters`
      * the restriction that some column names must be in here

    Unfortunately, the metadata which can be added as attribute to a :obj:`pandas.DataFrame` is not
    preserved under appending, merging and concatenating such dataframes.
    For this reason, we need this class.
    It might seem more natural to subclass the :obj:`pandas.DataFrame`, with as main benefits that
    the methods of the original dataframe are inherited and the user need not learn a new API.
    This comes with new problems, however: many methods of the :obj:`pandas.DataFrame` *create* a new
    DataFrame rather than changing the old one (e.g. `append`, `merge`, etc.) which makes the subclassing
    not straightforward: each of these functions should be overridden so that the metadata is preserved.
    In case not all functions are overridden, it is not a priori clear to the user whether the original
    methods of :obj:`pandas.DataFrame` will work as expected also for a `RepchainDataFrame`. For this reason,
    we decided to go for composition: we now have a `RepchainDataFrameHolder` which is not much more than
    a basket for both the dataframe and the metadata.
    """
    BASIS_TYPES = ["X", "Y", "Z"]
    OUTCOME_TYPES = [0, 1]
    PAULIS = [BellIndex.PHI_PLUS, BellIndex.PSI_PLUS, BellIndex.PSI_MINUS, BellIndex.PHI_MINUS]

    COLUMN_NAMES_AND_CONSTRAINTS = \
        [("basis_A", lambda x: x in RepchainDataFrameHolder.BASIS_TYPES),
         ("basis_B", lambda x: x in RepchainDataFrameHolder.BASIS_TYPES),
         ("outcome_A", lambda x: x in RepchainDataFrameHolder.OUTCOME_TYPES),
         ("outcome_B", lambda x: x in RepchainDataFrameHolder.OUTCOME_TYPES),
         ("generation_duration", lambda x: ((isinstance(x, int) or isinstance(x, float)) and x >= 0)),
         ("state", lambda x: (isinstance(x, np.ndarray) or sparse.isspmatrix_csr(x)) and
                             (x.shape[0] == x.shape[1] or (len(x.shape) == 2 and x.shape[1] == 1)))]

    MIDPOINT_INDEX_COLUMN_NAMES_AND_CONSTRAINTS = \
        [("midpoint_outcome_", lambda x: x in RepchainDataFrameHolder.PAULIS)]

    NODE_INDEX_COLUMN_NAMES_AND_CONSTRAINTS = \
        [("swap_outcome_", lambda x: x in RepchainDataFrameHolder.PAULIS)]

    def __init__(self, data, number_of_nodes=None, baseline_parameters=None, additional_packages=None,
                 description=None, generation_duration_unit="seconds", **kwargs):

        # define the input to the dataframe

        if 'dtype' in kwargs:
            raise Exception("Not allowed to set dtype")
        if 'columns' in kwargs:
            raise Exception("Not allowed to set \'columns\'; use method `assign` instead")

        self.dataframe = pandas.DataFrame(data=data, **kwargs)

        if number_of_nodes is None:
            number_of_nodes = sum(["swap_outcome" in column for column in self.dataframe.columns]) + 2

        # set number of nodes
        if isinstance(number_of_nodes, float):
            if number_of_nodes.is_integer():
                number_of_nodes = int(number_of_nodes)
        if not isinstance(number_of_nodes, int):
            raise TypeError("number_of_nodes must be an integer")

        # use 'number_of_nodes' to determine columns
        column_names = \
            [name for (name, __)
             in self._column_names_and_constraints(number_of_nodes=number_of_nodes)]
        self._column_names = column_names

        self._number_of_nodes = number_of_nodes
        self._description = description

        self.baseline_parameters = _StringKeyTypeDict()
        if baseline_parameters is not None:
            self.baseline_parameters.update(baseline_parameters)
        if self.baseline_parameters.get("generation_duration_unit", None) is None:
            self.baseline_parameters.update({"generation_duration_unit": generation_duration_unit})

        self.packages = _StringKeyTypeDict()
        # by default add netsquid and netsquid-simulationtools to the dictionary of baseline packages
        packages = {'netsquid', 'netsquid-simulationtools'}
        if additional_packages is not None:
            # take the union of the two
            packages = packages | set(additional_packages)
        for package in packages:
            # wrap the version of the package in a list for easy comparison during `combine()`
            self.packages[package] = [pkg_resources.get_distribution(package).version]

        self.check_dataframe_correctness()

        # once the property `varied_parameters` is called once, we
        # store its result in a variable in order to not have to
        # compute it again
        self._varied_parameters = None  # following the convention that
        # each variable should be defined in __init__, even if it is given
        # a different value directly afterwards:
        self._reset_varied_parameters()

    def _reset_varied_parameters(self):
        self._varied_parameters = None

    @property
    def varied_parameters(self):
        """The names of the parameters that are not constant over all rows
        of the data.

        :rtype: list of str
        """
        if self._varied_parameters is None:
            # once the property `varied_parameters` is called once, we
            # store its result in a variable in order to not have to
            # compute it again
            self._varied_parameters = []

            columns_not_about_sim_results = \
                set(self.dataframe.columns) - set(self._column_names)

            for column_name in columns_not_about_sim_results:
                does_column_hold_single_element = \
                    len(list(set(self.dataframe[column_name].tolist()))) == 1
                if not does_column_hold_single_element:
                    self._varied_parameters.append(column_name)
        return self._varied_parameters

    @classmethod
    def _column_names_and_constraints(cls, number_of_nodes):
        return \
            cls.COLUMN_NAMES_AND_CONSTRAINTS + \
            [(name + str(midpoint_index), constraint)
                for midpoint_index in range(number_of_nodes - 1)
                for name, constraint in cls.MIDPOINT_INDEX_COLUMN_NAMES_AND_CONSTRAINTS] + \
            [(name + str(node_index), constraint)
                for node_index in range(number_of_nodes - 2)
                for name, constraint in cls.NODE_INDEX_COLUMN_NAMES_AND_CONSTRAINTS]

    @property
    def description(self):
        return self._description

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    def update_dataframe_by_appending(self, check_for_correctness=True, **kwargs):
        """
        Uses the `append` function of :obj:`pandas.DataFrame`
        and subsequently checks correctness using
        :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder.check_dataframe_correctness`.

        Parameters
        ----------
        **kwargs : Any
            Passed on to :obj:`pandas.DataFrame.append`

        Note
        ----
        By using `ignore_index=True` (see API of pandas.DataFrame), the indices of the rows
        in the new data are ignored and the counting starts at the last index of
        the original dataframe.
        """
        self.dataframe = self.dataframe.append(**kwargs)
        self._reset_varied_parameters()
        if check_for_correctness:
            self.check_dataframe_correctness()

    @property
    def number_of_results(self):
        return self.dataframe.shape[0]

    def check_dataframe_correctness(self):
        """
        Check simulation data for validity and consistency.

        Checks whether required columns are present and filled with expected kinds of data
        and whether there are not too many midpoint/swap results.
        """
        column_names_and_constraints = \
            self._column_names_and_constraints(number_of_nodes=self._number_of_nodes)

        for index, row in self.dataframe.iterrows():
            for column_name, column_constraint in column_names_and_constraints:
                try:
                    item = row[column_name]
                    if not column_constraint(item):
                        raise ValueError(f"Row {index} contains {item} in column {column_name}"
                                         ", which does not have the correct type")
                except KeyError:
                    # column was not found, in which case we decide to not
                    # raise a warning. It is up to the user to define
                    # which columns should be in the dataframe
                    pass

        column_names = set(self.dataframe.columns)

        number_of_midpoint_substrings = 0
        for column in column_names:
            if 'midpoint_outcome_' in column:
                number_of_midpoint_substrings += 1
        if number_of_midpoint_substrings != self.number_of_nodes - 1:
            raise ValueError("Incorrect number of midpoint outcomes")

        number_of_swap_substrings = 0
        for column in column_names:
            if 'swap_outcome_' in column:
                number_of_swap_substrings += 1
        if number_of_swap_substrings != self.number_of_nodes - 2:
            raise ValueError("Incorrect number of swap outcomes")

    def copy_baseline_parameter_to_column(self, name,
                                          remove_from_baseline=True):
        """
        Adds the column `name` to the dataframe and fills the entries
        with (copies of) the single value found in the baseline parameters.

        Parameters
        ----------
        name : str
            The parameter in `baseline_parameters`.
        remove_from_baseline : bool
            Whether the parameter will be removed from `baseline_parameters`
            after the copying.
        """
        val = self.baseline_parameters[name]
        self.dataframe[name] = [val] * self.number_of_results
        if remove_from_baseline:
            del self.baseline_parameters[name]
            self._reset_varied_parameters()

    def combine(self, other, assert_equal_baseline_parameters=True, assert_equal_packages=True):
        """
        Combines two :obj:`RepchainDataFrameHolder`
        objects.

        Parameters
        ----------
        other : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`
        assert_equal_baseline_parameters : bool
            Use `assert_equal_baseline_parameters` to check for compatibility.
            That is, if one combines two :obj:`RepchainDataFrameHolder`
            objects and one has `probability_gate_error` set to 0.1 and
            another to 0.2, then it will complain.
            Default value is True.
        assert_equal_packages : bool
            Used to verify whether all the specified packages are equal.
            If set to True, an Exception is raised if a package is missing or has a different version.
            If set to False, it will only print a warning message and update the package version.
            When a package is missing in one of the two :obj:`RepchainDataFrameHolder`s, the value `missing` will be
            added to the version information.
            When multiple different versions of the same package are used, they will be stored in a combined list.
            Default value is True.

        Example
        -------

        >>> first_data = \
        >>>        {"outcome_A": [0, 0], "basis_A": ["X", "Z"],
        >>>         "outcome_B": [1, 0], "basis_B": ["Y", "Z"],
        >>>         "generation_duration": [12, 42],
        >>>         "midpoint_outcome_0": [1, 2],
        >>>         "midpoint_outcome_1": [3, 0],
        >>>         "swap_outcome_0": [0, 1],
        >>>         "birds": ["falcon", "finch"]}
        >>>
        >>>
        >>> first_holder = RepchainDataFrameHolder(data=data)
        >>> first_holder.baseline_parameters = \
        >>>      {'prob_dark_count': 0.5, 'prob_gate_error': 0.1}
        >>>
        >>> second_data = \
        >>>        {"outcome_A": [1, 1], "basis_A": ["Z", "Y"],
        >>>         "outcome_B": [0, 1], "basis_B": ["X", "X"],
        >>>         "generation_duration": [33, 32],
        >>>         "midpoint_outcome_0": [2, 1],
        >>>         "midpoint_outcome_1": [0, 3],
        >>>         "swap_outcome_0": [1, 0],
        >>>         "mammals": ["cow", "horse"]}
        >>>
        >>> second_holder = RepchainDataFrameHolder(data=data2)
        >>> second_holder.baseline_parameters = \
        >>>      {'prob_dark_count': 0.5, 'prob_gate_error': 0.2, 'fibre_length': 10}
        >>>
        >>> first_holder.combine(second_holder, assert_equal_baseline_parameters=False)
        >>>
        >>> print(first_holder)
        >>> #Baseline parameters: {'prob_dark_count': 0.5, 'generation_duration_unit': 'seconds'}
        >>> #Packages: {'netsquid': [X.X.X], 'netsquid-simulationtools': [X.X.X]}
        >>> #  basis_A basis_B   birds  fibre_length mammals  midpoint_outcome_0
        >>> # midpoint_outcome_1  generation_duration  outcome_A  outcome_B  prob_gate_error  swap_outcome_0
        >>> #  0       X       Y  falcon           NaN     NaN
        >>> # 1                   3                12          0          1              0.1               0
        >>> #  1       Z       Z   finch           NaN     NaN
        >>> # 2                   0                42          0          0              0.1               1
        >>> #  2       Z       X     NaN          10.0     cow
        >>> # 2                   0                33          1          0              0.2               1
        >>> #  3       Y       X     NaN          10.0   horse
        >>> # 1                   3                32          1          1              0.2               0
        """

        if not isinstance(other, RepchainDataFrameHolder):
            raise TypeError("{} is not a RepchainDataFrameHolder")

        if self.number_of_nodes != other.number_of_nodes:
            raise Exception

        if set(other._column_names) != set(self._column_names):
            raise Exception

        if assert_equal_baseline_parameters:
            if set(self.baseline_parameters.items()) != set(other.baseline_parameters.items()):
                raise Exception("Baseline parameters not equal")

        # compare packages both ways and print a warning or raise an Exception
        package_differences = set(self.packages.keys()) ^ set(other.packages.keys())
        if package_differences != set():
            if assert_equal_packages:
                raise VersionMismatchError(f"Packages {package_differences} not found in both baseline packages. "
                                           f"Self has packages {self.packages.keys()}, "
                                           f"while other has {other.packages.keys()}.")
            # otherwise update the package version set by adding None to it
            for package in package_differences:
                print(f"Warning: Package {package} not found in both baseline packages. "
                      "Adding `missing` to list of versions.")
                # update the list of versions with `missing`
                self_package_version = self.packages.get(package, ["missing"])
                other_package_version = other.packages.get(package, ["missing"])
                self.packages[package] = sorted(set(self_package_version + other_package_version),
                                                reverse=True)
        # now update version info for packages that are specified in both
        for package in set(self.packages.keys()) & set(other.packages.keys()):
            version, other_version = self.packages[package], other.packages[package]
            if set(version) != set(other_version):
                if assert_equal_packages:
                    raise VersionMismatchError(f"Version of package {package} not equal. "
                                               f"Self uses {version}, while other uses {other_version}.")
                print(f"Warning: multiple versions found of package {package}. Combining both versions into one list.")
                # print a warning statement and update the version information
                self.packages[package] = sorted(set(version + other_version), reverse=True)

        # copy all baseline parameters that are not part of both
        # baselines to the data
        for holder_A, holder_B in [(self, other), (other, self)]:
            params_A = set(holder_A.baseline_parameters.items())
            params_B = set(holder_B.baseline_parameters.items())
            for (name, val) in params_A:
                if (name, val) not in params_B:
                    holder_A.copy_baseline_parameter_to_column(name=name)

        # replace the current dataframe by the combined dataframe
        self.update_dataframe_by_appending(other=other.dataframe, ignore_index=True, check_for_correctness=False)

        self._reset_varied_parameters()

    def __str__(self):
        return f"Baseline parameters: {self.baseline_parameters}\n" \
               f"Packages: {self.packages}\n{self.dataframe}"


class VersionMismatchError(Exception):
    """RepChainDataFrameHolder is combined with another one which has a different version for one of the packages."""
    pass

from statistics import mean, stdev
import numpy as np
from netsquid import BellIndex

from netsquid_simulationtools.repchain_data_functions import estimate_duration_per_success, _expected_target_state


def estimate_bb84_secret_key_rate_from_data(dataframe, sifting_factor=1):
    """Estimate the secret-key rate of BB84 in the asymptotic limit based on entanglement-distribution data.

    The data should contain results of a (simulated) entanglement-distribution experiment, where two parties (Alice
    and Bob) share a Bell state and perform a measurement in either the X, Y or Z basis on their qubit.
    The data should detail how long it took to share each Bell state, what the basis was that each party measured
    each specific Bell state in, and what the outcome of the measurement was.
    This data can be used to estimate the rate of entanglement distribution and the Quantum-Bit Error Rate (QBER),
    from which the secret-key rate is calculated using

    max(0., 1 - H(qber_x) - H(qber_z)) * sifting_factor / avg_number_attempts_per_success = SKR in [bits/attempt],

    where H(p) is the binary entropy function

    H(p) = -p log(p) - (1-p) log(1-p),

    and qber_x and qber_z are the QBER when both Alice and Bob measure in the X or Z basis respectively.

    Note: the data can also represent the more traditional scenario where BB84 is performed without entanglement,
    i.e. Alice sends states that Bob measured.
    In that case, Alice's basis choice represents the basis in which she prepares her state, and her measurement
    outcome represents which of the two corresponding basis states she sends.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        For specification of how the data should be structured, see the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`.
    sifting_factor : float (optional)
        Probability that both Alice and Bob use the same measurement basis.
        Defaults to 1, representing fully-asymmetric BB84 (see notes below for explanation).
        If both bases are chosen individually at random, `sifting_factor` should be set to 0.5.

    Returns
    -------
    secret_key_rate : float
        Secret Key Rate in [bits/attempt]. Note: The User can than convert this into [bits/second] or [bits/channel_use]
    skr_min : float
        Minimal value of Secret Key Rate within the interval given by the Stdev of the Qber
    skr_max : float
        Maximal value of Secret Key Rate within the interval given by the Stdev of the Qber
    skr_error : float
        Symmetric error calculated from the standard deviations of the QBERs and the number of attempts/success.

    Notes
    -----

    __Protocol__

    The protocol considered here is the one described by
    Key rates for quantum key distribution protocols with asymmetric noise,
    Murta et al, 2020, https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.062321 ,
    page 2 (entanglement-based version without advantage distillation).


    __Sifting__

    In the paper
    Efficient Quantum Key Distribution Scheme and a Proof of its Unconditional Security,
    Lo et al, 2005, https://doi.org/10.1007/s00145-004-0142-y .
    they show that a sifting factor of 1 can be obtained by biasing basis choices to either the Z or X basis.
    The preferred basis is then used to generate key, while the other one is only used for parameter estimation
    (i.e. checking that the QBER is not too large, as required to ensure security).
    The bias can be made sufficiently large such that, in the asymptotic limit, the number of results for which
    Alice and Bob did not both use the preferred basis becomes negligible, resulting in a sifting factor of 1.
    Lo et al. show that the security of this protocol is unconditional.

    Which basis is used as preferred basis (key-generation basis) does not affect the secret-key rate,
    see Proposition 1 of

    Because of this, a default sifting factor of 1 is used in this function.
    However, it is left as a parameter in case the user does want to include sifting.
    E.g. if symmetric BB84 is considered (i.e. choosing measurement bases uniformly at random), a sifting factor
    of 1/2 should be chosen.


    __Parameter Estimation__

    The effects of parameter estimation (which has to be performed for both bases) are negligible
    in the asymptotic limit.
    In Lo et al, it is mentioned that the number of sacrificed results must be at least of order log(k),
    where k is the length of the final key.
    In the limit where k goes to infinity, log(k) / k goes to zero, and thus the fraction of sacrificed results becomes
    negligible.

    In the paper
    A largely self-contained and complete security proof for quantum key distribution,
    Tomamichel and Leverrier, 2017, https://doi.org/10.22331/q-2017-07-14-14 ,
    the number of sacrificed results that is used is sqrt(m), where m is the total number of (sifted) results
    (k = sqrt(m), between equations (58) and (59); note that here k means the number of results sacrificed
    for parameter estimation, which is different from the meaning in Lo et al.).
    Since sqrt(m) / m also goes to zero in the asymptotic limit, it seems safe to assume the number of sacrificed
    results is negligible.

    Therefore, we do not include any effects of parameter estimation in this function.

    Note that in the originally proposed BB84 protocol, half the results are lost due to sifting.
    Furthermore, half of the remaining results are then used for parameter estimation.
    Therefore, the secret-key rate calculated in this function (when using the default sifting factor of 1)
    differs by a factor of 4 from the secret-key rate that would have been achieved
    using the original protocol in the asymptotic limit.


    __Secret-Key Rate__

    For a derivation of the secret-key rate formula used in this function, see e.g.
    The security of practical quantum key distribution,
    Scarani et al, 2009, https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1103%2FRevModPhys.81.1301&v=4740d283 ,
    appendix A.
    See also e.g. Murta et al, eq. (4), or Tomamichel and Leverrier, section 5, or
    Simple Proof of Security of the BB84 Quantum Key Distribution Protocol
    Shor and Preskill, 2000, https://link.aps.org/doi/10.1103/PhysRevLett.85.441
    (left column, above the lower equation).

    These sources don't always define the secret-key rate the same way. Specifically, some include sifting and
    parameter estimation, and some don't (and sometimes it's just not clear).
    However, in the case where both the effects of sifting and parameter estimation on the secret-key rate vanish,
    as we are considering, the differences don't really matter.


    __Sifting in Data__

    For estimating the QBER, all results where the measurement bases are unequal are disregarded.
    However, all results are taken into account when calculating the number of attempts per success.
    To estimate the number of attempts per success, results for which the measurement bases are unequal are also
    considered a "success".
    Any sifting is only applied in post-processing, as determined by the `sifting_factor` parameter in this function.
    Thus, the sifting factor is never extracted from the data.
    If basis choices are not always the same in the data, this does not have an effect on the calculated
    secret-key rate.
    (except that the uncertainty in the QBER will be larger compared to the case where the basis choices
    in the data would always be aligned, since then there would have been more results usable for estimating the QBER).
    For example, in case measurement bases are chosen randomly in the data such that the same basis is used only
    in 1/2 of the cases, but the default value `sifting_factor = 1` is used,
    this does not mean that the secret-key rate outputted by this function is halved as compared to that for the same
    data set but with all results with unequal basis choice removed.

    """
    if dataframe.empty:
        return 0, 0, 0, 0

    qber_x, qber_x_error = qber(dataframe, "X")
    qber_z, qber_z_error = qber(dataframe, "Z")

    duration_per_success, duration_per_success_error = estimate_duration_per_success(dataframe)

    secret_key_rate, skr_min, skr_max, skr_error = _estimate_bb84_secret_key_rate(qber_x, qber_x_error,
                                                                                  qber_z, qber_z_error,
                                                                                  duration_per_success,
                                                                                  duration_per_success_error)

    return (secret_key_rate * sifting_factor, skr_min * sifting_factor,
            skr_max * sifting_factor, skr_error * sifting_factor)


def secret_key_rate_from_states():
    """ Function that should compute the secret key rate from the states saved in the input DataFrame.

    Returns
    -------
    secret_key_rate : float
        Secret key rate calculated from the DataFrame.
    secret_key_error: float
        Standard deviation of the secret key rate

    """
    # calculate QBER in X and Z from density matrix
    # return _estimate_bb84_secret_key_rate(qber_x, 0, qber_z, 0)
    raise NotImplementedError("Not yet implemented.")


def agreement_with_expected_outcome(dataframe, basis_a, basis_b=None):
    """ Function that checks whether the measured outcomes agree with the expected outcomes in the input DataFrame,
    for a given pair of basis choices.
    Or in other words: The function checks whether the measurement outcomes of Alice and Bob, for which we expect
    (anti-)correlation from the simulation data, are actually (anti-)correlated and assigns 0 for agreement and 1 for
    disagreement.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing simulation data
    basis_a: string
        Basis in which Alice measured. Should be in ["X", "Y", "Z"]
    basis_b: string , optional
        Basis in which Bob measured, if not specified will be set to the same basis as Alice by default.

    Returns
    -------
    agreement_for_basis : list
        list of {0,1} for a given basis pair where 0, 1 correspond to 'no agreement' and 'agreement' respectively

    """
    if basis_b is None:
        basis_b = basis_a

    agreement_for_basis = []
    for index, row in dataframe.iterrows():
        if row["basis_A"] == basis_a and row["basis_B"] == basis_b:
            if _do_we_expect_correlation(row):
                if row["outcome_A"] == row["outcome_B"]:
                    agreement_for_basis.append(1)
                else:
                    agreement_for_basis.append(0)
            else:
                if row["outcome_A"] != row["outcome_B"]:
                    agreement_for_basis.append(1)
                else:
                    agreement_for_basis.append(0)

    return agreement_for_basis


def qber(dataframe, basis_a, basis_b=None, quantile=1):
    """
    Function that calculates the QBER for a specified basis (set) from the input DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing simulation data
    basis_a : string
        Basis in which Alice measured. Should be in ["X", "Y", "Z"]
    basis_b : string , optional
        Basis in which Bob measured, if not specified will be set to the same basis as Alice by default.
    quantile : float , optional
        1 - alpha/2 quantile of a standard normal distribution corresponding to the target error rate alpha.
        For a 95% confidence level, the error alpha = 1 − 0.95 = 0.05 , so (1 − alpha/2) = 0.975 and quantile = 1.96.
        Default is quantile = 1


    Returns
    -------
    qber : float
        QBER (quantum bit error rate)

    qber_error : float
        QBER error (standard deviation of the mean)

    """
    if basis_b is None:
        basis_b = basis_a

    agreement_list = agreement_with_expected_outcome(dataframe, basis_a, basis_b)
    if agreement_list:
        qber = 1 - mean(agreement_list)
        qber_error = quantile * stdev(agreement_list) / np.sqrt(len(agreement_list))
    else:
        # list is empty
        qber, qber_error = 0, 0

    return qber, qber_error


def _binary_entropy(p):
    """Calculate binary entropy.

    H(p) = -p log(p) - (1-p) log(1-p)

    Parameters
    ----------
    p : float
        Probability value to calculate binary entropy for.


    Returns
    -------
    float
        Binary entropy.

    """
    a = - p * np.log2(p) if p > 0 else 0
    b = - (1 - p) * np.log2(1 - p) if p < 1 else 0
    return a + b


def _derivative_binary_entropy(x):
    """Derivative of binary-entropy function.

    Parameters
    ----------
    x : float in the open interval (0, 1)
        Value at which to evaluate the function.

    Returns
    -------
    float

    """
    return - np.log2(x / (1 - x))


def _error_binary_entropy(x, x_error):
    """Standard error in the binary-entropy function.

    Parameters
    ----------
    x : float in the closed interval [0, 1]
        Estimated value of the argument.
    x_error : float
        Standard error of the argument.

    Returns
    -------
    float

    Note
    ----
    Only accurate when x_error << 1.

    """
    if np.isclose(x, 0) or np.isclose(x, 1) or x_error == 0:
        return 0
    return _derivative_binary_entropy(x) * x_error


def _error_secret_key_rate(qber_x, qber_x_error, qber_z, qber_z_error,
                           attempts_per_success, attempts_per_success_error):
    """Calculating the standard error in the secret-key rate.

    Parameters
    ----------
    qber_x : float (between 0 and 1)
        Quantum Bit Error Rate in X basis (both Alice and Bob measure in X)
    qber_x_error : float
        Standard deviation of the Quantum Bit Error Rate in X basis
    qber_z : float (between 0 and 1)
        Quantum Bit Error Rate in Z basis (both Alice and Bob measure in Z)
    qber_z_error : float
        Standard deviation of the Quantum Bit Error Rate in Z basis
    attempts_per_success: float
        Average number of attempts required per successfully-distributed raw key bit.
    attempts_per_success_error: float
        Standard deviation of the number of attempts required per successfully-distributed raw key bit.

    Returns
    -------
    float

    """
    secret_key_rate = 1 - _binary_entropy(qber_x) - _binary_entropy(qber_z)
    if secret_key_rate <= 0:
        return 0
    return (np.sqrt(
        np.power(_error_binary_entropy(qber_x, qber_x_error), 2) +
        np.power(_error_binary_entropy(qber_z, qber_z_error), 2) +
        np.power(secret_key_rate * attempts_per_success_error, 2)
    ) / attempts_per_success)


def _estimate_bb84_secret_key_rate(qber_x, qber_x_error, qber_z, qber_z_error,
                                   attempts_per_success, attempts_per_success_error):
    """ Function that computes the secret key rate and its standard deviation from the simulation data in the input
    Dataframe.

    This is done by calculating the Quantum Bit Error Rate (QBER) in X and Z basis and then taking:
    max(0., 1 - H(qber_x) - H(qber_z)) / attempts_per_success

    where H(p) is the binary entropy function:
    H(p) = -p log(p) - (1-p) log(1-p)

    For more information about derivation of the formula, the protocol under consideration and the assumptions used,
    see the docstring of
    :func:`~netsquid_simulationtools.repchain_data_functions.estimate_bb84_secret_key_rate_from_data`.

    Parameters
    ----------
    qber_x : float (between 0 and 1)
        Quantum Bit Error Rate in X basis (both Alice and Bob measure in X)
    qber_x_error : float
        Standard deviation of the Quantum Bit Error Rate in X basis
    qber_z : float (between 0 and 1)
        Quantum Bit Error Rate in Z basis (both Alice and Bob measure in Z)
    qber_z_error : float
        Standard deviation of the Quantum Bit Error Rate in Z basis
    attempts_per_success: float
        Average number of attempts required per successfully-distributed raw key bit.
    attempts_per_success_error: float
        Standard deviation of the number of attempts required per successfully-distributed raw key bit.


    Returns
    -------
    secret_key_rate : float
        Secret Key Rate in [bits/attempt] as max(0., 1 - H(qber_x) - H(qber_z)) / attempts_per_success.
    skr_min : float
        Minimal value of Secret Key Rate within the interval given by the Stdev of the Qber.
    skr_max : float
        Maximal value of Secret Key Rate within the interval given by the Stdev of the Qber.
    skr_error : float
        Standard error in the secret-key rate (calculated using standard formula for propagation of uncertainty).

    """
    # calculate secret key rate using binary entropy function
    secret_key_rate = max(0., 1 - _binary_entropy(qber_x) - _binary_entropy(qber_z)) / attempts_per_success

    if (qber_x, qber_z) <= (0, 1):
        print("Beware: one of the QBER's is 0 or 1. "
              "This may indicate not enough statistics were obtained to estimate the error.")

    # pick min / max in interval [0,1]
    skr_min = min(1 - _binary_entropy(qber_x + qber_x_error) - _binary_entropy(qber_z + qber_z_error),
                  1 - _binary_entropy(qber_x + qber_x_error) - _binary_entropy(qber_z - qber_z_error),
                  1 - _binary_entropy(qber_x - qber_x_error) - _binary_entropy(qber_z + qber_z_error),
                  1 - _binary_entropy(qber_x - qber_x_error) - _binary_entropy(qber_z - qber_z_error),
                  1 - _binary_entropy(qber_x) - _binary_entropy(qber_z),
                  1.)
    skr_min = max(0., skr_min) / attempts_per_success
    skr_max = max(1 - _binary_entropy(qber_x + qber_x_error) - _binary_entropy(qber_z + qber_z_error),
                  1 - _binary_entropy(qber_x + qber_x_error) - _binary_entropy(qber_z - qber_z_error),
                  1 - _binary_entropy(qber_x - qber_x_error) - _binary_entropy(qber_z + qber_z_error),
                  1 - _binary_entropy(qber_x - qber_x_error) - _binary_entropy(qber_z - qber_z_error),
                  1 - _binary_entropy(qber_x) - _binary_entropy(qber_z),
                  0.)
    skr_max = min(1., skr_max) / attempts_per_success

    skr_error = _error_secret_key_rate(qber_x, qber_x_error, qber_z, qber_z_error,
                                       attempts_per_success, attempts_per_success_error)

    return secret_key_rate, skr_min, skr_max, skr_error


def _do_we_expect_correlation(row):
    """Function that computes the expected correlation of measurement results between Alice and Bob for a row of the
    input DataFrame.

    This is done by calculating the expected target state and then looking up its expected correlation for the basis
    choice specified in this row of the DataFrame.

    Parameters
    ----------
    row : pandas.Series
        Row of the DataFrame containing simulation data

    Returns
    -------
    bool
        Boolean, whether we expect measurement results between Alice and Bob to be correlated.

    Note: For indexing of Bell stated see `_expected_target_state()`

    """
    basis_a = row["basis_A"]
    basis_b = row["basis_B"]
    if basis_a != basis_b:
        # TODO: maybe this should not throw an error but just ignore the line?
        raise ValueError("Cannot get correlation for Bell states if A and B measured in different basis.")

    # when using atomic ensembles with presence-absence encoding, two chains have to be used. The expected correlation
    # depends on the target state of both chains
    if 'chain_2_midpoint_0' in row.keys():
        if basis_a == "X":
            expected_index_chain_1, expected_index_chain_2 = _expected_target_state(row)
            if expected_index_chain_1 == expected_index_chain_2:
                return True
            else:
                return False
        elif basis_a == "Z":
            return False
        elif basis_a == "Y":
            raise NotImplementedError("Y-basis correlations not implemented for two chain AE setup yet.")

    expected_target_index = _expected_target_state(row)
    basis_to_is_correlated = {"X": {BellIndex.PHI_PLUS: True,
                                    BellIndex.PSI_PLUS: True,
                                    BellIndex.PSI_MINUS: False,
                                    BellIndex.PHI_MINUS: False},
                              "Y": {BellIndex.PHI_PLUS: False,
                                    BellIndex.PSI_PLUS: True,
                                    BellIndex.PSI_MINUS: False,
                                    BellIndex.PHI_MINUS: True},
                              "Z": {BellIndex.PHI_PLUS: True,
                                    BellIndex.PSI_PLUS: False,
                                    BellIndex.PSI_MINUS: False,
                                    BellIndex.PHI_MINUS: True},
                              }
    return basis_to_is_correlated[basis_a][expected_target_index]

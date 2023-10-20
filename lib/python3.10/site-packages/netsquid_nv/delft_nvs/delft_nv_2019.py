"""
Parameters for the NV center devices as found in
papers from QuTech (Delft) up to 2019.
"""
# This file contains a large number of parameters concerning the emission of a
# photon by an NV-center, its intereference in a beam splitter and its detection
# in the midpoint.

import numpy as np
from netsquid_nv.nv_parameter_set import NVParameterSet


class NVParameterSet2019(NVParameterSet):

    ##############################
    # Photon emission parameters #
    ##############################

    """
    The probability that a photon is lost in the fiber,
    expressed in dB/km."""
    p_loss_lengths_with_conversion = 0.5
    p_loss_lengths_no_conversion = 5.

    c = 206753.41931034482  # speed of light in fibre [km/s]

    tau_emissions = lambda cavity: 6.48 if cavity else 12

    p_zero_phonon = lambda cavity: 0.46 if cavity else 0.03

    # Duration of the time window in the midpoint in nanoseconds
    # The time window duration is set to 40 nanoseconds in
    # "Dephasing mechanims ..." (https://arxiv.org/abs/1802.05996)
    #...and to 25 nanoseconds in "Deterministic delivery of remote entanglement"
    # (https://doi.org/10.1038/s41586-018-0200-5)
    time_window = 25.

    # The probability that a photon that enters the detector is also measured.
    # Source: Hensen et al. (2015) "Loophole-free Bell inequality violation using
    # electron spins separated by 1.3 kilometers", supplementary information, page 2.
    real_detection_eff = 0.80

    # <-- begin of a source of information -->
    # The following values come from the experiment described in
    # the article "Deterministic delivery of remote entanglement"
    # (https://doi.org/10.1038/s41586-018-0200-5).
    # These values were given to us by Arian Stolk.

    # the photon indistinguishability.
    visibility = 0.9

    # the "total" detector efficiency, i.e. the detection efficiency as
    # measured in an experiment. By this we mean: in an experiment where an NV is connected with
    # a negligibly short glass fibre to a detector.
    # The total detection efficiency is then the probability
    # that an excitation of the NV really results in a click of the detector.
    total_detection_eff_detector_1 = 0.00028
    total_detection_eff_detector_2 = 0.00042
    total_detection_eff = (total_detection_eff_detector_1 + total_detection_eff_detector_2) / 2.

    # Dark count rate in Hz
    dark_count_rate = 20.

    # The probability of having a number of `k` dark counts in
    # a detector follows the probability distribution called 'exponential distribution'
    prob_dark_count = 1. - np.exp(-1. * time_window * 10 ** (-9) * dark_count_rate)

    # The final state that is produced in the entire
    # heralded-entanglement-generation setup has a relative phase
    # that is caused by the drift in the interferometer.
    # This phase, named "residual phase noise in the interferometer" in the article,
    # is a random variable and thus has an average and a standard deviation
    avg_electron_electron_phase_drift = 0.
    std_electron_electron_phase_drift = 15. * np.pi / 180.  # [radians]

    # The photon emission noise model asks for the standard deviation of the
    # phase that a single photon-electron pair gets; assuming that the two
    # pairs that arrive at the beam splitter have the same variance, we get:
    #     Var(total phase) = 2 * Var(phase of a single pair)
    # since the variance is additive.
    # Since the variance is the square of the standard deviation, we obtain
    std_electron_photon_phase_drift = std_electron_electron_phase_drift / np.sqrt(2)
    # was 14.3 / sqrt(2) in a previous calculation

    # The time of resetting the electron and emitting the photon in nanoseconds
    photon_emission_delay = 5.5 * 10**3  # in a previous file, this was set at 6 * 10^3

    # <-- end of a source of information -->

    # The total detection efficiency is the product of four probabilities:
    # - the probability that a photon is emitted from the electron is
    #           in the zero-phonon line
    # - the collection efficiency (probability that an emitted photon is
    #           collected into the fiber)
    # - the probability that the photon is not lost in the fiber
    # - the "real" detection efficiency, i.e. the probability that the detectors
    #           click, given that there was a photon
    #
    # We can thus compute the collection efficiency in case the of no cavities using
    # the other three parameters together with the total detection efficiency
    zero_phonon_prob = p_zero_phonon(cavity=False)

    length = 0.001  # the presumed fibre length of the experiment (i.e. from
    # node to beamsplitter) in which the total detection efficiency was measured
    p_photon_not_lost = 10 ** (-1. * p_loss_lengths_no_conversion * length / 10.)

    collection_eff_in_case_no_cavity_no_conversion = \
        (total_detection_eff / (p_photon_not_lost * zero_phonon_prob * real_detection_eff))
    # Note: in a previous similar calculation by Axel, the collection efficiency was set at 0.02

    """
    The collection efficiency: the probability that an emitted
    photon is collected into the fiber.

    In case of frequency conversion, the collection efficiency
    is reduced with 70%.
    """
    collection_eff_with_conversion = collection_eff_in_case_no_cavity_no_conversion * 0.3

    prob_detect_excl_transmission_with_conversion_with_cavities = \
        p_zero_phonon(cavity=True) * collection_eff_with_conversion * real_detection_eff

    prob_detect_excl_transmission_no_conversion_no_cavities = \
        p_zero_phonon(cavity=False) * \
        collection_eff_in_case_no_cavity_no_conversion * \
        real_detection_eff

    # See section IV in https://arxiv.org/src/1712.07567v2/anc/SupplementaryInformation.pdf
    p_double_exc = 0.04

    # These values are taken from
    # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.062330, see FIG 2 and eq (2)
    # for definitions

    # see table I
    delta_w_list = [377, 62, 77]  # in kHz / (2pi)
    tau_decay_list = [263, 837, 640]  # in ns

    # now we consider carbon C_2 from the paper
    delta_w = 77.  # in kHz / (2pi), see table I
    tau_decay = 163.  # in ns, see figure (2)

    delta_w_optimistic = delta_w
    tau_decay_optimistic = tau_decay

    # the probability of nuclear dephasing
    # is a function of the product of tau_decay
    # and the coupling strength delta_w, with
    # some constant factors. See also
    # netsquid_nv.nv_state_delivery_model.prob_nuclear_dephasing_during_entgen_attempt
    product_tau_decay_delta_w = tau_decay * 10 ** (-9) * delta_w * 2 * np.pi * 10 ** 3

    ####################
    # Qubit parameters #
    ####################

    # The decoherence properties of the qubits *without* dynamical decoupling
    electron_T1_native = 2.68e6
    electron_T2_native = 3.3e3
    carbon_T1_native = 0.
    carbon_T2_native = 3.5e6

    # For the dynamical-decoupling decoherence times:
    # (source: https://doi.org/10.1038/s41467-018-04916-z)
    electron_T1_dynamical_decoupling = 3600 * 1e9
    electron_T2_dynamical_decoupling = 1.46e9

    # Source: C. Bradley et al. in preparation (see p. 33 link layer paper)
    carbon_T2_dynamical_decoupling = 1.e9

    # Source: private communication with Matteo Pompili
    carbon_T1_dynamical_decoupling = 10. * 3600 * 1e9

    # Final choice for the decoherence times
    electron_T1 = electron_T1_dynamical_decoupling
    electron_T2 = electron_T2_dynamical_decoupling
    carbon_T1 = carbon_T1_dynamical_decoupling
    carbon_T2 = carbon_T2_dynamical_decoupling

    ###################
    # Gate parameters #
    ###################

    # TODO we should still maybe add gate noise parameters and execution times here

    # In "Deterministic delivery of remote entanglement", https://arxiv.org/abs/1712.07567,
    # the fidelities are stated of measurement of |0> and |1>; these are the probabilities that
    # a measurement outcome is flipped. For more information, see
    # also fig. 2c in https://arxiv.org/abs/1508.05949
    prob_error_0 = 1. - 0.950
    prob_error_1 = 1. - 0.995

    prob_error_0_optimistic = 1 - 0.99  # See arXiv:1603:01602

    # We model gate noise and initialization noise using
    # the depolarizing map.

    # The depolarization channel with parameter p is implemented
    # in NetSquid (netsquid.qubits.qubitapi) as
    # \rho |-> (1-3p/4) \rho + p/4 * (X\rho X + Y\rho Y + Z\rho Z).

    # Example (1): if \rho is the Bell state \Phi^+ and the depolarization
    # channel is applied to one side of the Bell pair, then
    # the fidelity between the resulting state and \Phi^+ is 1-3p/4.

    # Example (2): if \rho = |0><0|, e.g. when the state has just
    # been initialized, then the fidelity between the resulting
    # state and |0> is (1-3p)/4 + p/4 = 1 - p/2

    # Initialization of the electron into the |0>-state has been
    # investigated in Reiserer et al. (2016)
    # "Robust Quantum-Network Memory Using Decoherence-Protected Subspaces of Nuclear Spins"
    # We set it to 0.99.
    # By example (2) above, we want that 1-p/2=0.99, yielding p=2*0.01.
    electron_init_depolar_prob = 2 * 0.01

    # Via a similar reasoning, we get for initializatin the carbon in |0>
    # that (Bradley et al, PRX 2019, showed carbon initialization fidelity of 0.997)
    carbon_init_depolar_prob = 2 * (1 - 0.997)

    electron_single_qubit_depolar_prob = 0.

    # By example (1) above, we see that if we want that the fidelity
    # of a state upon which an ideal Z-rotation gate has acted or
    # a noisy Z-rotation gate, is given by F=1-3p/4, where p is the parameter
    # of the depolarization map. Hence if we want F=0.999, then p=4/3 * 0.001.
    # (gate fidelity showed in Taminiau et al, Nature Nanotechnology 2014, is 1)
    carbon_z_rot_depolar_prob = 4. / 3. * 0.001

    # This value is taken such that we get the correct value (0.96) when running the
    # gate sequence in Fig. 2a in https://arxiv.org/pdf/1703.03244
    ec_gate_depolar_prob = 0.02

    coherent_phase = 0.
    p_fail_class_corr = 0.
    initial_nuclear_phase = 0.

    @classmethod
    def p_loss_lengths(cls, conversion=True):
        return cls.p_loss_lengths_with_conversion if conversion else cls.p_loss_lengths_no_conversion

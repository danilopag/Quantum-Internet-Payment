#
# Heralded entanglement generation using absorption midpoint station
#

import abc
import numpy as np

from netsquid import EventHandler, EventType
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_H,
    INSTR_MEASURE,
    INSTR_INIT,
    INSTR_Z,
)
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits.qubit import Qubit
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import operators as ops
from netsquid.nodes.connections import Connection
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.cqchannel import CombinedChannel
from netsquid.components.models import FibreDelayModel, FibreLossModel
from netsquid.util.simlog import logger
from netsquid.util.simtools import get_random_state

random = get_random_state()

__all__ = [
    'BaseAbsorptionConnection',
    'NoResetAbsorptionConnection'
]


class BaseAbsorptionConnection(Connection, metaclass=abc.ABCMeta):
    """
    Abstract class for absorption connection.
    This connection contains a midpoint that can absorp photons from left and right, for example to
    a silicon-vacancy.
    These absorbed qubits can then be measured in a Bell measurement to create remote entanglement between the left and right node.
    In this case this is a complete Bell measurement, compared to a beamsplitter and detectors.

    The setup looks as follows:

    .. code-block :: text

            -----------------------------
            | > CC  <> ------- <>  CC < |
        A > |          |  M  |          | < B
            | > CQC <> ------- <> CQC < |
            -----------------------------

    where:
    * `A`: Left node.
    * `B`: Right node.
    * `CC`: classical channel
    * `CQC`: quantum-classical channel
    * `M`: Midpoint absorbing photons and performing complete Bell measurement.

    Subclass and override the methods `handle_absorptionA` and `handle_absorptionB` to determine
    what the midpoint should do when detectors for absorption clicks.
    """

    def __init__(
        self,
        name,
        length_A=0,
        length_B=0,
        p_loss_init_A=0,
        p_loss_init_B=0,
        p_loss_length_A=0,
        p_loss_length_B=0,
        speed_of_light_A=200000,
        speed_of_light_B=200000,
        qm_size=2,
        com_physical_IDA=0,
        com_physical_IDB=1,
        spin_init_delay=0,
        spin_corr_delay=0,
        spin_bell_meas_delay=0,
        time_window=1,
        t0=0,
        t_cycle=10,
        dark_count_rateA=0,
        dark_count_rateB=0,
        p_detectionA=1,
        p_detectionB=1,
    ):
        super().__init__(name)

        # Physical ID for spins used
        self.com_physical_IDA = com_physical_IDA
        self.com_physical_IDB = com_physical_IDB

        # Setup a quantum node for us (i.e. midpoint)
        if self.com_physical_IDA == self.com_physical_IDB:
            if qm_size < 1:
                raise ValueError("qm_size is to small")
        else:
            if qm_size < 2:
                raise ValueError("qm_size is to small")

        midpoint_qpd = QuantumProcessor("midpoint_qpd", num_positions=qm_size, phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=spin_init_delay, parallel=True),
            PhysicalInstruction(INSTR_Z, duration=spin_corr_delay, parallel=True),
            PhysicalInstruction(INSTR_CNOT, duration=spin_bell_meas_delay, parallel=True),
            PhysicalInstruction(INSTR_H, duration=0, parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=0, parallel=True),
        ])
        midpoint_qpd.add_ports(["in_left", "in_right", "out_left", "out_right"])
        self.QPD = midpoint_qpd

        # create models
        delay_model_A = FibreDelayModel(c=speed_of_light_A)
        delay_model_B = FibreDelayModel(c=speed_of_light_B)
        loss_model_A = FibreLossModel(p_loss_init=p_loss_init_A, p_loss_length=p_loss_length_A)
        loss_model_B = FibreLossModel(p_loss_init=p_loss_init_B, p_loss_length=p_loss_length_B)

        # create classical channels
        cchannel_A = ClassicalChannel(
            name="CChannel_A_of_{}".format(self.name),
            length=length_A,
            models={"delay_model": delay_model_A}
        )
        cchannel_B = ClassicalChannel(
            name="CChannel_B_of_{}".format(self.name),
            length=length_B,
            models={"delay_model": delay_model_B},
        )

        # create quantum channels
        cqchannel_A = CombinedChannel(
            name="QChannel_A_of_{}".format(self.name),
            length=length_A,
            models={
                "delay_model": delay_model_A,
                "quantum_loss_model": loss_model_A,
            },
            transmit_empty_items=True,
        )
        cqchannel_B = CombinedChannel(
            name="QChannel_B_of{}".format(self.name),
            length=length_B,
            models={
                "delay_model": delay_model_B,
                "quantum_loss_model": loss_model_B,
            },
            transmit_empty_items=True,
        )

        # add as subcomponents
        self.add_subcomponent(cchannel_A, "CCh_A")
        self.add_subcomponent(cchannel_B, "CCh_B")
        self.add_subcomponent(cqchannel_A, "CQCh_A")
        self.add_subcomponent(cqchannel_B, "CQCh_B")
        self.add_subcomponent(midpoint_qpd, "MidpointQPD")

        # connect ports
        cqchannel_A.ports["recv"].connect(midpoint_qpd.ports["in_left"])
        cqchannel_B.ports["recv"].connect(midpoint_qpd.ports["in_right"])
        midpoint_qpd.ports["out_left"].connect(cchannel_A.ports["send"])
        midpoint_qpd.ports["out_right"].connect(cchannel_B.ports["send"])
        self.ports["A"].forward_input(cqchannel_A.ports["send"])
        self.ports["B"].forward_input(cqchannel_B.ports["send"])
        cchannel_A.ports["recv"].forward_output(self.ports["A"])
        cchannel_B.ports["recv"].forward_output(self.ports["B"])

        # Handle arrivals
        midpoint_qpd.ports["in_left"].bind_input_handler(self._handle_photon_arrival_left)
        midpoint_qpd.ports["in_right"].bind_input_handler(self._handle_photon_arrival_right)

        # Set timings
        self.time_window = time_window
        self.t0 = t0
        self.t_cycle = t_cycle

        # Dark counts and detection probabilities
        self.p_detectionA = p_detectionA
        self.p_detectionB = p_detectionB
        self.p_darkA = 1 - np.exp(-(10 ** (-9) * dark_count_rateA * self.time_window))
        self.p_darkB = 1 - np.exp(-(10 ** (-9) * dark_count_rateB * self.time_window))

        # Photons arrived in time window
        self._photonA = None
        self._photonB = None

        # Keep track of which detector clicked
        # 0, 0 = "No click"
        # 0, 1 = "Clicked in horizontal mode"
        # 1, 0 = "Clicked in vertical mode"
        # 1, 1 = "Clicked in both modes"
        self.clickedA = [0, 0]
        self.clickedB = [0, 0]

        # Keep track of whether we are in the time window or not
        self._in_window = False

        # Keep track of whether connection is started or not
        self._is_running = False

        # Setup events for start and end of detection window
        self._EV_START_WINDOW = EventType("START WINDOW", "Start the detection window")
        self._EV_STOP_WINDOW = EventType("STOP WINDOW", "Stop the detection window")

        # Start the connection
        self.start()

    def comm_delay(self, from_left):
        """
        Returns the communication delay from the node to the midpoint and back
        :param node_id: int
        :return: float
        """
        if from_left:
            node_to_midpoint = self.channel_from_A
            midpoint_to_node = self.channel_to_A
        else:
            node_to_midpoint = self.channel_from_B
            midpoint_to_node = self.channel_to_B

        return node_to_midpoint.delay_mean + midpoint_to_node.delay_mean

    def _handle_photon_arrival_left(self, message):
        self._handle_photon_arrival(data=message.items[0], from_left=True)

    def _handle_photon_arrival_right(self, message):
        self._handle_photon_arrival(data=message.items[0], from_left=False)

    def _handle_photon_arrival(self, data, from_left):
        """
        Internal unphysical handler for photon arrival within time window.
        Photons are measured/detected by the end of the time window.
        :param node_id: int
        :return: None
        """
        if self._in_window:
            # Add photon to arrived photons
            photon = None
            c_msg, photon_list = data
            [photon] = photon_list
            assert isinstance(photon, Qubit), "Photon is not a qubit"

            # Check if photon was lost
            if photon.qstate is None:
                logger.warning("Photon was lost")
                photon = None

            if from_left:
                self._photonA = photon
            else:
                self._photonB = photon

    def _absorb_photon(self, from_left):
        """
        Performs the absorption operations and measures a photon that arrived in time window.
        :param node_id:  int
        :return:
        """
        if self.QPD.busy:
            logger.debug("Photon could not be absorbed since QPD is busy")
            return

        if from_left:
            com_physical_ID = self.com_physical_IDA
            photon = self._photonA
            clicked = self.clickedA
            p_dark = self.p_darkA
            p_detection = self.p_detectionA
        else:
            com_physical_ID = self.com_physical_IDB
            photon = self._photonB
            clicked = self.clickedB
            p_dark = self.p_darkB
            p_detection = self.p_detectionB

        if photon is not None:
            # Perform the effective absorption operations
            spin = self.QPD.peek(com_physical_ID)[0]
            qapi.operate([photon, spin], ops.CNOT)
            qapi.operate(photon, ops.H)

            m = qapi.measure(photon)[0]

            if m == 0:
                clicked[0] = 1
            else:
                clicked[1] = 1

        # Add dark counts and detector efficiencies
        for detector in range(2):
            if clicked[detector] == 0:
                r = random.random()
                if r < p_dark:
                    clicked[detector] = 1
            else:
                r = random.random()
                if r < 1 - p_detection:
                    clicked[detector] = 0

    def _handle_clicks(self, from_left):
        """
        Calls the appropriate handler depending on what detector clicked
        :return: None
        """
        if from_left:
            clicked = self.clickedA
            no_click_handler = self.handle_no_absorptionA
            click_handler = self.handle_absorptionA
            failed_click_handler = self.handle_failed_absorptionA
        else:
            clicked = self.clickedB
            no_click_handler = self.handle_no_absorptionB
            click_handler = self.handle_absorptionB
            failed_click_handler = self.handle_failed_absorptionB

        if (clicked[0] == 0) and (clicked[1] == 0):
            no_click_handler()
        elif (clicked[0] == 1) and (clicked[1] == 1):
            failed_click_handler()
        else:
            assert clicked[0] != clicked[1]
            click_handler()

    @abc.abstractmethod
    def handle_absorptionA(self):
        """
        Should be overriden to determine what the midpoint should do when detectors for absorption clicks.
        :return: None
        """
        pass

    def handle_no_absorptionA(self):
        """
        Should be overriden to determine what the midpoint should do when there was no click by the end of a
        detection window.
        :return: None
        """
        pass

    def handle_failed_absorptionA(self):
        """
        Should be overriden to determine what the midpoint should do when there was a failed event,
        for example when the detectors in both modes clicked due to dark counts.
        :return: None
        """
        pass

    @abc.abstractmethod
    def handle_absorptionB(self):
        """
        Should be overriden to determine what the midpoint should do when detectors for absorption clicks.
        :return: None
        """
        pass

    def handle_no_absorptionB(self):
        """
        Should be overriden to determine what the midpoint should do when there was no click by the end of a
        detection window.
        :return: None
        """
        pass

    def handle_failed_absorptionB(self):
        """
        Should be overriden to determine what the midpoint should do when there was a failed event,
        for example when the detectors in both modes clicked due to dark counts.
        :return: None
        """
        pass

    def _reset_clicks(self):
        """
        Resets the information about the clicks by for example the end of the time window
        :return: None
        """
        self._photonA = None
        self._photonB = None
        self.clickedA = [0, 0]
        self.clickedB = [0, 0]

    def send_classical_to_both(self, c_msg):
        """
        Sends a classical message both of the nodes.
        :param c_msg: The classical message
        :return: None
        """
        self.QPD.ports["out_left"].tx_output(c_msg)
        self.QPD.ports["out_right"].tx_output(c_msg)

    def start(self):
        """
        Start connection. This will initialize the timer marshalling the detection time window.
        :return: None
        """

        if self._is_running:
            return None

        logger.debug("Starting absorption connection")
        self._is_running = True

        # Start time detection windows
        self._wait_once(EventHandler(lambda event: self._start_detection()), entity=self,
                        event_type=self._EV_START_WINDOW)
        self._schedule_after(self.t0, self._EV_START_WINDOW)

        self.init_spins()

    def stop(self):
        """
        Stop connection
        :return: None
        """
        logger.debug("Stopping absorption connection")
        self._is_running = False
        self._in_window = False

    def _start_detection(self):
        """
        Start the detection window
        :return: None
        """
        logger.debug("Start of detection window on absorption connection")

        self._reset_clicks()

        self._in_window = True
        self._wait_once(EventHandler(lambda event: self._stop_detection()), entity=self,
                        event_type=self._EV_STOP_WINDOW)
        self._schedule_after(self.time_window, self._EV_STOP_WINDOW)

    def _stop_detection(self):
        """
        Stop the detection window
        :return: None
        """
        logger.debug("End of detection window on absorption connection")
        self._in_window = False
        self._wait_once(EventHandler(lambda event: self._start_detection()), entity=self,
                        event_type=self._EV_START_WINDOW)

        self._schedule_after(self.t_cycle - self.time_window, self._EV_START_WINDOW)

        # In simulation we simply measured photons in the end of the time-window
        self._absorb_photon(from_left=True)
        self._absorb_photon(from_left=False)

        self._handle_clicks(from_left=True)
        self._handle_clicks(from_left=False)

        self.detection_window_post_processing()

    def detection_window_post_processing(self):
        """
        Determines what should be done after the detection time window is over.
        For example, reseting the spins.
        Should be subclassed.
        :return:
        """
        pass

    def init_spins(self):
        """
        Reset the two spins used.
        :return: None
        """
        logger.debug("Midpoint reseting spins on absorption connection")

        q_program = QuantumProgram()
        qA, qB = q_program.get_qubit_indices(2)

        # Re-initialize spins after measurement (could be done by flipping depending on outcome?)
        q_program.apply(INSTR_INIT, qubit_indices=qA)
        q_program.apply(INSTR_INIT, qubit_indices=qB)
        self.QPD.execute_program(q_program, qubit_mapping=[self.com_physical_IDA, self.com_physical_IDB])


class NoResetAbsorptionConnection(BaseAbsorptionConnection):
    """
    Connection using a midpoint which absorbs photons and makes a Bell-measurement
    upon receiving two photon absorption detections.
    This connection never resets the local memory spins at the midpoint.
    """

    _PROGRAM_CORRECTION = 1
    _PROGRAM_BELL_MEAS = 2
    _PROGRAM_INIT = 3

    def __init__(self, name, **kwargs):
        self._current_program = None, None

        super().__init__(name=name, **kwargs)

        self.QPD.set_program_done_callback(self.handle_program_completion, once=False)

        self.successful_absorbA = False
        self.successful_absorbB = False

    def handle_absorptionA(self):
        """
        Handle absorption from A
        :return: None
        """
        logger.debug("Handling succesful absorption event from left node")
        self.successful_absorbA = True
        if self.successful_absorbB:
            self.handle_two_absorptions()

    def handle_absorptionB(self):
        """
        Handle absorption from A
        :return: None
        """
        logger.debug("Handling succesful absorption event from right node")
        self.successful_absorbB = True
        if self.successful_absorbA:
            self.handle_two_absorptions()

    def handle_failed_absorptionA(self):
        logger.warning("Absorption failed from node A")

    def handle_failed_absorptionB(self):
        logger.warning("Absorption failed from node B")

    def handle_no_absorptionA(self):
        # pass
        logger.warning("No click occurred within time window, no photon from from node A")

    def handle_no_absorptionB(self):
        # pass
        logger.warning("No click occurred within time window, no photon from from node B")

    def handle_two_absorptions(self):
        """
        Handle absorptions from both A and B
        :return: None
        """
        # Apply corrections depending on absorption measurement
        self.apply_absorption_corrections()

    def reset_successful_absorb_data(self):
        """
        Resets data whether absorptions where successful
        :return: None
        """
        self.successful_absorbA = False
        self.successful_absorbB = False

    def reset_current_program(self):
        """
        Resets the current program data
        :return: None
        """
        self._current_program = None, None

    def apply_absorption_corrections(self):
        logger.debug("Midpoint applying corrections on absorption connection")
        assert self.clickedA[0] != self.clickedA[1]
        assert self.clickedB[0] != self.clickedB[1]
        mA = self.clickedA[0]
        mB = self.clickedB[0]

        if (mA == 0) and (mB == 0):
            self.handle_correction()
        else:
            q_program = QuantumProgram()
            qA, qB = q_program.get_qubit_indices(2)
            if mA == 1:
                q_program.apply(INSTR_Z, qubit_indices=qA)
            if mB == 1:
                q_program.apply(INSTR_Z, qubit_indices=qB)
            self._current_program = self._PROGRAM_CORRECTION, q_program
            self.QPD.execute_program(q_program, qubit_mapping=[self.com_physical_IDA, self.com_physical_IDB])

    def handle_program_completion(self):
        """
        Handler for when a quantum program has finished execution
        :return: None
        """
        if self._current_program[0] == self._PROGRAM_CORRECTION:
            self.handle_correction()
        elif self._current_program[0] == self._PROGRAM_BELL_MEAS:
            self.handle_bell_meas()
        elif self._current_program[0] == self._PROGRAM_INIT:
            self.handle_init()
        else:
            raise RuntimeError("Unkown current program")

    def handle_correction(self):
        """
        Handles the completion of the correction
        :return: None
        """
        self.reset_current_program()

        # Perform Bell-measurement
        self.perform_bell_meas()

    def perform_bell_meas(self):
        """
        Performs a Bell measurement between the spins that absorbed photons
        :return: The measurement outcome
        :rtype: [int, int]
        """
        logger.debug("Midpoint performing Bell measurement on absorption connection")

        q_program = QuantumProgram()

        qA, qB = q_program.get_qubit_indices(2)
        q_program.apply(INSTR_CNOT, qubit_indices=[qA, qB])
        q_program.apply(INSTR_H, qubit_indices=qA)
        q_program.apply(INSTR_MEASURE, qubit_indices=qA, output_key='m1')
        q_program.apply(INSTR_MEASURE, qubit_indices=qB, output_key='m2')

        self._current_program = self._PROGRAM_BELL_MEAS, q_program

        self.QPD.execute_program(q_program, qubit_mapping=[self.com_physical_IDA, self.com_physical_IDB])

    def handle_bell_meas(self):
        """
        Handles the completion of the bell measurement.
        :return: None
        """

        # Prepare classical message
        q_program = self._current_program[1]
        m1 = q_program.output['m1'][0]
        m2 = q_program.output['m2'][0]

        logger.debug("Midpoint finished Bell measurement and sending m1={} and m2={} to nodes".format(m1, m2))

        self.reset_current_program()

        c_msg = "{}{}".format(m1, m2)

        # Send measurement outcome
        self.send_classical_to_both(c_msg)

        # Reset the two spins
        self.init_spins()

    def handle_init(self):
        """
        Handles the completion of the re-initialization of spins
        :return:
        """
        logger.debug("Midpoint finished reseting spins on absorption connection")
        self.reset_current_program()

    def detection_window_post_processing(self):
        self.reset_successful_absorb_data()

    def init_spins(self):
        """
        Reset the two spins used.
        :return: None
        """
        super().init_spins()
        self._current_program = self._PROGRAM_INIT, None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: repeater.py
# 
# This file is part of the NetSquid package (https://netsquid.org).
# It is subject to the NetSquid Software End User License Conditions.
# A copy of these conditions can be found in the LICENSE.md file of this package.
# 
# NetSquid Authors
# ================
# 
# NetSquid is being developed within [Quantum Internet division](https://qutech.nl/research-engineering/quantum-internet/) at QuTech.
# QuTech is a collaboration between TNO and the TUDelft.
# 
# Active authors (alphabetical):
# 
# - Tim Coopmans (scientific contributor)
# - Chris Elenbaas (software developer)
# - David Elkouss (scientific supervisor)
# - Rob Knegjens (tech lead, software architect)
# - Iñaki Martin Soroa (software developer)
# - Julio de Oliveira Filho (software architect)
# - Ariana Torres Knoop (HPC contributor)
# - Stephanie Wehner (scientific supervisor)
# 
# Past authors (alphabetical):
# 
# - Axel Dahlberg (scientific contributor)
# - Damian Podareanu (HPC contributor)
# - Walter de Jong (HPC contributor)
# - Loek Nijsten (software developer)
# - Martijn Papendrecht (software developer)
# - Filip Rozpedek (scientific contributor)
# - Matt Skrzypczyk (software contributor)
# - Leon Wubben (software developer)
# 
# The simulation engine of NetSquid depends on the pyDynAA package,
# which is developed at TNO by Julio de Oliveira Filho, Rob Knegjens, Coen van Leeuwen, and Joost Adriaanse.
# 
# Ariana Torres Knoop, Walter de Jong and Damian Podareanu from SURFsara have contributed towards the optimization and parallelization of NetSquid.
# 
# Hana Jirovska and Chris Elenbaas have built Python packages for MacOS.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file uses NumPy style docstrings: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""In this example we will show how to setup a basic repeater network simulation,
combining the entangle nodes and purify protocols.

Let *A* and *B* be the two nodes to be entangled and *R* the intermediary node with which they both share a bipartite state.
Each node has a quantum processor with a number of memory positions: A and B have two while R has four.
A quantum processor is an quantum memory with the ability to run programs on its memory positions,
as explained in the `quantum processor tutorial section <tutorial.quantumprocessor.html>`_.
Next to that, both nodes A and B hold a quantum source.
Combined, the network looks like this:

.. aafig::
    :proportional:

    +---------+                                        +---------+                                        +---------+
    |         | +------------------------------------+ |         | +------------------------------------+ |         |
    |         | |                                    | |         | |                                    | |         |
    |         O-* "Connection: QuantumChannel -->"   *-O         O-* "Connection: QuantumChannel <--"   *-O         |
    |         | |                                    | |         | |                                    | |         |
    |         | +------------------------------------+ |         | +------------------------------------+ |         |
    | "NodeA" |                                        | "NodeR" |                                        | "NodeB" |
    |         | +------------------------------------+ |         | +------------------------------------+ |         |
    |         | |                                    | |         | |                                    | |         |
    |         O-* "Connection: ClassicalChannels <->"*-O         O-* "Connection: ClassicalChannels <->"*-O         |
    |         | |                                    | |         | |                                    | |         |
    |         | +------------------------------------+ |         | +------------------------------------+ |         |
    +---------+                                        +---------+                                        +---------+

To create this network with node A, R and B fully linked via classical and quantum
connections we call :func:`~netsquid.examples.repreater.example_network_setup()`.
Note that we create the quantum processors with ``fallback_to_nonphysical=True``
resulting in the use of ideal instructions without noise.

.. literalinclude:: ../../netsquid/examples/repeater.py
    :pyobject: example_network_setup


Nodes A and R and nodes B and R both generate entanglement following the
:class:`~netsquid.examples.entanglenodes.EntangleNodes` and :class:`~netsquid.examples.purify.Distil` or
:class:`~netsquid.examples.purify.Filter` purification protocols.
To do the entanglement swapping we define a :class:`~netsquid.examples.repeater.Repeater` protocol.

.. literalinclude:: ../../netsquid/examples/repeater.py
    :pyobject: Repeater
    :end-before: Parameters
    :append: ... <remaining code omitted>

The `Repeater` protocol take over from entanglement generation and purification as follows.
Each node waits for qubit input on their quantum memory before beginning.
Node R, once ready, performs a Bell measurement on its two qubits and sends the results via a classical channel to node B.
When B has received its qubit input and the repeater results, it applies the teleport corrections
and signals its qubit is available as output on its quantum memory.

To conveniently bundle all these local protocols together we define the non-local :class:`~netsquid.examples.repeater.RepeaterExample` protocol,
which stores them as sub-protocols and does the linking of success signals to the starting event expressions.

.. literalinclude:: ../../netsquid/examples/repeater.py
    :pyobject: RepeaterExample
    :end-before: def _add_subprotocols
    :append: ... <remaining code omitted>

The *run()* method takes care of iterating for the specified number of runs and
returning a final success signal for each run containing statistics for that run.

.. literalinclude:: ../../netsquid/examples/repeater.py
    :pyobject: RepeaterExample.run
    :dedent: 4

To set up the simulation we use the :class:`~netsquid.examples.repeater.example_sim_setup()` function including
a data collector storing the squared fidelity after each success:

.. literalinclude:: ../../netsquid/examples/repeater.py
    :pyobject: example_sim_setup

The following script runs an example simulation:

>>> import netsquid as ns
>>> print("This example module is located at: "
...       "{}".format(ns.examples.repeater.__file__))
This example module is located at: .../netsquid/examples/repeater.py
>>> from netsquid.examples.repeater import example_network_setup, example_sim_setup
>>> network = example_network_setup()
>>> repeater_example, dc = example_sim_setup(
...     network.get_node("node_A"), network.get_node("node_B"),
...     network.get_node("node_R"), num_runs=1000)
>>> repeater_example.start()
>>> ns.sim_run()
>>> print("Average fidelity of generated entanglement via a repeater "
...       "and with filtering: {}".format(dc.dataframe["F2"].mean()))
Average fidelity of generated entanglement via a repeater and with filtering: ...

"""
import pydynaa as pd
import netsquid as ns
from netsquid.components import instructions as instr
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.protocols.nodeprotocols import LocalProtocol, NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.component import Message, Port
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.examples.purify import Filter, Distil
from pydynaa import EventExpression

__all__ = [
    "Repeater",
    "RepeaterExample",
    "example_network_setup",
    "example_sim_setup",
]


class Repeater(NodeProtocol):
    """Entangles two nodes given both are entangled with an intermediary midpoint node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node to run the repeater or corrector protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication between the repeater and
        corrector node.
    role : "repeater" or "corrector"
        Whether this protocol should act as a repeater or a corrector. Both are needed.
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting.
        The EventExpression should have a protocol as source, this protocol should
        signal the quantum memory position of the qubit. In the case of a midpoint
        repeater it requires two such protocols, both signalling a quantum memory
        position
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    MSG_HEADER = "repeater:corrections"

    def __init__(self, node, port, role, start_expression=None, name=None):
        if role.lower() not in ["repeater", "corrector"]:
            raise ValueError
        self.role = role.lower()
        name = name if name else "Repeater({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        self.start_expression = start_expression
        self.port = port
        # Used by corrector:
        self._correction = None
        self._mem_pos = None

    @property
    def start_expression(self):
        return self._start_expression

    @start_expression.setter
    def start_expression(self, value):
        if value:
            if not isinstance(value, EventExpression):
                raise TypeError("Start expression of the corrector role should be an "
                                "event expression")
            elif self.role == "repeater" and value.type != EventExpression.AND:
                raise TypeError("Start expression of the repeater role should be an "
                                "expression that returns two values.")
        self._start_expression = value

    def run(self):
        if self.role == "repeater":
            yield from self._run_repeater()
        else:
            yield from self._run_corrector()

    def _run_repeater(self):
        # Run loop for midpoint repeater node
        while True:
            evexpr = yield self.start_expression
            assert evexpr.first_term.value and evexpr.second_term.value
            source_A = evexpr.first_term.atomic_source
            source_B = evexpr.second_term.atomic_source
            signal_A = source_A.get_signal_by_event(
                evexpr.first_term.triggered_events[0], self)
            signal_B = source_B.get_signal_by_event(
                evexpr.second_term.triggered_events[0], self)
            # Run bell state measurement program
            measure_program = BellMeasurementProgram()
            pos_A = signal_A.result
            pos_B = signal_B.result
            yield self.node.qmemory.execute_program(measure_program, [pos_A, pos_B])
            m, = measure_program.output["BellStateIndex"]
            # Send measurement to B
            self.port.tx_output(Message([m], header=self.MSG_HEADER))

            self.send_signal(Signals.SUCCESS)

    def _run_corrector(self):
        # Run loop for endpoint corrector node
        port_expression = self.await_port_input(self.port)
        while True:
            evexpr = yield self.start_expression | port_expression
            if evexpr.second_term.value:
                cmessage = self.port.rx_input(header=self.MSG_HEADER)
                if cmessage:
                    self._correction = cmessage.items
            else:
                source_protocol = evexpr.first_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=evexpr.first_term.triggered_events[0], receiver=self)
                self._mem_pos = ready_signal.result
            if self._mem_pos is not None and self._correction is not None:
                yield from self._do_corrections()

    def _do_corrections(self):
        m = self._correction[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
            self.node.qmemory.execute_instruction(instr.INSTR_X, [self._mem_pos])
        if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
            self.node.qmemory.execute_instruction(instr.INSTR_Z, [self._mem_pos])
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.send_signal(Signals.SUCCESS, self._mem_pos)
        # Reset values
        self._mem_pos = None
        self._correction = None

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned([self.node], Node):
            return False
        if not self.check_assigned([self.port], Port):
            return False
        if self.role == "repeater" and (self.node.qmemory is None or
                                        self.node.qmemory.num_positions < 2):
            return False
        if self.role == "corrector" and (self.node.qmemory is None or
                                         self.node.qmemory.num_positions < 1):
            return False
        return True


class BellMeasurementProgram(QuantumProgram):
    """Program to perform a Bell measurement on two qubits.

    Measurement results are stored in output key "BellStateIndex""

    """
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_MEASURE_BELL, [q1, q2], inplace=False,
                   output_key="BellStateIndex")
        yield self.run()


class RepeaterExample(LocalProtocol):
    """Protocol for a complete repeater experiment including purification.

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_A : :py:class:`~netsquid.nodes.node.Node`
        Node to be entangled via repeater.
        Must be specified before protocol can start.
    node_B : :py:class:`~netsquid.nodes.node.Node`
        Node to be entangled via repeater.
        Must be specified before protocol can start.
    node_R : :py:class:`~netsquid.nodes.node.Node`
        Repeater node that will entangle nodes A and B.
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    purify : "filter" or "distil" or None, optional
        Purification protocol to run. If None, no purification is done.
    epsilon : float
        Parameter used in filter's measurement operator.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A to entangle with R.
    entangle_Ra : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node R to entangle with A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B to entangle with R.
    entangle_Rb : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node R to entangle with B.
    purify_A : :class:`Filter` or :class:`Distil`
        Purification protocol running on node A to purify entanglement with R.
    purify_Ra : :class:`Filter` or :class:`Distil`
        Purification protocol running on node R to purify entanglement with A.
    purify_B : :class:`Filter` or :class:`Distil`
        Purification protocol running on node B to purify entanglement with R.
    purify_Rb : :class:`Filter` or :class:`Distil`
        Purification protocol running on node R to purify entanglement with B.
    repeater_R : :class:`~netsquid.examples.repeater.Repeater`
        Repeater protocol running on node R to do midpoint entanglement swap.
    repeater_B : :class:`~netsquid.examples.repeater.Repeater`
        Repeater protocol running on node B to do endpoint correction.

    """

    def __init__(self, node_A, node_B, node_R, num_runs, purify="filter", epsilon=0.3):
        super().__init__(nodes={"A": node_A, "B": node_B, "R": node_R},
                         name="Repeater with purification example")
        self.num_runs = num_runs
        purify = purify.lower()
        if purify not in ("filter", "distil"):
            raise ValueError("{} unknown purify option".format(purify))
        self._add_subprotocols(node_A, node_B, node_R, purify, epsilon)
        # Set entangle start expressions
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
            self.subprotocols["purify_A"], Signals.FAIL) |
            self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
            self.subprotocols["purify_B"], Signals.FAIL) |
            self.subprotocols["entangle_B"].await_signal(self, Signals.WAITING))
        self.subprotocols["entangle_B"].start_expression = start_expr_ent_B
        # Set purify start expressions
        self._start_on_success("purify_A", "entangle_A")
        self._start_on_success("purify_Ra", "entangle_Ra")
        self._start_on_success("purify_B", "entangle_B")
        self._start_on_success("purify_Rb", "entangle_Rb")
        # Set repeater start expressions
        self._start_on_success("repeater_B", "purify_B")
        start_expr_repeater = (self.subprotocols["repeater_R"].await_signal(
            self.subprotocols["purify_Ra"], Signals.SUCCESS) &
            self.subprotocols["repeater_R"].await_signal(
                self.subprotocols["purify_Rb"], Signals.SUCCESS))
        self.subprotocols["repeater_R"].start_expression = start_expr_repeater

    def _start_on_success(self, start_subprotocol, success_subprotocol):
        # Convenience method to set subprotocol's start expression to be success of another
        self.subprotocols[start_subprotocol].start_expression = (
            self.subprotocols[start_subprotocol].await_signal(
                self.subprotocols[success_subprotocol], Signals.SUCCESS))

    def _add_subprotocols(self, node_A, node_B, node_R, purify, epsilon):
        # Setup all of the subprotocols
        purify = purify.lower()
        # Add entangle subprotocols
        pairs = 2 if purify == "distil" else 1
        self.add_subprotocol(EntangleNodes(
            node=node_A, role="source", input_mem_pos=0, num_pairs=pairs, name="entangle_A"))
        self.add_subprotocol(EntangleNodes(
            node=node_B, role="source", input_mem_pos=0, num_pairs=pairs, name="entangle_B"))
        self.add_subprotocol(EntangleNodes(
            node=node_R, role="receiver", input_mem_pos=0, num_pairs=pairs, name="entangle_Ra"))
        self.add_subprotocol(EntangleNodes(
            node=node_R, role="receiver", input_mem_pos=1, num_pairs=pairs, name="entangle_Rb"))
        # Add purify subprotocols
        if purify == "filter":
            purify_cls, kwargs = Filter, {"epsilon": epsilon}
        else:
            distil_role = "A"
            purify_cls, kwargs = Distil, {"role": distil_role}
        for node1, node2, name, distil_role in [
                (node_A, node_R, "purify_A", "A"),
                (node_R, node_A, "purify_Ra", "B"),
                (node_B, node_R, "purify_B", "A"),
                (node_R, node_B, "purify_Rb", "B")]:
            self.add_subprotocol(purify_cls(
                node1, port=node1.get_conn_port(node2.ID), name=name, **kwargs))
        # Add repeater subprotocols
        self.add_subprotocol(Repeater(
            node_R, node_R.get_conn_port(node_B.ID), role="repeater", name="repeater_R"))
        self.add_subprotocol(Repeater(
            node_B, node_B.get_conn_port(node_R.ID), role="corrector", name="repeater_B"))

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.subprotocols["entangle_B"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["repeater_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(
                label=Signals.SUCCESS, receiver=self)
            signal_B = self.subprotocols["repeater_B"].get_signal_result(
                label=Signals.SUCCESS, receiver=self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs_A": self.subprotocols["entangle_A"].entangled_pairs,
                "pairs_B": self.subprotocols["entangle_B"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1000,
                          node_distance=20):
    """Create an example network for use with the repeater protocols.

    Returns
    -------
    :class:`~netsquid.components.component.Component`
        A network component with nodes and channels as subcomponents.

    Notes
    -----
        This network is also used by the matching integration test.

    """
    network = Network("Repeater_network")
    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a, node_b, node_r = network.add_nodes(["node_A", "node_B", "node_R"])
    # Setup end-node A:
    node_a.add_subcomponent(QuantumProcessor(
        "quantum_processor_a", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    source_a = QSource(
        "QSource_A", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)})
    node_a.add_subcomponent(source_a)
    # Setup end-node B:
    node_b.add_subcomponent(QuantumProcessor(
        "quantum_processor_b", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    source_b = QSource(
        "QSource_B", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)})
    node_b.add_subcomponent(source_b)
    # Setup midpoint repeater node R
    node_r.add_subcomponent(QuantumProcessor(
        "quantum_processor_r", num_positions=4, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    # Setup classical connections
    conn_cfibre_ar = DirectConnection(
        "CChannelConn_AR",
        ClassicalChannel("CChannel_A->R", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_R->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_r, connection=conn_cfibre_ar)
    conn_cfibre_br = DirectConnection(
        "CChannelConn_BR",
        ClassicalChannel("CChannel_B->R", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_R->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_b, node_r, connection=conn_cfibre_br)
    # Setup quantum channels
    qchannel_ar = QuantumChannel(
        "QChannel_A->R", length=node_distance,
        models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=200e3)})
    port_name_a, port_name_ra = network.add_connection(
        node_a, node_r, channel_to=qchannel_ar, label="quantum")
    qchannel_br = QuantumChannel(
        "QChannel_B->R", length=node_distance,
        models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=200e3)})
    port_name_b, port_name_rb = network.add_connection(
        node_b, node_r, channel_to=qchannel_br, label="quantum")
    # Setup Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    # Setup Bob ports:
    node_b.subcomponents["QSource_B"].ports["qout1"].forward_output(
        node_b.ports[port_name_b])
    node_b.subcomponents["QSource_B"].ports["qout0"].connect(
        node_b.qmemory.ports["qin0"])
    # Setup repeater ports:
    node_r.ports[port_name_ra].forward_input(node_r.qmemory.ports["qin0"])
    node_r.ports[port_name_rb].forward_input(node_r.qmemory.ports["qin1"])
    return network


def example_sim_setup(node_A, node_B, node_R, num_runs, purify="filter", epsilon=0.3):
    """Example simulation setup of repeater protocol.

    Returns
    -------
    :class:`~netsquid.examples.repeater.RepeaterExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    repeater_example = RepeaterExample(
        node_A, node_B, node_R, num_runs=num_runs, purify=purify, epsilon=0.3)

    def record_run(evexpr):
        # Record a repeater run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_a, = node_A.qmemory.pop(positions=[result["pos_A"]])
        q_b, = node_B.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_a, q_b], ks.b00, squared=True)
        return {"F2": f2,
                "pairs_A": result["pairs_A"],
                "pairs_B": result["pairs_B"],
                "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False, include_entity_name=False)
    dc.collect_on(pd.EventExpression(
        source=repeater_example, event_type=Signals.SUCCESS.value))
    return repeater_example, dc


if __name__ == "__main__":
    network = example_network_setup()
    repeater_example, dc = example_sim_setup(network.get_node("node_A"),
                                             network.get_node("node_B"),
                                             network.get_node("node_R"),
                                             num_runs=1000)
    repeater_example.start()
    ns.sim_run()
    print("Average fidelity of generated entanglement via a repeater and with "
          "filtering: {}".format(dc.dataframe["F2"].mean()))

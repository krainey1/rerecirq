#code needed for conversion, adjusted to fix bad register name issue
import cmath
import re
from logging import warning
from typing import Any, cast

import cirq_google
from sympy import Basic, Symbol, pi

import cirq.ops
from cirq.devices import GridQubit, LineQubit
from pytket.architecture import Architecture
from pytket.circuit import Bit, Circuit, Node, OpType, Qubit

# For translating cirq circuits to tket circuits
cirq_common = cirq.ops.common_gates
cirq_pauli = cirq.ops.pauli_gates

cirq_CH = cirq_common.H.controlled(1)

# map cirq common gates to pytket gates
_cirq2ops_mapping = {
    cirq_common.CNOT: OpType.CX,
    cirq_common.H: OpType.H,
    cirq_common.MeasurementGate: OpType.Measure,
    cirq_common.XPowGate: OpType.Rx,
    cirq_common.YPowGate: OpType.Ry,
    cirq_common.ZPowGate: OpType.Rz,
    cirq_common.XPowGate(exponent=0.5): OpType.V,
    cirq_common.XPowGate(exponent=-0.5): OpType.Vdg,
    cirq_common.S: OpType.S,
    cirq_common.SWAP: OpType.SWAP,
    cirq_common.T: OpType.T,
    cirq_pauli.X: OpType.X,
    cirq_pauli.Y: OpType.Y,
    cirq_pauli.Z: OpType.Z,
    cirq.ops.I: OpType.noop,
    cirq_common.CZPowGate: OpType.CU1,
    cirq_common.CZ: OpType.CZ,
    cirq_CH: OpType.CH,
    cirq.ops.CSwapGate: OpType.CSWAP,
    cirq_common.ISwapPowGate: OpType.ISWAP,
    cirq_common.ISWAP: OpType.ISWAPMax,
    cirq.ops.FSimGate: OpType.FSim,
    cirq_google.SYC: OpType.Sycamore,
    cirq.ops.parity_gates.ZZPowGate: OpType.ZZPhase,
    cirq.ops.parity_gates.XXPowGate: OpType.XXPhase,
    cirq.ops.parity_gates.YYPowGate: OpType.YYPhase,
    cirq.ops.PhasedXPowGate: OpType.PhasedX,
    cirq.ops.PhasedISwapPowGate: OpType.PhasedISWAP,
    cirq.ops.common_channels.ResetChannel: OpType.Reset,
}
# reverse mapping for convenience
_ops2cirq_mapping: dict = dict((item[1], item[0]) for item in _cirq2ops_mapping.items())  # noqa: C402
# spot special rotation gates
_constant_gates = (
    cirq_common.CNOT,
    cirq_common.H,
    cirq_common.S,
    cirq_common.SWAP,
    cirq_common.T,
    cirq_pauli.X,
    cirq_pauli.Y,
    cirq_pauli.Z,
    cirq_common.CZ,
    cirq_CH,
    cirq_common.ISWAP,
    cirq_google.SYC,
    cirq.ops.I,
)

_radian_gates = (
    cirq_common.Rx,
    cirq_common.Ry,
    cirq_common.Rz,
)
_cirq2ops_radians_mapping = {
    cirq_common.Rx: OpType.Rx,
    cirq_common.Ry: OpType.Ry,
    cirq_common.Rz: OpType.Rz,
}


def cirq_to_tk(circuit: cirq.circuits.Circuit) -> Circuit:  # noqa: PLR0912, PLR0915
    """Converts a Cirq :py:class:`Circuit` to a tket :py:class:`Circuit` object.

    :param circuit: The input Cirq :py:class:`Circuit`

    :raises NotImplementedError: If the input contains a Cirq :py:class:`Circuit`
        operation which is not yet supported by pytket

    :return: The tket :py:class:`Circuit` corresponding to the input circuit
    """
    tkcirc = Circuit()
    qmap = {}
    i = 0
    for qb in circuit.all_qubits():
        if isinstance(qb, LineQubit):
            uid = Qubit("q", qb.x)
        elif isinstance(qb, GridQubit):
            uid = Qubit("g", qb.row, qb.col)
        elif isinstance(qb, cirq.ops.NamedQubit):
            uid = Qubit(qb.name)
        else:
            raise NotImplementedError("Cannot convert qubits of type " + str(type(qb)))
        tkcirc.add_qubit(uid)
        qmap.update({qb: uid})
    for moment in circuit:
        for op in moment.operations:
            gate = op.gate
            gatetype = type(gate)
            qb_lst = [qmap[q] for q in op.qubits]
            if isinstance(gate, cirq.ops.global_phase_op.GlobalPhaseGate):
                tkcirc.add_phase(cmath.phase(gate.coefficient) / pi)
                continue
            if isinstance(gate, cirq_common.HPowGate) and gate.exponent == 1:
                gate = cirq_common.H
            elif (
                gatetype == cirq_common.CNotPowGate
                and cast("cirq_common.CNotPowGate", gate).exponent == 1
            ):
                gate = cirq_common.CNOT
            elif (
                gatetype == cirq_pauli._PauliX  # noqa: SLF001
                and cast("cirq_pauli._PauliX", gate).exponent == 1  # noqa: SLF001
            ):
                gate = cirq_pauli.X
            elif (
                gatetype == cirq_pauli._PauliY  # noqa: SLF001
                and cast("cirq_pauli._PauliY", gate).exponent == 1  # noqa: SLF001
            ):
                gate = cirq_pauli.Y
            elif (
                gatetype == cirq_pauli._PauliZ  # noqa: SLF001
                and cast("cirq_pauli._PauliZ", gate).exponent == 1  # noqa: SLF001
            ):
                gate = cirq_pauli.Z

            apply_in_parallel = False
            if isinstance(gate, cirq.ops.ParallelGate):
                if gate.num_copies != len(qb_lst):
                    raise NotImplementedError(
                        "ParallelGate parameters defined incorrectly."
                    )
                gate = gate.sub_gate
                gatetype = type(gate)
                apply_in_parallel = True

            if gate in _constant_gates:
                try:
                    optype = _cirq2ops_mapping[gate]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params: list[float | Basic | Symbol] = []
            elif isinstance(gate, cirq.ops.common_channels.ResetChannel):
                optype = OpType.Reset
                params = []
            elif gatetype in _radian_gates:
                try:
                    optype = _cirq2ops_radians_mapping[
                        cast("type[cirq.ops.EigenGate]", gatetype)
                    ]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params = [gate._rads / pi]  # type: ignore  # noqa: SLF001
            elif isinstance(gate, cirq_common.MeasurementGate):
                # Adding "_b" to the bit uid since for cirq.NamedQubit,
                # the gate.key is equal to the qubit id (the qubit name)
                bitid = Bit("c", [i]) #no longer produces bad register name q(0,0)_b, etc. -> c[0]...
                i += 1
                tkcirc.add_bit(bitid)
                assert len(qb_lst) == 1
                tkcirc.Measure(qb_lst[0], bitid)
                continue
            elif isinstance(gate, cirq.ops.PhasedXPowGate):
                optype = OpType.PhasedX
                pe = gate.phase_exponent
                params = [gate.exponent, pe]
            elif isinstance(gate, cirq.ops.FSimGate):
                optype = OpType.FSim
                params = [gate.theta / pi, gate.phi / pi]
            elif isinstance(gate, cirq.ops.PhasedISwapPowGate):
                optype = OpType.PhasedISWAP
                params = [gate.phase_exponent, gate.exponent]
            else:
                try:
                    optype = _cirq2ops_mapping[gatetype]
                    params = [cast("Any", gate).exponent]
                except (KeyError, AttributeError) as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
            if apply_in_parallel:
                for qbit in qb_lst:
                    tkcirc.add_gate(optype, params, [qbit])
            else:
                tkcirc.add_gate(optype, params, qb_lst)
    return tkcirc


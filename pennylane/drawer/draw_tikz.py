# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The public ``draw_tikz`` circuit-drawing function."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Literal

from pennylane.allocation import DynamicWire
from pennylane.core.qscript import make_qscript
from pennylane.workflow import construct_batch

from .draw import _apply_partial_args, _unwrap_partial, catalyst_qjit
from .tape_tikz import tape_tikz


def _fallback_wire_order(tape):
    """Return a stable wire order for a tape without externally specified wires."""
    wires = [wire for wire in tape.wires if not isinstance(wire, DynamicWire)]
    try:
        return sorted(wires)
    except TypeError:
        return wires


def draw_tikz(
    qnode: Callable,
    wire_order: Sequence | None = None,
    show_all_wires: bool = False,
    decimals: int | None = 2,
    *,
    max_length: int | None = None,
    show_matrices: bool = True,
    show_wire_labels: bool = True,
    level: Literal["top", "user", "device", "gradient"] | int | slice = "gradient",
):
    r"""Create a callable that returns a TikZ circuit diagram as a string.

    Args:
        qnode (.QNode or Callable): QNode or quantum function to draw.
        wire_order (Sequence[Any] or None): Wire order from top to bottom.
        show_all_wires (bool): Whether to include empty wires from ``wire_order`` or the device.
        decimals (int or None): Decimal places used in operation parameters. ``None`` omits them.
        max_length (int or None): Maximum number of operation layers in each vertically stacked
            TikZ scope. Wrapped scopes include continuation dots on every wire.
        show_matrices (bool): Whether to display matrix-valued parameters beneath the circuit.
        show_wire_labels (bool): Whether to include labels to the left of the wires.
        level (str, int, or slice): QNode transform level to apply before drawing.

    Returns:
        Callable: A callable with the same arguments as ``qnode``. Calling it returns a string
        containing one or more ``tikzpicture`` environments.

    The resulting string requires ``\usepackage{tikz}`` in the LaTeX document preamble.
    """
    qnode, partial_args, partial_kwargs = _unwrap_partial(qnode)

    if catalyst_qjit(qnode):
        qnode = qnode.user_function

    if hasattr(qnode, "construct"):
        wrapper = _draw_tikz_qnode(
            qnode,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=show_matrices,
            show_wire_labels=show_wire_labels,
            level=level,
        )
        return _apply_partial_args(wrapper, partial_args, partial_kwargs)

    if level not in {"gradient", 0, "top"}:
        warnings.warn(
            "When the input to qp.draw_tikz is not a QNode, the level argument is ignored.",
            UserWarning,
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tape = make_qscript(qnode)(*args, **kwargs)
        resolved_wire_order = wire_order if wire_order is not None else _fallback_wire_order(tape)
        return tape_tikz(
            tape,
            wire_order=resolved_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=show_matrices,
            show_wire_labels=show_wire_labels,
        )

    return _apply_partial_args(wrapper, partial_args, partial_kwargs)


def _draw_tikz_qnode(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    *,
    max_length=None,
    show_matrices=True,
    show_wire_labels=True,
    level="gradient",
):
    """Create the TikZ wrapper for a QNode."""

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tapes, _ = construct_batch(qnode, level=level)(*args, **kwargs)

        if wire_order is not None:
            resolved_wire_order = wire_order
        elif qnode.device.wires:
            resolved_wire_order = qnode.device.wires
        else:
            resolved_wire_order = _fallback_wire_order(tapes[0])

        return "\n\n".join(
            tape_tikz(
                tape,
                wire_order=resolved_wire_order,
                show_all_wires=show_all_wires,
                decimals=decimals,
                max_length=max_length,
                show_matrices=show_matrices,
                show_wire_labels=show_wire_labels,
            )
            for tape in tapes
        )

    return wrapper

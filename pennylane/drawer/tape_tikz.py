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
""" Draws a TikZ circuit diagram from a quantum tape for use in LaTeX documents."""

from pennylane import math, ops

from .drawable_layers import drawable_layers
from .utils import (
    convert_wire_order,
    cwire_connections,
    default_bit_map,
    transform_deferred_measurements_tape,
    unwrap_controls,
)

_LAYER_SPACING = 1.0
_TIKZ_HEADER = r"""\begin{tikzpicture}[
    x=1cm,
    y=1cm,
    wire/.style={},
    gate/.style={draw, fill=white, minimum width=0.7cm, minimum height=0.55cm, inner sep=2pt},
    wire hider/.style={draw=white, fill=white, minimum height=1.5pt, inner sep=0pt, outer sep=0pt},
    control/.style={circle, fill=black, inner sep=1.5pt},
    open control/.style={circle, draw, fill=white, inner sep=1.5pt},
    measure/.style={draw, fill=white, minimum width=0.7cm, minimum height=0.55cm, inner sep=2pt},
    classical wire/.style={double, double distance=1.2pt, line width=0.45pt},
    matrix label/.style={anchor=north west, inner sep=0pt},
    pics/target/.style={code={
        \draw (0,0) circle[radius=0.22cm];
        \draw (-0.22cm,0) -- (0.22cm,0);
        \draw (0,-0.22cm) -- (0,0.22cm);
    }},
    pics/swap/.style={code={
        \draw (-0.13cm,-0.13cm) -- (0.13cm,0.13cm);
        \draw (-0.13cm,0.13cm) -- (0.13cm,-0.13cm);
    }},
    pics/measure/.style={code={
        \draw (-0.2cm,-0.08cm) arc[start angle=180, end angle=0, radius=0.2cm];
        \draw[->] (0,-0.08cm) -- (0.24cm,0.16cm);
    }},
    pics/continuation/.style={code={
        \node[fill=white, inner sep=1pt] at (0,0) {$\cdots$};
    }}
]"""


def _escape_latex(value):
    """Escape text that will be placed inside a TikZ node."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _format_tikz_label(label):
    """Escape a label and use math mode for a small set of mathematical labels."""
    if len(label) == 1 and label.isalpha():
        return f"${_escape_latex(label)}$"

    for axis in "XYZ":
        rotation = f"R{axis}"
        if label == rotation:
            return rf"$R_{axis}$"
        if label.startswith(f"{rotation} (") and label.endswith(")"):
            parameter = label[len(rotation) + 2 : -1]
            try:
                float(parameter)
            except ValueError:
                continue
            return rf"$R_{axis}({_escape_latex(parameter)})$"

    if label.startswith("U (M") and label.endswith(")"):
        matrix_index = label[4:-1]
        if matrix_index.isdigit():
            return rf"$U(M_{{{matrix_index}}})$"

    return _escape_latex(label)


def _operation_label(op, decimals, cache):
    """Return an operation label suitable for a single TikZ node."""
    label = op.label(decimals=decimals, cache=cache).replace("\n", " ")
    return _format_tikz_label(label)


def _operation_width(op, decimals, cache):
    """Estimate the rendered width of an operation in centimetres."""
    if isinstance(op, ops.Conditional):
        return _operation_width(op.base, decimals, cache)

    if isinstance(op, (ops.MidMeasure, ops.PauliMeasure)):
        label = "M" if op.postselect is None else f"M={int(op.postselect)}"
    else:
        control_wires, _, base = unwrap_controls(op)
        if (control_wires and isinstance(base, ops.PauliX)) or isinstance(base, ops.SWAP):
            return 0.7
        label = base.label(decimals=decimals, cache=cache).replace("\n", " ")

    # TikZ ultimately determines text width using the document font. This conservative
    # estimate uses the average width of a LaTeX character plus the node's inner padding.
    return max(0.7, 0.18 * len(label) + 0.2)


def _segment_geometry(layers, decimals, cache, segment_start_x, layer_x_offset):
    """Return width-aware layer positions, wire end, and terminal measurement position."""
    layer_widths = [
        max((_operation_width(op, decimals, cache) for op in layer), default=0.7)
        for layer in layers
    ]
    layer_x_positions = []

    for index, layer_width in enumerate(layer_widths):
        if index == 0:
            layer_x = max(
                layer_x_offset + _LAYER_SPACING,
                segment_start_x + layer_width / 2 + 0.15,
            )
        else:
            previous_width = layer_widths[index - 1]
            layer_spacing = max(
                _LAYER_SPACING,
                previous_width / 2 + layer_width / 2 + 0.3,
            )
            layer_x = layer_x_positions[-1] + layer_spacing

        layer_x_positions.append(layer_x)

    if not layer_x_positions:
        return layer_x_positions, layer_x_offset + 0.5, layer_x_offset + _LAYER_SPACING

    last_layer_width = layer_widths[-1]
    segment_end_x = layer_x_positions[-1] + max(0.5, last_layer_width / 2 + 0.15)
    measurement_x = layer_x_positions[-1] + max(_LAYER_SPACING, last_layer_width / 2 + 0.65)
    return layer_x_positions, segment_end_x, measurement_x


def _draw_box(x, wires, label, width):
    """Draw a labelled box over one or more wires."""

    # label = "T" # @Temp: replacing labels for testing purposes
    first_wire = min(wires)
    last_wire = max(wires)
    centre = -(first_wire + last_wire) / 2

    if first_wire == last_wire:
        return [rf"\node[gate] at ({x:g},{centre}) {{{label}}};"]

    height = last_wire - first_wire + 0.55
    hidden_wires = sorted(set(range(first_wire, last_wire + 1)) - set(wires))
    if not hidden_wires:
        return [rf"\node[gate, minimum height={height}cm] at ({x:g},{centre}) {{{label}}};"]

    wire_hider_width = width + 0.3
    commands = []
    for wire in hidden_wires:
        commands.append(
            rf"\node[wire hider, minimum width={wire_hider_width:g}cm] "
            rf"at ({x:g},{-wire}) {{}};"
        )

    commands.append(
        rf"\node[gate, minimum width={width:g}cm, minimum height={height}cm] "
        rf"at ({x:g},{centre}) {{{label}}};"
    )

    return commands


def _draw_target(x, wire):
    """Draw the target symbol used by Pauli-X controlled gates."""
    return [rf"\pic at ({x:g},{-wire}) {{target}};"]


def _draw_swap(x, wires):
    """Draw crosses on the target wires of a SWAP operation."""
    return [rf"\pic at ({x:g},{-wire}) {{swap}};" for wire in wires]


def _draw_measure(x, wire):
    """Draw a measurement box containing a gauge symbol."""
    return [
        rf"\node[measure] at ({x:g},{-wire}) {{}};",
        rf"\pic at ({x:g},{-wire}) {{measure}};",
    ]


def _draw_mid_measure(op, x):
    """Draw a mid-circuit measurement box on each measured wire."""
    return [command for wire in op.wires for command in _draw_measure(x, wire)]


def _draw_operation(op, x, decimals, cache):
    """Convert one mapped PennyLane operation into TikZ commands."""
    if isinstance(op, (ops.MidMeasure, ops.PauliMeasure)):
        return _draw_mid_measure(op, x)

    if isinstance(op, ops.Conditional):
        return _draw_operation(op.base, x, decimals, cache)

    control_wires, control_values, base = unwrap_controls(op)
    controls = list(control_wires)
    targets = list(base.wires)
    commands = []

    if controls:
        connected_wires = controls + targets
        commands.append(
            rf"\draw ({x:g},{-min(connected_wires)}) -- ({x:g},{-max(connected_wires)});"
        )
        for wire, value in zip(controls, control_values, strict=True):
            style = "control" if value else "open control"
            commands.append(rf"\node[{style}] at ({x:g},{-wire}) {{}};")

    if controls and isinstance(base, ops.PauliX) and len(targets) == 1:
        commands.extend(_draw_target(x, targets[0]))
    elif isinstance(base, ops.SWAP):
        if len(targets) > 1 and not controls:
            commands.append(rf"\draw ({x:g},{-min(targets)}) -- ({x:g},{-max(targets)});")
        commands.extend(_draw_swap(x, targets))
    else:
        wires = targets or list(op.wires)
        label = _operation_label(base, decimals, cache)
        width = _operation_width(base, decimals, cache)
        commands.extend(_draw_box(x, wires, label, width))

    return commands


def _draw_classical_wires(
    cwire_layers,
    cwire_wires,
    number_of_wires,
    layer_start,
    layer_stop,
    segment_start_x,
    segment_end_x,
    layer_x_positions,
):
    """Draw the part of each classical wire that intersects a circuit segment.

    Layer indices in cwire_layers refer to the complete circuit. They are clipped and shifted
    here so a classical dependency can continue across wrapped circuit segments.
    """
    commands = []
    for classical_wire, layer_stretches in cwire_layers.items():
        lane_y = -(number_of_wires + classical_wire)

        for layer_ids, connected_wires in zip(
            layer_stretches, cwire_wires[classical_wire], strict=True
        ):
            if not layer_ids:
                continue

            padded_wires = list(connected_wires) + [None] * (len(layer_ids) - len(connected_wires))
            stretch_start = layer_ids[0]
            stretch_stop = layer_ids[-1]

            if stretch_stop < layer_start or stretch_start >= layer_stop:
                continue

            accesses = [
                (layer_id, connected_wire)
                for layer_id, connected_wire in zip(layer_ids, padded_wires, strict=True)
                if layer_start <= layer_id < layer_stop
            ]

            starts_here = stretch_start >= layer_start
            stops_here = stretch_stop < layer_stop
            coordinates = []

            if starts_here:
                start_layer, start_wire = accesses[0]
                start_x = layer_x_positions[start_layer - layer_start]
                if start_wire is not None:
                    coordinates.append((start_x, -start_wire))
                coordinates.append((start_x, lane_y))
            else:
                coordinates.append((segment_start_x, lane_y))

            first_middle_access = 1 if starts_here else 0
            last_middle_access = len(accesses) - 1 if stops_here else len(accesses)
            for access_layer, connected_wire in accesses[first_middle_access:last_middle_access]:
                if connected_wire is not None:
                    access_x = layer_x_positions[access_layer - layer_start]
                    coordinates.extend(
                        [(access_x, lane_y), (access_x, -connected_wire), (access_x, lane_y)]
                    )

            if not stops_here:
                coordinates.append((segment_end_x, lane_y))
            elif stretch_start != stretch_stop:
                stop_layer, stop_wire = accesses[-1]
                stop_x = layer_x_positions[stop_layer - layer_start]
                coordinates.append((stop_x, lane_y))
                if stop_wire is not None:
                    coordinates.append((stop_x, -stop_wire))

            coordinates = [
                coordinate
                for index, coordinate in enumerate(coordinates)
                if index == 0 or coordinate != coordinates[index - 1]
            ]
            path = " -- ".join(f"({x:g},{y})" for x, y in coordinates)
            commands.append(rf"\draw[classical wire] {path};")

    return commands


def _measured_wires(tape, number_of_wires):
    """Return mapped wires that receive terminal measurement symbols."""
    measured = set()
    for measurement in tape.measurements:
        if not measurement.wires:
            return set(range(number_of_wires))
        measured.update(measurement.wires)
    return measured


def _draw_continuation_dots(number_of_wires, x):
    """Draw a continuation symbol on every quantum wire."""
    return [rf"\pic at ({x:g},{-wire}) {{continuation}};" for wire in range(number_of_wires)]


def _format_matrix_entry(value, decimals):
    """Format one real or complex matrix entry for LaTeX math mode."""
    try:
        value = complex(value)
    except (TypeError, ValueError):
        return _escape_latex(value)

    tolerance = 10 ** (-(decimals if decimals is not None else 8)) / 2
    real = 0.0 if abs(value.real) < tolerance else value.real
    imag = 0.0 if abs(value.imag) < tolerance else value.imag
    formatter = (
        (lambda component: f"{component:.{decimals}f}")
        if decimals is not None
        else (lambda component: f"{component:g}")
    )

    if imag == 0:
        return formatter(real)
    if real == 0:
        return f"{formatter(imag)}i"

    sign = "+" if imag > 0 else "-"
    return f"{formatter(real)} {sign} {formatter(abs(imag))}i"


def _matrix_latex(matrix, decimals):
    """Convert an array-valued parameter into a LaTeX array."""
    if math.requires_grad(matrix) and hasattr(matrix, "detach"):
        matrix = matrix.detach()
    matrix = math.toarray(matrix)
    shape = math.shape(matrix)

    if len(shape) == 0:
        rows = [[matrix]]
    elif len(shape) == 1:
        rows = [matrix]
    elif len(shape) == 2:
        rows = matrix
    else:
        rows = math.reshape(matrix, (-1, shape[-1]))

    formatted_rows = [
        " & ".join(_format_matrix_entry(value, decimals) for value in row) for row in rows
    ]
    alignment = "c" * max(shape[-1] if shape else 1, 1)
    contents = r" \\ ".join(formatted_rows)
    return rf"\left[\begin{{array}}{{{alignment}}}{contents}\end{{array}}\right]"


def _draw_matrices(matrices, top_y, scope_yshift, decimals):
    """Draw cached matrices below the quantum and classical wires."""
    commands = [rf"\begin{{scope}}[yshift={scope_yshift}cm]"]
    for index, matrix in enumerate(matrices):
        position = (
            f"(0,{top_y})" if index == 0 else f"([yshift=-0.35cm]matrix-{index - 1}.south west)"
        )
        matrix_latex = _matrix_latex(matrix, decimals)
        commands.append(
            rf"\node[matrix label] (matrix-{index}) at {position} "
            rf"{{$M_{{{index}}} = {matrix_latex}$}};"
        )
    commands.append(r"\end{scope}")
    return commands


def _draw_tikz_segment(
    tape,
    layers,
    used_wire_map,
    cwire_layers,
    cwire_wires,
    decimals,
    cache,
    show_wire_labels,
    *,
    layer_start=0,
    scope_yshift=0,
    starting_dots=False,
    ending_dots=False,
):
    """Draw one contiguous segment of a possibly wrapped circuit."""
    number_of_wires = len(used_wire_map)
    layer_x_offset = _LAYER_SPACING / 2 if starting_dots else 0
    segment_start_x = 0.15 + layer_x_offset
    layer_x_positions, segment_end_x, measurement_x = _segment_geometry(
        layers, decimals, cache, segment_start_x, layer_x_offset
    )
    starting_dots_x = segment_start_x - 0.4
    ending_dots_x = segment_end_x + 0.4

    if ending_dots:
        wire_end = segment_end_x
    elif tape.measurements:
        wire_end = measurement_x + 0.5
    else:
        wire_end = max(segment_end_x, layer_x_offset + 1.5)

    commands = [rf"\begin{{scope}}[yshift={scope_yshift}cm]"]
    for wire_label, mapped_wire in used_wire_map.items():
        y = -mapped_wire
        if show_wire_labels:
            label = _escape_latex(wire_label)
            commands.append(rf"\node[anchor=east] at (0,{y}) {{{label}}};")
        commands.append(rf"\draw[wire] ({segment_start_x:g},{y}) -- ({wire_end:g},{y});")

    commands.extend(
        _draw_classical_wires(
            cwire_layers,
            cwire_wires,
            number_of_wires,
            layer_start,
            layer_start + len(layers),
            segment_start_x,
            segment_end_x,
            layer_x_positions,
        )
    )

    for local_layer, layer_operations in enumerate(layers, start=1):
        operation_x = layer_x_positions[local_layer - 1]
        for op in layer_operations:
            print(layer_start + local_layer, op)
            commands.extend(_draw_operation(op, operation_x, decimals, cache))

    if not ending_dots:
        for wire in sorted(_measured_wires(tape, number_of_wires)):
            commands.extend(_draw_measure(measurement_x, wire))

    if starting_dots:
        commands.extend(_draw_continuation_dots(number_of_wires, starting_dots_x))
    if ending_dots:
        commands.extend(_draw_continuation_dots(number_of_wires, ending_dots_x))

    commands.append(r"\end{scope}")
    return commands


def tape_tikz(
    tape,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    *,
    max_length=None,
    show_matrices=True,
    show_wire_labels=True,
    cache=None,
):
    r"""Return a ``tikzpicture`` string representing a quantum tape.

    The returned string can be placed in a LaTeX document that loads ``\usepackage{tikz}``.
    This initial renderer supports ordinary gates, controlled gates, swaps, conditional
    mid-circuit measurements, and terminal measurements.

    When max_length is provided, each scope contains at most that many operation layers.
    Continuation dots connect vertically stacked scopes in one TikZ picture.
    Matrix-valued parameters are labelled ``M0``, ``M1``, and so on. When show_matrices is
    true, their values are displayed below the final circuit scope.
    """
    cache = {} if cache is None else cache
    cache.setdefault("matrices", [])
    tape = transform_deferred_measurements_tape(tape)
    full_wire_map, used_wire_map = convert_wire_order(
        tape, wire_order=wire_order, show_all_wires=show_all_wires
    )
    tape = ops.functions.map_wires(tape, wire_map=full_wire_map)[0][0]

    mapped_wire_map = {wire: wire for wire in full_wire_map.values()}
    bit_map = default_bit_map(tape)
    layers = drawable_layers(tape.operations, wire_map=mapped_wire_map, bit_map=bit_map)
    _, cwire_layers, cwire_wires = cwire_connections(layers, bit_map, mapped_wire_map)

    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    commands = [_TIKZ_HEADER]
    scope_height = max(len(used_wire_map) + len(cwire_layers), 1)
    if max_length is None or len(layers) <= max_length:
        commands.extend(
            _draw_tikz_segment(
                tape,
                layers,
                used_wire_map,
                cwire_layers,
                cwire_wires,
                decimals,
                cache,
                show_wire_labels,
            )
        )
        final_scope_yshift = 0
    else:
        for scope_index, layer_start in enumerate(range(0, len(layers), max_length)):
            layer_stop = min(layer_start + max_length, len(layers))
            commands.extend(
                _draw_tikz_segment(
                    tape,
                    layers[layer_start:layer_stop],
                    used_wire_map,
                    cwire_layers,
                    cwire_wires,
                    decimals,
                    cache,
                    show_wire_labels,
                    layer_start=layer_start,
                    scope_yshift=-scope_index * scope_height,
                    starting_dots=layer_start > 0,
                    ending_dots=layer_stop < len(layers),
                )
            )
        final_scope_yshift = -scope_index * scope_height

    if show_matrices and cache["matrices"]:
        commands.extend(
            _draw_matrices(
                cache["matrices"],
                top_y=-scope_height,
                scope_yshift=final_scope_yshift,
                decimals=decimals,
            )
        )

    commands.append(r"\end{tikzpicture}")
    return "\n".join(commands)

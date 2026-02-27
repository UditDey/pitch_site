#!/usr/bin/env python3
"""
Generate structural Verilog netlists for crossbar and Echo Cache topologies.

Each design is built from explicit mux2 instances + wires.
(No generate blocks, no behavioral always blocks.)

Echo (Option A):
- Top-level pins match crossbar's data/select interface, PLUS one extra pin: mode
- mode=0: shared pins operate the control Omega network (ctrl)
- mode=1: shared pins operate the transfer Omega network (xfer)
- Both Omega networks share the SAME select pins as crossbar:
    For each stage s and switch k:
      lo mux uses sel_{2k}_{s}
      hi mux uses sel_{2k+1}_{s}

Usage:
    python3 gen.py <N> [W]

    N: Number of ports (must be power of 2)
    W: Data width in bits (default: 8)

Outputs:
    crossbar.v   - Crossbar topology netlist
    echo.v       - Echo topology netlist (2x Omega MINs, shared pins + mode)
    crossbar.sdc - Timing constraints for crossbar
    echo.sdc     - Timing constraints for echo
"""

import math
import sys


def gen_mux2(W: int) -> str:
    """Generate a simple W-bit 2:1 mux module (no ternary)."""
    return f"""\
(* keep_hierarchy = "yes" *)
module mux2 (
    input  wire [{W-1}:0] a,
    input  wire [{W-1}:0] b,
    input  wire        sel,
    output wire [{W-1}:0] out
);
    // out = sel ? b : a  (written without ternary)
    assign out = ({{{W}{{~sel}}}} & a) | ({{{W}{{sel}}}} & b);
endmodule
"""


def gen_crossbar(N: int, W: int) -> str:
    """
    Generate a crossbar netlist.

    Structure: Each of N outputs gets a balanced binary mux tree selecting
    from N inputs. Each tree has log2(N) stages and N-1 mux2 instances.

    Total mux2 instances: N * (N - 1)
    """
    STAGES = int(math.log2(N))
    assert 2**STAGES == N, f"N={N} must be a power of 2"

    lines = []
    lines.append(gen_mux2(W))

    # Module header
    lines.append("module crossbar (")

    ports = []
    for i in range(N):
        ports.append(f"    input  wire [{W-1}:0] data_in_{i}")
    for i in range(N):
        ports.append(f"    output wire [{W-1}:0] data_out_{i}")
    for j in range(N):
        for s in range(STAGES):
            ports.append(f"    input  wire sel_{j}_{s}")

    lines.append(",\n".join(ports))
    lines.append(");")
    lines.append("")

    # For each output, build a mux tree
    for j in range(N):
        lines.append(f"    // ---- Mux tree for output {j} ----")
        prev_wires = [f"data_in_{i}" for i in range(N)]

        for s in range(STAGES):
            n_muxes = len(prev_wires) // 2
            curr_wires = []

            for m in range(n_muxes):
                if s == STAGES - 1:
                    assert n_muxes == 1
                    out_name = f"data_out_{j}"
                else:
                    out_name = f"t_{j}_s{s}_m{m}"
                    lines.append(f"    wire [{W-1}:0] {out_name};")

                lines.append(
                    f"    mux2 u_xbar_o{j}_s{s}_m{m} ("
                    f".a({prev_wires[2*m]}), "
                    f".b({prev_wires[2*m+1]}), "
                    f".sel(sel_{j}_{s}), "
                    f".out({out_name}));"
                )
                curr_wires.append(out_name)

            prev_wires = curr_wires
            lines.append("")

    lines.append("endmodule")
    return "\n".join(lines)


def perfect_shuffle(i: int, bits: int) -> int:
    """Left-rotate the 'bits'-bit representation of i by 1 position."""
    return ((i << 1) | (i >> (bits - 1))) & ((1 << bits) - 1)


def sel_for_omega(prefix: str, stage: int, k: int, which: str) -> str:
    """
    Map Omega switch select pins to the crossbar-style top-level selects.

    Omega stage 'stage' has switches k=0..N/2-1, each with two mux outputs:
      - lo output corresponds to index 2k
      - hi output corresponds to index 2k+1

    We reuse top-level pins:
      lo mux sel = sel_{2k}_{stage}
      hi mux sel = sel_{2k+1}_{stage}

    'prefix' is unused here but kept for clarity/extension.
    """
    if which == "lo":
        return f"sel_{2*k}_{stage}"
    elif which == "hi":
        return f"sel_{2*k+1}_{stage}"
    else:
        raise ValueError("which must be 'lo' or 'hi'")


def gen_omega_inline(
    N: int, W: int, STAGES: int, prefix: str
) -> str:
    """
    Generate one Omega network as inline Verilog (wires + mux2 instances).

    Interface (internal wires, not ports):
        {prefix}_in_i   : W-bit inputs  (i=0..N-1)
        {prefix}_out_i  : W-bit outputs (i=0..N-1)

    Selects are NOT ports here; they are mapped to top-level sel_{j}_{s}.
    """
    lines = []

    # Declare per-stage outputs
    for s in range(STAGES):
        for i in range(N):
            lines.append(f"    wire [{W-1}:0] {prefix}_stg{s}_out_{i};")
    lines.append("")

    # Build each stage
    for s in range(STAGES):
        lines.append(f"    // ---- {prefix}: Stage {s} ----")

        if s == 0:
            in_wires = [f"{prefix}_in_{i}" for i in range(N)]
        else:
            in_wires = [None] * N
            for i in range(N):
                dest = perfect_shuffle(i, STAGES)
                in_wires[dest] = f"{prefix}_stg{s-1}_out_{i}"

        for k in range(N // 2):
            wire_a = in_wires[2 * k]
            wire_b = in_wires[2 * k + 1]
            out_lo = f"{prefix}_stg{s}_out_{2 * k}"
            out_hi = f"{prefix}_stg{s}_out_{2 * k + 1}"

            sel_lo = sel_for_omega(prefix, s, k, "lo")
            sel_hi = sel_for_omega(prefix, s, k, "hi")

            lines.append(
                f"    mux2 u_{prefix}_s{s}_sw{k}_lo ("
                f".a({wire_a}), "
                f".b({wire_b}), "
                f".sel({sel_lo}), "
                f".out({out_lo}));"
            )
            lines.append(
                f"    mux2 u_{prefix}_s{s}_sw{k}_hi ("
                f".a({wire_a}), "
                f".b({wire_b}), "
                f".sel({sel_hi}), "
                f".out({out_hi}));"
            )

        lines.append("")

    # Output assignments to internal out wires
    lines.append(f"    // ---- {prefix}: Output assignments ----")
    last = STAGES - 1
    for i in range(N):
        lines.append(f"    assign {prefix}_out_{i} = {prefix}_stg{last}_out_{i};")
    lines.append("")

    return "\n".join(lines)


def gen_echo(N: int, W: int) -> str:
    """
    Generate an Echo netlist with two Omega MINs that share the same top-level pins.

    Top-level pins match crossbar:
      - data_in_0..N-1 (W-bit)
      - data_out_0..N-1 (W-bit)
      - sel_{j}_{s} for j in [0..N-1], s in [0..log2(N)-1]
    Plus one extra pin:
      - mode (0=ctrl network on the pins, 1=xfer network on the pins)

    Total mux2 instances inside the Omega fabrics: 2 * N * log2(N)
    Plus output selection muxes: N
    """
    STAGES = int(math.log2(N))
    assert 2**STAGES == N, f"N={N} must be a power of 2"

    lines = []
    lines.append(gen_mux2(W))

    # Module header
    lines.append("module echo (")

    ports = []
    for i in range(N):
        ports.append(f"    input  wire [{W-1}:0] data_in_{i}")
    for i in range(N):
        ports.append(f"    output wire [{W-1}:0] data_out_{i}")
    for j in range(N):
        for s in range(STAGES):
            ports.append(f"    input  wire sel_{j}_{s}")
    ports.append(f"    input  wire mode")

    lines.append(",\n".join(ports))
    lines.append(");")
    lines.append("")

    # Internal interfaces for both networks
    for i in range(N):
        lines.append(f"    wire [{W-1}:0] ctrl_in_{i};")
        lines.append(f"    wire [{W-1}:0] ctrl_out_{i};")
        lines.append(f"    wire [{W-1}:0] xfer_in_{i};")
        lines.append(f"    wire [{W-1}:0] xfer_out_{i};")

    lines.append("")
    lines.append("    // Shared inputs drive BOTH networks (outputs are selected by mode)")
    for i in range(N):
        lines.append(f"    assign ctrl_in_{i} = data_in_{i};")
        lines.append(f"    assign xfer_in_{i} = data_in_{i};")
    lines.append("")

    # Inline Omega networks
    lines.append("    // ========== Control Omega Network (mode=0 selects these outputs) ==========")
    lines.append(gen_omega_inline(N, W, STAGES, "ctrl"))

    lines.append("    // ========== Transfer Omega Network (mode=1 selects these outputs) ==========")
    lines.append(gen_omega_inline(N, W, STAGES, "xfer"))

    # Output selection muxes
    lines.append("    // ========== Output selection (shared pins) ==========")
    for i in range(N):
        # mode=0 -> ctrl_out, mode=1 -> xfer_out
        lines.append(
            f"    mux2 u_echo_outsel_{i} ("
            f".a(ctrl_out_{i}), "
            f".b(xfer_out_{i}), "
            f".sel(mode), "
            f".out(data_out_{i}));"
        )
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def gen_sdc(design_name: str) -> str:
    """Generate minimal SDC for a combinational design."""
    return f"""\
# Timing constraints for {design_name} (combinational)
create_clock -name virtual_clk -period 10.0
set_input_delay  -clock virtual_clk 0.0 [all_inputs]
set_output_delay -clock virtual_clk 0.0 [all_outputs]
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    N = int(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    STAGES = int(math.log2(N))
    if 2**STAGES != N:
        print(f"Error: N={N} is not a power of 2")
        sys.exit(1)

    print(f"Generating netlists for N={N}, W={W}")
    print(f"  Crossbar: {N} outputs x {N-1} mux2 each = {N*(N-1)} mux2 instances")
    print(f"  Echo:     2 x ({STAGES} stages x {N} mux2) + {N} outsel mux2 = {2*N*STAGES + N} mux2 instances")
    print(f"           (Echo shares crossbar-style pins, plus 'mode')")

    with open("crossbar.v", "w") as f:
        f.write(gen_crossbar(N, W))
    print("  Wrote crossbar.v")

    with open("echo.v", "w") as f:
        f.write(gen_echo(N, W))
    print("  Wrote echo.v")

    for name in ("crossbar", "echo"):
        with open(f"{name}.sdc", "w") as f:
            f.write(gen_sdc(name))
        print(f"  Wrote {name}.sdc")

    print("Done.")


if __name__ == "__main__":
    main()
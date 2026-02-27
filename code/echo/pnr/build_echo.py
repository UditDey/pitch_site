#!/usr/bin/env python3
"""
SiliconCompiler build script for Echo topology (2x Omega MINs, shared pins + mode).

Expects echo.v and echo.sdc in the working directory.
Run gen.py first to produce them.

Usage:
    python3 build_echo.py
    python3 build_echo.py --pin-constraints

Pin placement (when --pin-constraints is set):
    LEFT edge ("shared read-port interface"):
        - data_in:  shared inputs (drive ctrl_in and xfer_in internally)
        - data_out: shared outputs (muxed between ctrl_out and xfer_out via mode)
    mode: placed on BOTTOM edge
    Select lines: unconstrained
"""

import argparse
from siliconcompiler import ASIC, Design
from siliconcompiler.targets import skywater130_demo

# Must match the N, W used in gen.py
N = 32
W = 8


def main():
    parser = argparse.ArgumentParser(description="Build Echo topology with SiliconCompiler")
    parser.add_argument("--pin-constraints", action="store_true",
                        help="Constrain data pins to die edges for shared-pin Echo")
    args = parser.parse_args()

    design = Design("echo")
    design.set_topmodule("echo", fileset="rtl")
    design.add_file("echo.v", fileset="rtl")
    design.add_file("echo.sdc", fileset="sdc")

    project = ASIC(design)
    project.add_fileset(["rtl", "sdc"])
    skywater130_demo(project)

    # ----------------------------------------------------------------
    # Pin constraints
    #
    # Side encoding (clockwise from lower-left):
    #   1 = left, 2 = bottom, 3 = right, 4 = top
    #
    # Keypath: ('constraint', 'pin', <name>, 'side'|'order')
    # ----------------------------------------------------------------

    if args.pin_constraints:
        for i in range(N):
            for b in range(W):
                in_name = f"data_in_{i}[{b}]"
                project.set('constraint', 'pin', in_name, 'side', 1)  # left
                project.set('constraint', 'pin', in_name, 'order', i * W + b)

                out_name = f"data_out_{i}[{b}]"
                project.set('constraint', 'pin', out_name, 'side', 3)  # bottom
                project.set('constraint', 'pin', out_name, 'order', i * W + b)

        # Place mode deterministically (bottom, after the outputs)
        project.set('constraint', 'pin', "mode", 'side', 2)
        project.set('constraint', 'pin', "mode", 'order', N * W)

    project.run()
    project.summary()


if __name__ == "__main__":
    main()
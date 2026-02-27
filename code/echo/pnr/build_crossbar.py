#!/usr/bin/env python3
"""
SiliconCompiler build script for crossbar topology.

Expects crossbar.v and crossbar.sdc in the working directory.
Run gen.py first to produce them.

Usage:
    python3 build_crossbar.py
    python3 build_crossbar.py --pin-constraints

Pin placement (when --pin-constraints is set):
    - data inputs:  LEFT  edge (side 1)
    - data outputs: BOTTOM edge (side 2)
    - select lines: unconstrained
"""

import argparse
from siliconcompiler import ASIC, Design
from siliconcompiler.targets import skywater130_demo

# Must match the N, W used in gen.py
N = 32
W = 8


def main():
    parser = argparse.ArgumentParser(description="Build crossbar topology with SiliconCompiler")
    parser.add_argument("--pin-constraints", action="store_true",
                        help="Constrain data pins to die edges matching crossbar grid layout")
    args = parser.parse_args()

    design = Design("crossbar")
    design.set_topmodule("crossbar", fileset="rtl")
    design.add_file("crossbar.v", fileset="rtl")
    design.add_file("crossbar.sdc", fileset="sdc")

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
                project.set('constraint', 'pin', in_name, 'side', 1)
                project.set('constraint', 'pin', in_name, 'order', i * W + b)

                out_name = f"data_out_{i}[{b}]"
                project.set('constraint', 'pin', out_name, 'side', 2)
                project.set('constraint', 'pin', out_name, 'order', i * W + b)

    project.run()
    project.summary()


if __name__ == "__main__":
    main()

# Echo Cache for Wide Gather Ops

This directory contains the RTL model and testbench for the Echo Cache.

### `echo_cache.v`
The RTL model implementation with a simple storage array exposing word taps. Top module is:
```v
module echo_cache #(
    parameter N          = 8,
    parameter DATA_WIDTH = 16
) (
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        req_valid,
    output wire                        req_ready,
    input  wire [N*ADDR_WIDTH-1:0]     addrs,
    output wire                        resp_valid,
    output wire [N*DATA_WIDTH-1:0]     data
);
```
The valid/ready lines are to handle the backpressure from the retry logic.

### `tb_echo_cache.v`
Self-checking testbench for the Echo Cache. Tests the following scenarios:
1. Identity mapping (port `i` reads address `i`)
2. Reverse identity mapping (port `i` reads address `N-1-i`)
3. Broadcast
4. Conflict pattern (tests packet dropping and retry logic)
5. Back to back requests (tests backpressure)

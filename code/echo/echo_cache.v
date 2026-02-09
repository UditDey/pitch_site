`timescale 1ns/1ps

module storage_array #(
    parameter N          = 8,
    parameter DATA_WIDTH = 16
) (
    input  wire clk,
    input  wire rst,
    output wire [N*DATA_WIDTH-1:0] tap_ports
);
    reg [DATA_WIDTH-1:0] memory [N-1:0];

    // Tap ports to corresponding memory words
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : taps
            assign tap_ports[i*DATA_WIDTH +: DATA_WIDTH] = memory[i];
        end
    endgenerate

    // Initialize memory: max value - i
    integer j;
    always @(posedge clk) begin
        if (rst) begin
            for (j = 0; j < N; j = j + 1) begin
                memory[j] <= {DATA_WIDTH{1'b1}} - j;
            end
        end
    end
endmodule


module perfect_shuffle #(
    parameter N         = 8,
    parameter PKT_WIDTH = 4
) (
    input  wire [N*PKT_WIDTH-1:0] in_pkts,
    output wire [N*PKT_WIDTH-1:0] out_pkts
);
    localparam LOG_N = $clog2(N);

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : shuffle
            localparam integer j = ((i << 1) & (N - 1)) | (i >> (LOG_N - 1));

            assign out_pkts[j*PKT_WIDTH +: PKT_WIDTH] = in_pkts[i*PKT_WIDTH +: PKT_WIDTH];
        end
    endgenerate
endmodule


module inverse_perfect_shuffle #(
    parameter N         = 8,
    parameter PKT_WIDTH = 4
) (
    input  wire [N*PKT_WIDTH-1:0] in_pkts,
    output wire [N*PKT_WIDTH-1:0] out_pkts
);
    localparam LOG_N = $clog2(N);

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : unshuffle
            localparam integer j = (i >> 1) | ((i & 1) << (LOG_N - 1));

            assign out_pkts[j*PKT_WIDTH +: PKT_WIDTH] = in_pkts[i*PKT_WIDTH +: PKT_WIDTH];
        end
    endgenerate
endmodule


module ctrl_switch #(
    parameter ADDR_WIDTH = 3,
    parameter STAGE_BIT  = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire [ADDR_WIDTH:0] in0,
    input  wire [ADDR_WIDTH:0] in1,
    output reg  [ADDR_WIDTH:0] out0,
    output reg  [ADDR_WIDTH:0] out1,
    output reg                 did_swap,
    output reg                 did_merge
);
    // Control packet format: {valid, addr}
    // Switch states are defined by (did_swap, did_merge):
    // 1) (0, 0): Passthru
    // 2) (1, 0): Swap
    // 3) (0, 1): Merge left
    // 4) (1, 1): Merge Right
    localparam PKT_WIDTH = ADDR_WIDTH + 1;
    localparam [PKT_WIDTH-1:0] ZERO_PKT = {PKT_WIDTH{1'b0}};

    wire in0_valid       = in0[PKT_WIDTH-1];
    wire in0_wants_right = in0[STAGE_BIT];
    wire in1_valid       = in1[PKT_WIDTH-1];
    wire in1_wants_right = in1[STAGE_BIT];

    // Switch decision
    wire do_swap  = in0_valid ? in0_wants_right :
                    in1_valid ? ~in1_wants_right :
                    1'b0;

    wire do_merge = in0_valid && in1_valid && (in0[ADDR_WIDTH-1:0] == in1[ADDR_WIDTH-1:0]);

    // Output crossbar
    wire [ADDR_WIDTH:0] xbar0 = do_swap ? in1 : in0;
    wire [ADDR_WIDTH:0] xbar1 = do_swap ? in0 : in1;

    // Kill misrouted packet
    wire kill0 = xbar0[STAGE_BIT] == 1'b1;
    wire kill1 = xbar1[STAGE_BIT] == 1'b0;

    // Final ouput packets
    wire [ADDR_WIDTH:0] final_out0 = kill0 ? ZERO_PKT : xbar0;
    wire [ADDR_WIDTH:0] final_out1 = kill1 ? ZERO_PKT : xbar1;

    always @(posedge clk) begin
        if (rst) begin
            out0      <= ZERO_PKT;
            out1      <= ZERO_PKT;
            did_swap  <= 1'b0;
            did_merge <= 1'b0;
        end else begin
            out0      <= final_out0;
            out1      <= final_out1;
            did_swap  <= do_swap;
            did_merge <= do_merge;
        end
    end
endmodule


module xfer_switch #(
    parameter DATA_WIDTH = 8,
    parameter STAGE_BIT  = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire [DATA_WIDTH:0] in0,
    input  wire [DATA_WIDTH:0] in1,
    input  wire                did_swap,
    input  wire                did_merge,
    output reg  [DATA_WIDTH:0] out0,
    output reg  [DATA_WIDTH:0] out1
);
    // Transfer packet format: {valid, data}
    // ADDITIONAL CONSTRAINT: when valid == 0, then data == 0
    localparam PKT_WIDTH = DATA_WIDTH + 1;
    localparam [PKT_WIDTH-1:0] ZERO_PKT = {PKT_WIDTH{1'b0}};

    // Reverse the crossbar using stored swap decision
    wire [DATA_WIDTH:0] xbar0 = did_swap ? in1 : in0;
    wire [DATA_WIDTH:0] xbar1 = did_swap ? in0 : in1;
    
    // Merge case: Choose the valid packet to multicast
    // OR'ing works because if merge was done in control phase, then one input here must
    // necessarily have valid == 0 and addr == 0
    wire [DATA_WIDTH:0] merge_chosen = in0 | in1;
    
    wire [DATA_WIDTH:0] final_out0 = did_merge ? merge_chosen : xbar0;
    wire [DATA_WIDTH:0] final_out1 = did_merge ? merge_chosen : xbar1;

    always @(posedge clk) begin
        if (rst) begin
            out0 <= ZERO_PKT;
            out1 <= ZERO_PKT;
        end else begin
            out0 <= final_out0;
            out1 <= final_out1;
        end
    end
endmodule


module ctrl_layer #(
    parameter N          = 8,
    parameter ADDR_WIDTH = 3,
    parameter STAGE_BIT  = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire [N*(ADDR_WIDTH+1)-1:0] in_pkts,
    output wire [N*(ADDR_WIDTH+1)-1:0] out_pkts,
    output wire [(N/2)-1:0]            did_swaps,
    output wire [(N/2)-1:0]            did_merges
);
    localparam PKT_WIDTH = ADDR_WIDTH + 1;

    genvar i;
    generate
        wire [N*PKT_WIDTH-1:0] sh_pkts;

        perfect_shuffle #(
            .N(N),
            .PKT_WIDTH(PKT_WIDTH)
        ) shuffle (
            .in_pkts(in_pkts),
            .out_pkts(sh_pkts)
        );

        for (i = 0; i < N/2; i = i + 1) begin : sw        
            ctrl_switch #(
                .ADDR_WIDTH(ADDR_WIDTH),
                .STAGE_BIT(STAGE_BIT)
            ) switch (
                .clk(clk),
                .rst(rst),
                .in0 (sh_pkts [(2*i  ) * PKT_WIDTH +: PKT_WIDTH]),
                .in1 (sh_pkts [(2*i+1) * PKT_WIDTH +: PKT_WIDTH]),
                .out0(out_pkts[(2*i  ) * PKT_WIDTH +: PKT_WIDTH]),
                .out1(out_pkts[(2*i+1) * PKT_WIDTH +: PKT_WIDTH]),
                .did_swap(did_swaps[i]),
                .did_merge(did_merges[i])
            );
        end
    endgenerate
endmodule


module xfer_layer #(
    parameter N          = 8,
    parameter DATA_WIDTH = 3,
    parameter STAGE_BIT  = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire [N*(DATA_WIDTH+1)-1:0] in_pkts,
    input  wire [(N/2)-1:0]            did_swaps,
    input  wire [(N/2)-1:0]            did_merges,
    output wire [N*(DATA_WIDTH+1)-1:0] out_pkts
);
    localparam PKT_WIDTH = DATA_WIDTH + 1;

    genvar i;
    generate
        wire [N*PKT_WIDTH-1:0] tmp_pkts;

        for (i = 0; i < N/2; i = i + 1) begin : sw        
            xfer_switch #(
                .DATA_WIDTH(DATA_WIDTH),
                .STAGE_BIT(STAGE_BIT)
            ) switch (
                .clk(clk),
                .rst(rst),
                .in0 (in_pkts [(2*i  ) * PKT_WIDTH +: PKT_WIDTH]),
                .in1 (in_pkts [(2*i+1) * PKT_WIDTH +: PKT_WIDTH]),
                .did_swap(did_swaps[i]),
                .did_merge(did_merges[i]),
                .out0(tmp_pkts[(2*i  ) * PKT_WIDTH +: PKT_WIDTH]),
                .out1(tmp_pkts[(2*i+1) * PKT_WIDTH +: PKT_WIDTH])
            );
        end

        inverse_perfect_shuffle #(
            .N(N),
            .PKT_WIDTH(PKT_WIDTH)
        ) shuffle (
            .in_pkts(tmp_pkts),
            .out_pkts(out_pkts)
        );
    endgenerate
endmodule


module ctrl_network #(
    parameter N = 8
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire [N-1:0]           req_valids,
    input  wire [N*$clog2(N)-1:0] addrs,
    output wire [N*($clog2(N)+1)-1:0]     out_pkts,
    output wire [((N/2) * $clog2(N))-1:0] did_swaps,
    output wire [((N/2) * $clog2(N))-1:0] did_merges
);
    localparam ADDR_WIDTH = $clog2(N);
    localparam NUM_STAGES = $clog2(N);
    localparam PKT_WIDTH  = ADDR_WIDTH + 1;

    // Expand addresses into packets: {valid, addr}
    wire [N*PKT_WIDTH-1:0] in_pkts;
    genvar p;
    generate
        for (p = 0; p < N; p = p + 1) begin : expand
            // Use the external valid signal instead of hardcoded 1'b1
            assign in_pkts[p*PKT_WIDTH +: PKT_WIDTH] = 
                {req_valids[p], addrs[p*ADDR_WIDTH +: ADDR_WIDTH]};
        end
    endgenerate

    // Inter-stage packet wires
    wire [N*PKT_WIDTH-1:0] stage_pkts [0:NUM_STAGES];
    assign stage_pkts[0] = in_pkts;

    // Instantiate stages (MSB routed first)
    genvar s;
    generate
        for (s = 0; s < NUM_STAGES; s = s + 1) begin : stage
            ctrl_layer #(
                .N(N),
                .ADDR_WIDTH(ADDR_WIDTH),
                .STAGE_BIT(ADDR_WIDTH - 1 - s)
            ) layer (
                .clk(clk),
                .rst(rst),
                .in_pkts(stage_pkts[s]),
                .out_pkts(stage_pkts[s+1]),
                .did_swaps(did_swaps[s*(N/2) +: (N/2)]),
                .did_merges(did_merges[s*(N/2) +: (N/2)])
            );
        end
    endgenerate

    assign out_pkts = stage_pkts[NUM_STAGES];
endmodule


module xfer_network #(
    parameter N          = 8,
    parameter DATA_WIDTH = 16
) (
    input  wire                           clk,
    input  wire                           rst,
    input  wire [N*DATA_WIDTH-1:0]        tap_ports,
    input  wire [N*($clog2(N)+1)-1:0]     ctrl_out_pkts,
    input  wire [((N/2) * $clog2(N))-1:0] did_swaps,
    input  wire [((N/2) * $clog2(N))-1:0] did_merges,
    output wire [N*(DATA_WIDTH+1)-1:0]    out_pkts
);
    localparam ADDR_WIDTH  = $clog2(N);
    localparam NUM_STAGES  = $clog2(N);
    localparam CTRL_PKT_WIDTH = ADDR_WIDTH + 1;
    localparam PKT_WIDTH   = DATA_WIDTH + 1;

    // Expand tap_ports into packets, using ctrl_out valid bits
    wire [N*PKT_WIDTH-1:0] in_pkts;
    genvar p;
    generate
        for (p = 0; p < N; p = p + 1) begin : expand
            wire ctrl_valid = ctrl_out_pkts[(p+1)*CTRL_PKT_WIDTH - 1];
            assign in_pkts[p*PKT_WIDTH +: PKT_WIDTH] = 
                ctrl_valid ? {1'b1, tap_ports[p*DATA_WIDTH +: DATA_WIDTH]}
                           : {PKT_WIDTH{1'b0}};
        end
    endgenerate

    // Inter-stage packet wires
    wire [N*PKT_WIDTH-1:0] stage_pkts [0:NUM_STAGES];
    assign stage_pkts[0] = in_pkts;

    // Instantiate stages in reverse order
    genvar s;
    generate
        for (s = 0; s < NUM_STAGES; s = s + 1) begin : stage
            localparam CTRL_STAGE = NUM_STAGES - 1 - s;
            
            xfer_layer #(
                .N(N),
                .DATA_WIDTH(DATA_WIDTH),
                .STAGE_BIT(s)
            ) layer (
                .clk(clk),
                .rst(rst),
                .in_pkts(stage_pkts[s]),
                .out_pkts(stage_pkts[s+1]),
                .did_swaps(did_swaps[CTRL_STAGE*(N/2) +: (N/2)]),
                .did_merges(did_merges[CTRL_STAGE*(N/2) +: (N/2)])
            );
        end
    endgenerate

    assign out_pkts = stage_pkts[NUM_STAGES];
endmodule


module retry_logic #(
    parameter N          = 8,
    parameter DATA_WIDTH = 16
) (
    input  wire clk,
    input  wire rst,

    // External request interface
    input  wire                        req_valid,
    output wire                        req_ready,
    input  wire [N*ADDR_WIDTH-1:0]     addrs_in,

    // External response interface
    output wire                        resp_valid,
    output wire [N*DATA_WIDTH-1:0]     data_out,

    // To ctrl_network
    output wire [N*ADDR_WIDTH-1:0]     ctrl_addrs,
    output wire [N-1:0]                ctrl_valids, // <--- NEW OUTPUT
    output wire                        ctrl_issue,

    // From xfer_network
    input  wire [N*DATA_WIDTH-1:0]     xfer_data,
    input  wire [N-1:0]                xfer_valid
);
    localparam ADDR_WIDTH = $clog2(N);
    // Latency correction: Pipeline depth is 6, wait 5 cycles (cnt 0..4) then capture
    localparam LATENCY    = 2 * $clog2(N); 

    // State machine
    localparam STATE_IDLE    = 2'd0;
    localparam STATE_ISSUE   = 2'd1;
    localparam STATE_WAIT    = 2'd2;
    localparam STATE_CAPTURE = 2'd3;

    reg [1:0] state, state_next;
    reg [$clog2(LATENCY+1)-1:0] lat_cnt;
    reg [ADDR_WIDTH-1:0] stored_addrs [N-1:0];
    reg [N-1:0]          pending;
    reg [DATA_WIDTH-1:0] captured_data [N-1:0];

    assign req_ready  = (state == STATE_IDLE);
    assign resp_valid = (state == STATE_CAPTURE) && (pending == 0);
    assign ctrl_issue = (state == STATE_ISSUE);

    // Output valids to network: Only pending requests are valid
    assign ctrl_valids = pending; 

    // Flatten captured_data
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : out_flat
            assign data_out[g*DATA_WIDTH +: DATA_WIDTH] = captured_data[g];
        end
    endgenerate

    // Generate ctrl_addrs
    generate
        for (g = 0; g < N; g = g + 1) begin : addr_gen
            assign ctrl_addrs[g*ADDR_WIDTH +: ADDR_WIDTH] = 
                pending[g] ? stored_addrs[g] : {ADDR_WIDTH{1'b0}};
        end
    endgenerate

    // State machine
    always @(*) begin
        state_next = state;
        case (state)
            STATE_IDLE: begin
                if (req_valid) state_next = STATE_ISSUE;
            end
            STATE_ISSUE: begin
                state_next = STATE_WAIT;
            end
            STATE_WAIT: begin
                // FIX: -2 to transition into CAPTURE exactly when data arrives
                if (lat_cnt == LATENCY - 2) 
                    state_next = STATE_CAPTURE;
            end
            STATE_CAPTURE: begin
                if (pending == 0) state_next = STATE_IDLE;
                else              state_next = STATE_ISSUE;
            end
        endcase
    end

    // Sequential logic
    always @(posedge clk) begin
        if (rst) state <= STATE_IDLE;
        else     state <= state_next;
    end

    // Latency counter
    always @(posedge clk) begin
        if (rst) lat_cnt <= 0;
        else if (state == STATE_ISSUE) lat_cnt <= 0;
        else if (state == STATE_WAIT)  lat_cnt <= lat_cnt + 1;
    end

    // Data capture logic
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            pending <= 0;
            for (i=0; i<N; i=i+1) begin
                stored_addrs[i] <= 0;
                captured_data[i] <= 0;
            end
        end else begin
            case (state)
                STATE_IDLE: begin
                    if (req_valid) begin
                        pending <= {N{1'b1}};
                        for (i=0; i<N; i=i+1) begin
                            stored_addrs[i] <= addrs_in[i*ADDR_WIDTH +: ADDR_WIDTH];
                            captured_data[i] <= 0;
                        end
                    end
                end
                STATE_CAPTURE: begin
                    for (i=0; i<N; i=i+1) begin
                        if (xfer_valid[i] && pending[i]) begin
                            captured_data[i] <= xfer_data[i*DATA_WIDTH +: DATA_WIDTH];
                            pending[i] <= 1'b0;
                        end
                    end
                end
            endcase
        end
    end
endmodule


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
    localparam ADDR_WIDTH     = $clog2(N);
    localparam NUM_STAGES     = $clog2(N);
    localparam CTRL_PKT_WIDTH = ADDR_WIDTH + 1;
    localparam XFER_PKT_WIDTH = DATA_WIDTH + 1;

    wire [N*ADDR_WIDTH-1:0]      ctrl_addrs;
    wire [N-1:0]                 ctrl_valids; // <--- WIRE
    wire                         ctrl_issue;
    
    wire [((N/2) * NUM_STAGES)-1:0] did_swaps;
    wire [((N/2) * NUM_STAGES)-1:0] did_merges;
    wire [N*CTRL_PKT_WIDTH-1:0]  ctrl_out_pkts;
    wire [N*DATA_WIDTH-1:0]      tap_ports;
    wire [N*XFER_PKT_WIDTH-1:0]  xfer_out_pkts;
    wire [N*DATA_WIDTH-1:0]      xfer_data;
    wire [N-1:0]                 xfer_valid;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : extract
            assign xfer_data[i*DATA_WIDTH +: DATA_WIDTH] = 
                xfer_out_pkts[i*XFER_PKT_WIDTH +: DATA_WIDTH];
            assign xfer_valid[i] = xfer_out_pkts[(i+1)*XFER_PKT_WIDTH - 1];
        end
    endgenerate

    retry_logic #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH)
    ) retry (
        .clk(clk),
        .rst(rst),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .addrs_in(addrs),
        .resp_valid(resp_valid),
        .data_out(data),
        .ctrl_addrs(ctrl_addrs),
        .ctrl_valids(ctrl_valids), // <--- CONNECTED
        .ctrl_issue(ctrl_issue),
        .xfer_data(xfer_data),
        .xfer_valid(xfer_valid)
    );

    ctrl_network #(
        .N(N)
    ) ctrl (
        .clk(clk),
        .rst(rst),
        .req_valids(ctrl_valids),  // <--- CONNECTED
        .addrs(ctrl_addrs),
        .out_pkts(ctrl_out_pkts),
        .did_swaps(did_swaps),
        .did_merges(did_merges)
    );

    storage_array #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH)
    ) storage (
        .clk(clk),
        .rst(rst),
        .tap_ports(tap_ports)
    );

    xfer_network #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH)
    ) xfer (
        .clk(clk),
        .rst(rst),
        .tap_ports(tap_ports),
        .ctrl_out_pkts(ctrl_out_pkts),
        .did_swaps(did_swaps),
        .did_merges(did_merges),
        .out_pkts(xfer_out_pkts)
    );
endmodule

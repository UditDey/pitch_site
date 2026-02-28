`timescale 1ns/1ps
`include "echo_cache.v"

module tb_echo_cache_with_retry;
    parameter N          = 8;
    parameter DATA_WIDTH = 16;
    localparam ADDR_WIDTH = $clog2(N);

    reg  clk = 0;
    reg  rst;
    
    // Request interface
    reg                        req_valid;
    wire                       req_ready;
    reg  [N*ADDR_WIDTH-1:0]    addrs;
    
    // Response interface
    wire                       resp_valid;
    wire [N*DATA_WIDTH-1:0]    data;

    // DUT
    echo_cache #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .addrs(addrs),
        .resp_valid(resp_valid),
        .data(data)
    );

    always #5 clk = ~clk;

    // =========================================================================
    // Test infrastructure
    // =========================================================================
    integer i;
    integer cycle_count;
    integer test_num;
    integer errors;
    integer total_errors;

    // Expected data: storage[addr] = 0xFFFF - addr
    function [DATA_WIDTH-1:0] expected_data;
        input [ADDR_WIDTH-1:0] addr;
        begin
            expected_data = {DATA_WIDTH{1'b1}} - addr;
        end
    endfunction

    task reset_dut;
        begin
            rst = 1;
            req_valid = 0;
            addrs = 0;
            repeat(4) @(posedge clk);
            rst = 0;
            repeat(4) @(posedge clk);  // Let storage initialize
        end
    endtask

    task submit_and_wait;
        input [N*ADDR_WIDTH-1:0] a;
        begin
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            addrs = a;
            req_valid = 1;
            @(posedge clk);
            req_valid = 0;
            cycle_count = 0;
            while (!resp_valid) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
                if (cycle_count > 500) begin
                    $display("ERROR: Timeout waiting for resp_valid");
                    $finish;
                end
            end
        end
    endtask

    task check_port;
        input integer port;
        input [ADDR_WIDTH-1:0] addr;
        begin
            if (data[port*DATA_WIDTH +: DATA_WIDTH] !== expected_data(addr)) begin
                $display("  ERROR port %0d: addr=%0d expected=%h got=%h", 
                    port, addr, expected_data(addr), data[port*DATA_WIDTH +: DATA_WIDTH]);
                errors = errors + 1;
            end
        end
    endtask

    task print_results;
        begin
            $display("  Results (completed in %0d cycles):", cycle_count);
            for (i = 0; i < N; i = i + 1) begin
                $display("    port %0d: data=%h", i, data[i*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    endtask

    // =========================================================================
    // Test cases
    // =========================================================================
    initial begin
        total_errors = 0;

        // ==== Test 1: Identity mapping (port i reads addr i) ====
        test_num = 1;
        errors = 0;
        $display("\n=== Test %0d: Identity (port i reads addr i) ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = i;
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, i);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 2: Broadcast addr 0 (all ports read same address) ====
        test_num = 2;
        errors = 0;
        $display("\n=== Test %0d: Broadcast addr 0 ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = 0;
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, 0);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 3: Broadcast addr 7 ====
        test_num = 3;
        errors = 0;
        $display("\n=== Test %0d: Broadcast addr 7 ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = N - 1;
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, N - 1);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 4: Reversal (port i reads addr N-1-i) ====
        test_num = 4;
        errors = 0;
        $display("\n=== Test %0d: Reversal (port i reads addr N-1-i) ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = N - 1 - i;
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, N - 1 - i);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 5: Conflict pattern (ports paired after shuffle want same direction) ====
        // This pattern caused blocking in the original echo_cache without retry
        test_num = 5;
        errors = 0;
        $display("\n=== Test %0d: Conflict pattern (requires retries) ===", test_num);
        reset_dut();
        
        // After shuffle: switch pairs are (0,4), (1,5), (2,6), (3,7)
        // Both ports in each pair request addresses with same MSB -> blocking
        addrs[0*ADDR_WIDTH +: ADDR_WIDTH] = 3'b100;  // port 0 wants addr 4
        addrs[1*ADDR_WIDTH +: ADDR_WIDTH] = 3'b000;  // port 1 wants addr 0
        addrs[2*ADDR_WIDTH +: ADDR_WIDTH] = 3'b101;  // port 2 wants addr 5
        addrs[3*ADDR_WIDTH +: ADDR_WIDTH] = 3'b001;  // port 3 wants addr 1
        addrs[4*ADDR_WIDTH +: ADDR_WIDTH] = 3'b110;  // port 4 wants addr 6
        addrs[5*ADDR_WIDTH +: ADDR_WIDTH] = 3'b010;  // port 5 wants addr 2
        addrs[6*ADDR_WIDTH +: ADDR_WIDTH] = 3'b111;  // port 6 wants addr 7
        addrs[7*ADDR_WIDTH +: ADDR_WIDTH] = 3'b011;  // port 7 wants addr 3
        
        submit_and_wait(addrs);
        print_results();
        
        check_port(0, 4);
        check_port(1, 0);
        check_port(2, 5);
        check_port(3, 1);
        check_port(4, 6);
        check_port(5, 2);
        check_port(6, 7);
        check_port(7, 3);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 6: All ports read addr 3 (middle address) ====
        test_num = 6;
        errors = 0;
        $display("\n=== Test %0d: All read addr 3 ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = 3;
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, 3);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 7: Pairs pattern (adjacent ports read same address) ====
        test_num = 7;
        errors = 0;
        $display("\n=== Test %0d: Pairs (ports 0,1->addr0; 2,3->addr2; etc) ===", test_num);
        reset_dut();
        
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = (i / 2) * 2;  // 0,0,2,2,4,4,6,6
        
        submit_and_wait(addrs);
        print_results();
        
        for (i = 0; i < N; i = i + 1)
            check_port(i, (i / 2) * 2);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 8: Back-to-back requests ====
        test_num = 8;
        errors = 0;
        $display("\n=== Test %0d: Back-to-back requests ===", test_num);
        reset_dut();
        
        // Request 1: identity
        $display("  Request 1: identity");
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = i;
        submit_and_wait(addrs);
        $display("    Completed in %0d cycles", cycle_count);
        for (i = 0; i < N; i = i + 1)
            check_port(i, i);
        
        // Request 2: reversal (immediately after)
        $display("  Request 2: reversal");
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = N - 1 - i;
        submit_and_wait(addrs);
        $display("    Completed in %0d cycles", cycle_count);
        for (i = 0; i < N; i = i + 1)
            check_port(i, N - 1 - i);
        
        // Request 3: broadcast
        $display("  Request 3: broadcast addr 5");
        for (i = 0; i < N; i = i + 1)
            addrs[i*ADDR_WIDTH +: ADDR_WIDTH] = 5;
        submit_and_wait(addrs);
        $display("    Completed in %0d cycles", cycle_count);
        for (i = 0; i < N; i = i + 1)
            check_port(i, 5);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Test 9: Worst-case conflict (all want high addresses from low ports) ====
        test_num = 9;
        errors = 0;
        $display("\n=== Test %0d: Worst-case conflict pattern ===", test_num);
        reset_dut();
        
        // Lower ports want high addresses, upper ports want low addresses
        // This should maximize blocking
        addrs[0*ADDR_WIDTH +: ADDR_WIDTH] = 7;
        addrs[1*ADDR_WIDTH +: ADDR_WIDTH] = 6;
        addrs[2*ADDR_WIDTH +: ADDR_WIDTH] = 5;
        addrs[3*ADDR_WIDTH +: ADDR_WIDTH] = 4;
        addrs[4*ADDR_WIDTH +: ADDR_WIDTH] = 3;
        addrs[5*ADDR_WIDTH +: ADDR_WIDTH] = 2;
        addrs[6*ADDR_WIDTH +: ADDR_WIDTH] = 1;
        addrs[7*ADDR_WIDTH +: ADDR_WIDTH] = 0;
        
        submit_and_wait(addrs);
        print_results();
        
        check_port(0, 7);
        check_port(1, 6);
        check_port(2, 5);
        check_port(3, 4);
        check_port(4, 3);
        check_port(5, 2);
        check_port(6, 1);
        check_port(7, 0);
        
        if (errors == 0) $display("  PASSED");
        else total_errors = total_errors + errors;

        // ==== Summary ====
        $display("\n========================================");
        if (total_errors == 0)
            $display("All %0d tests PASSED", test_num);
        else
            $display("FAILED: %0d total errors across %0d tests", total_errors, test_num);
        $display("========================================\n");
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Global timeout");
        $finish;
    end

endmodule
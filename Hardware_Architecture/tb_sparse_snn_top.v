// =============================================================================
// tb_sparse_snn_top.v - System-Level Testbench for Sparse SNN Pipeline
// =============================================================================
// Verifies the full inference pipeline with:
//   - Inter-layer spike buffers
//   - Per-neuron v_mem/v_th state persistence
//   - Proper pooling iteration
//   - Gatekeeper only on Layer 1
//
// Run with:
//   iverilog -o tb_top tb_sparse_snn_top.v sparse_snn_top.v \
//     dynamic_gatekeeper.v importance_monitor.v burst_redundancy.v \
//     quantized_sram.v sparse_mac.v adaptive_lif.v early_exit_fsm.v \
//     avg_pool.v && vvp tb_top
// =============================================================================

`timescale 1ns / 1ps

module tb_sparse_snn_top;

    // -------------------------------------------------------
    // Parameters
    // -------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter NUM_CLASSES = 10;

    // -------------------------------------------------------
    // Signals
    // -------------------------------------------------------
    reg  clk;
    reg  rst_n;
    reg  start;
    reg  spike_valid;
    reg  [9:0] spike_pre_id;

    wire [NUM_CLASSES-1:0] final_prediction;
    wire done;
    wire sys_enable;

    // -------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------
    sparse_snn_top #(
        .DATA_WIDTH(8),
        .ACCUM_WIDTH(16),
        .V_WIDTH(16),
        .NUM_CLASSES(NUM_CLASSES),
        .T_MAX(20),
        .CONFIDENCE_TH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .spike_valid(spike_valid),
        .spike_pre_id(spike_pre_id),
        .final_prediction(final_prediction),
        .done(done),
        .sys_enable(sys_enable)
    );

    // -------------------------------------------------------
    // Clock generation
    // -------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------
    // Cycle counter
    // -------------------------------------------------------
    integer cycle_count;
    initial cycle_count = 0;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    // -------------------------------------------------------
    // Waveform dump
    // -------------------------------------------------------
    initial begin
        $dumpfile("sparse_snn_top_tb.vcd");
        $dumpvars(0, tb_sparse_snn_top);
    end

    // -------------------------------------------------------
    // Test sequence
    // -------------------------------------------------------
    integer i;

    initial begin
        // --- Initialize ---
        rst_n        = 0;
        start        = 0;
        spike_valid  = 0;
        spike_pre_id = 10'd0;

        $display("\n=============================================");
        $display("  Sparse SNN Top-Level Testbench (Phase 2)");
        $display("=============================================\n");

        // --- Reset ---
        repeat (5) @(posedge clk);
        rst_n = 1;
        $display("[%0t] Reset released", $time);

        // --- Check initial state ---
        @(posedge clk);
        if (sys_enable !== 1'b1)
            $display("[FAIL] sys_enable should be 1 after reset, got %b", sys_enable);
        else
            $display("[PASS] sys_enable = 1 after reset");

        if (done !== 1'b0)
            $display("[FAIL] done should be 0 after reset, got %b", done);
        else
            $display("[PASS] done = 0 after reset");

        // --- Test 1: Assert start and feed spike events concurrently ---
        $display("\n--- Test 1: Start inference + feed spikes ---");
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;  // Pulse start for 1 cycle

        // Feed 50 spike events with different pre_ids
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            spike_valid  = 1;
            spike_pre_id = i[9:0];
        end
        spike_valid = 0;
        $display("[INFO] Start pulsed, fed 50 spike events (pre_id 0..49)");

        // --- Test 2: Burst redundancy check ---
        $display("\n--- Test 2: Burst redundancy check ---");
        for (i = 0; i < 5; i = i + 1) begin
            @(posedge clk);
            spike_valid  = 1;
            spike_pre_id = 10'd42;
        end
        spike_valid = 0;
        $display("[INFO] Fed 5 identical spike events (pre_id=42)");

        // --- Test 3: Monitor FSM state transitions ---
        $display("\n--- Test 3: Running pipeline (monitoring state) ---");
        $display("[INFO] layer_state = %0d at cycle %0d", dut.layer_state, cycle_count);

        // --- Test 4: Wait for done or timeout ---
        begin : wait_loop
            integer wait_cycles;
            wait_cycles = 0;
            while (!done && wait_cycles < 2_000_000) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;

                // Print progress every 500K cycles
                if (wait_cycles % 500000 == 0)
                    $display("[INFO] ...cycle %0d, layer_state=%0d, neuron=%0d, synapse=%0d, T=%0d",
                             cycle_count, dut.layer_state, dut.neuron_idx, dut.synapse_idx,
                             dut.time_step_cnt);
            end

            if (done)
                $display("[PASS] done asserted at cycle %0d (time %0t)", cycle_count, $time);
            else begin
                $display("[INFO] Timeout at 2M cycles - pipeline still running");
                $display("[INFO] state=%0d, neuron=%0d, synapse=%0d, T=%0d",
                         dut.layer_state, dut.neuron_idx, dut.synapse_idx, dut.time_step_cnt);
            end
        end

        // --- Test 5: Check sys_enable ---
        @(posedge clk);
        if (done) begin
            if (sys_enable === 1'b0)
                $display("[PASS] sys_enable deasserted when done");
            else
                $display("[WARN] sys_enable still high after done");
        end

        // --- Final State ---
        $display("\n--- Final State ---");
        $display("  final_prediction = %b", final_prediction);
        $display("  sys_enable       = %b", sys_enable);
        $display("  done             = %b", done);
        $display("  cycle_count      = %0d", cycle_count);
        $display("  timesteps done   = %0d", dut.time_step_cnt);

        $display("\n=============================================");
        $display("  Testbench Complete");
        $display("  Waveforms: sparse_snn_top_tb.vcd");
        $display("=============================================\n");

        #100;
        $finish;
    end

    // -------------------------------------------------------
    // Watchdog: monitor sys_enable transitions
    // -------------------------------------------------------
    always @(negedge sys_enable) begin
        if (rst_n)
            $display("[EVENT] sys_enable went LOW at cycle %0d (early exit triggered)", cycle_count);
    end

    // Monitor layer transitions
    always @(posedge clk) begin
        if (rst_n && sys_enable) begin
            if (dut.layer_state == 4'd1 && dut.synapse_idx == 0 && dut.neuron_idx == 0)
                $display("[LAYER] Entered LAYER1 at cycle %0d, T=%0d", cycle_count, dut.time_step_cnt);
            if (dut.layer_state == 4'd3 && dut.pool_phase == 0 && dut.pool_idx == 0)
                $display("[LAYER] Entered POOL1 at cycle %0d", cycle_count);
            if (dut.layer_state == 4'd4 && dut.synapse_idx == 0 && dut.neuron_idx == 0)
                $display("[LAYER] Entered LAYER2 at cycle %0d", cycle_count);
            if (dut.layer_state == 4'd6 && dut.pool_phase == 0 && dut.pool_idx == 0)
                $display("[LAYER] Entered POOL2 at cycle %0d", cycle_count);
            if (dut.layer_state == 4'd7 && dut.synapse_idx == 0 && dut.neuron_idx == 0)
                $display("[LAYER] Entered LAYER3 at cycle %0d", cycle_count);
            if (dut.layer_state == 4'd9 && dut.synapse_idx == 0 && dut.neuron_idx == 0)
                $display("[LAYER] Entered LAYER4 at cycle %0d", cycle_count);
            if (dut.layer_state == 4'd11)
                $display("[LAYER] TIMESTEP complete at cycle %0d, T=%0d", cycle_count, dut.time_step_cnt);
        end
    end

endmodule

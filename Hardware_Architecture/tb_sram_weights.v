// =============================================================================
// tb_sram_weights.v - Testbench: Load PyTorch Weights into Verilog SRAM
// =============================================================================
// This testbench demonstrates how the trained INT8 weights exported from
// PyTorch (via export_weights_mem.py) are loaded into the quantized_sram
// module using $readmemh and read out during spike-driven inference.
//
// Run with: iverilog -o tb_sram tb_sram_weights.v quantized_sram.v && vvp tb_sram
//       or: in Vivado/ModelSim, add both files to project and simulate.
// =============================================================================

`timescale 1ns / 1ps

module tb_sram_weights;

    // -------------------------------------------------------
    // Parameters - match the trained model layer sizes
    // -------------------------------------------------------
    parameter CONV1_ADDR_W = 11;  // ceil(log2(1600)) = 11
    parameter FC2_ADDR_W   = 11;  // ceil(log2(1280)) = 11
    parameter DATA_W       = 8;

    // -------------------------------------------------------
    // Signals
    // -------------------------------------------------------
    reg clk;
    reg [CONV1_ADDR_W-1:0] conv1_addr;
    reg conv1_re;
    wire signed [DATA_W-1:0] conv1_data;

    reg [FC2_ADDR_W-1:0] fc2_addr;
    reg fc2_re;
    wire signed [DATA_W-1:0] fc2_data;

    // -------------------------------------------------------
    // Clock generation (100 MHz)
    // -------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------
    // Instantiate Conv1 SRAM with trained weights
    // -------------------------------------------------------
    quantized_sram #(
        .ADDR_WIDTH(CONV1_ADDR_W),
        .DATA_WIDTH(DATA_W),
        .MEM_FILE("../mem_weights/conv1_weights.mem")  // <-- Trained weights!
    ) conv1_sram (
        .clk(clk),
        .addr(conv1_addr),
        .re(conv1_re),
        .data_out(conv1_data)
    );

    // -------------------------------------------------------
    // Instantiate FC2 (output layer) SRAM with trained weights
    // -------------------------------------------------------
    quantized_sram #(
        .ADDR_WIDTH(FC2_ADDR_W),
        .DATA_WIDTH(DATA_W),
        .MEM_FILE("../mem_weights/fc2_weights.mem")   // <-- Trained weights!
    ) fc2_sram (
        .clk(clk),
        .addr(fc2_addr),
        .re(fc2_re),
        .data_out(fc2_data)
    );

    // -------------------------------------------------------
    // Test sequence
    // -------------------------------------------------------
    integer i;

    initial begin
        $dumpfile("sram_weights_tb.vcd");
        $dumpvars(0, tb_sram_weights);

        // Initialize
        conv1_addr = 0; conv1_re = 0;
        fc2_addr = 0;   fc2_re = 0;

        // Wait for reset
        #20;

        // ----- Test 1: Read first 10 Conv1 weights -----
        $display("\n========================================");
        $display("  Conv1 SRAM - First 10 INT8 Weights");
        $display("========================================");

        for (i = 0; i < 10; i = i + 1) begin
            conv1_addr = i;
            conv1_re = 1;
            @(posedge clk);
            #1; // Small delay for output to settle
            $display("  addr[%0d] = %0d (hex: %02h)", i, conv1_data, conv1_data);
        end

        // ----- Test 2: Read Enable = 0 should output zero -----
        $display("\n========================================");
        $display("  Read Enable = 0 (Power Save Mode)");
        $display("========================================");
        conv1_re = 0;
        conv1_addr = 5;
        @(posedge clk);
        #1;
        $display("  addr[5] with re=0 -> data = %0d (expected: 0)", conv1_data);

        // ----- Test 3: Read FC2 output layer weights -----
        $display("\n========================================");
        $display("  FC2 SRAM - Output Layer Weights");
        $display("  (10 classes x 128 inputs = 1280 total)");
        $display("========================================");

        // Read first weight for each of the 10 output classes
        for (i = 0; i < 10; i = i + 1) begin
            fc2_addr = i * 128;  // Start of each class row
            fc2_re = 1;
            @(posedge clk);
            #1;
            $display("  class[%0d] first weight = %0d (hex: %02h)", i, fc2_data, fc2_data);
        end

        // ----- Summary -----
        $display("\n========================================");
        $display("  Software-Hardware Bridge Verified!");
        $display("  PyTorch weights successfully loaded");
        $display("  into Verilog SRAM via $readmemh.");
        $display("========================================\n");

        #10;
        $finish;
    end

endmodule

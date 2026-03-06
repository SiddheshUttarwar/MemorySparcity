module quantized_sram #(
    parameter ADDR_WIDTH = 10,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire re, // Read enable (Triggered by Spike_In)
    output reg signed [DATA_WIDTH-1:0] data_out
);
    // Standard SRAM memory array holding INT8 Quantized weights
    reg signed [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    always @(posedge clk) begin
        // POWER SAVING: Read Enable is uniquely tied to incoming Spikes!
        // If re == 0, the SRAM matrix is not activated, saving dynamic memory power
        if (re) begin
            data_out <= mem[addr];
        end else begin
            // Latched value or zero to prevent MAC accumulation propagation
            data_out <= {DATA_WIDTH{1'b0}}; 
        end
    end

    // Note: Weights are loaded externally or through $readmemh during synthesis/FPGA load
endmodule

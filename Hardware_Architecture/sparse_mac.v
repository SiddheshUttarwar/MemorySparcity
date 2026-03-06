module sparse_mac #(
    parameter DATA_WIDTH = 8,
    parameter ACCUM_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire sys_en, // Top-level enable (halts completely on Early Exit)
    input wire spike_in, // Input spike representing binary '1'
    input wire signed [DATA_WIDTH-1:0] weight_in,
    
    // We send read_req directly back to the SRAM core to awaken it
    output reg read_req,
    output reg signed [ACCUM_WIDTH-1:0] current_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_out <= {ACCUM_WIDTH{1'b0}};
            read_req <= 1'b0;
        end else if (sys_en) begin
            // Awake the SRAM solely if we have an incoming spike this cycle!
            read_req <= spike_in;
            
            // Wait for SRAM to serve weight, then accumulate. 
            // Because SNN inputs are purely binary {0, 1}, we do not need physical 
            // DSP multipliers! We just ADDer the weight if spike_in was true.
            if (spike_in) begin
                current_out <= current_out + weight_in;
            end
        end
    end
endmodule

// =============================================================================
// avg_pool.v - 2x2 Average Pooling Unit (stride 2)
// =============================================================================
// Matches Python AvgPool2d(kernel_size=2, stride=2).
// Accepts 4 input values sequentially (2x2 window), outputs their average.
// On binary spike data, average = sum / 4, implemented as sum >> 2.
// =============================================================================

module avg_pool #(
    parameter DATA_WIDTH = 16   // Width of input values (spike counts or membrane currents)
)(
    input wire clk,
    input wire rst_n,
    input wire sys_en,
    input wire  signed [DATA_WIDTH-1:0] data_in,
    input wire  data_valid,      // Asserted when data_in is valid
    output reg  signed [DATA_WIDTH-1:0] pool_out,
    output reg  pool_valid       // Asserted for 1 cycle when pool_out is ready
);

    // Internal accumulator (2 extra bits to hold sum of 4 values without overflow)
    reg signed [DATA_WIDTH+1:0] accum;
    reg [1:0] count;  // Counts 0,1,2,3 inputs within the 2x2 window

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum      <= {(DATA_WIDTH+2){1'b0}};
            count      <= 2'd0;
            pool_out   <= {DATA_WIDTH{1'b0}};
            pool_valid <= 1'b0;
        end else if (sys_en && data_valid) begin
            if (count == 2'd3) begin
                // Fourth input: compute average and output
                // sum >> 2  =  sum / 4  (arithmetic right shift for signed values)
                pool_out   <= (accum + data_in) >>> 2;
                pool_valid <= 1'b1;
                accum      <= {(DATA_WIDTH+2){1'b0}};
                count      <= 2'd0;
            end else begin
                // Accumulate inputs 0, 1, 2
                accum      <= accum + data_in;
                count      <= count + 1;
                pool_valid <= 1'b0;
            end
        end else begin
            pool_valid <= 1'b0;
        end
    end
endmodule

module burst_redundancy #(
    parameter ID_W = 10,
    parameter K_MAX = 1
)(
    input wire clk,
    input wire rst_n,
    input wire spike_valid,
    input wire [ID_W-1:0] pre_id,
    output reg corr_keep
);
    reg [ID_W-1:0] last_pre_id;
    reg [2:0] repeat_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_pre_id <= {ID_W{1'b0}};
            repeat_count <= 3'd0;
        end else if (spike_valid) begin
            if (pre_id == last_pre_id) begin
                if (repeat_count < 3'd7) repeat_count <= repeat_count + 1;
            end else begin
                last_pre_id <= pre_id;
                repeat_count <= 3'd1;
            end
        end
    end

    // Combinational evaluation for this cycle
    always @(*) begin
        if (spike_valid && pre_id == last_pre_id && repeat_count >= K_MAX) begin
            corr_keep = 1'b0; // Redundant spike! Drop it.
        end else begin
            corr_keep = 1'b1; // Keep
        end
    end
endmodule

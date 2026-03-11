module burst_redundancy #(
    parameter ID_W = 10,
    parameter NUM_PRE = 1024,
    parameter K_MAX = 1
)(
    input wire clk,
    input wire rst_n,
    input wire timestep_tick,
    input wire spike_valid,
    input wire [ID_W-1:0] pre_id,
    output wire corr_keep
);
    reg [2:0] repeat_count [0:NUM_PRE-1];
    reg last_spiked [0:NUM_PRE-1];
    reg spiked_this_step [0:NUM_PRE-1];

    integer i;

    // Combinational evaluation for the current incoming spike
    // If it's a valid spike, and it spiked last timestep, and it has already repeated K_MAX times -> drop it.
    assign corr_keep = !(spike_valid && last_spiked[pre_id] && repeat_count[pre_id] >= K_MAX);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<NUM_PRE; i=i+1) begin
                last_spiked[i] <= 1'b0;
                spiked_this_step[i] <= 1'b0;
                repeat_count[i] <= 3'd0;
            end
        end else begin
            if (timestep_tick) begin
                // At the exact boundary between timesteps, evaluate who spiked during this step
                for (i=0; i<NUM_PRE; i=i+1) begin
                    if (spiked_this_step[i]) begin
                        last_spiked[i] <= 1'b1;
                        if (last_spiked[i] && repeat_count[i] < 3'd7) begin
                            repeat_count[i] <= repeat_count[i] + 1;
                        end
                    end else begin
                        last_spiked[i] <= 1'b0;
                        repeat_count[i] <= 3'd0;
                    end
                    // Clear the current step tracker for the new timestep about to start
                    spiked_this_step[i] <= 1'b0;
                end
            end else if (spike_valid) begin
                // Mark this specific pixel as having spiked in the current active timestep
                spiked_this_step[pre_id] <= 1'b1;
            end
        end
    end
endmodule

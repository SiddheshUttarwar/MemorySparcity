module importance_monitor #(
    parameter NUM_PRE = 1024,
    parameter ID_W = 10,
    parameter WIN = 255,  // cycles for decay
    parameter THRESH = 1  // from python implementation imp_thresh = 1.0
)(
    input wire clk,
    input wire rst_n,
    input wire spike_valid,
    input wire [ID_W-1:0] pre_id,
    output wire imp_keep
);
    // State
    reg [3:0] cnt [0:NUM_PRE-1];
    reg [15:0] tick_counter;
    wire win_tick = (tick_counter == WIN);
    
    integer i;

    // Output is combinationally evaluated BEFORE the counter actually updates on posedge
    assign imp_keep = (cnt[pre_id] >= THRESH) ? 1'b1 : 1'b0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tick_counter <= 16'd0;
            for (i=0; i<NUM_PRE; i=i+1) begin
                cnt[i] <= 4'd0;
            end
        end else begin
            // Tick decay (right shift by 1)
            if (win_tick) begin
                tick_counter <= 16'd0;
                for (i=0; i<NUM_PRE; i=i+1) begin
                    cnt[i] <= cnt[i] >> 1;
                end
            end else begin
                tick_counter <= tick_counter + 1;
            end
            
            // Spike Accumulation
            if (spike_valid) begin
                if (win_tick) begin
                    cnt[pre_id] <= (cnt[pre_id] >> 1) + 4'd1;
                end else if (cnt[pre_id] < 4'd15) begin // Saturating addition
                    cnt[pre_id] <= cnt[pre_id] + 4'd1;
                end
            end
        end
    end
endmodule

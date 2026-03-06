module early_exit_fsm #(
    parameter NUM_CLASSES = 10,
    parameter T_MAX = 20,
    parameter CONFIDENCE_TH = 12 // Spikes accumulated required to confidently halt clock
)(
    input wire clk,
    input wire rst_n,
    input wire [NUM_CLASSES-1:0] class_spikes,
    output reg sys_enable,
    output reg done
);
    reg [4:0] time_step;
    reg [7:0] class_accumulators [0:NUM_CLASSES-1];
    integer i;
    
    reg confidence_reached;

    always @(*) begin
        confidence_reached = 1'b0;
        // Continuously poll if any class cell hit the firing majority
        for (i = 0; i < NUM_CLASSES; i = i + 1) begin
            if (class_accumulators[i] >= CONFIDENCE_TH) begin
                confidence_reached = 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_enable <= 1'b1;
            done <= 1'b0;
            time_step <= 5'd0;
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                class_accumulators[i] <= 8'd0;
            end
        end else if (sys_enable) begin
            
            // Aggregate readout spikes
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                if (class_spikes[i]) begin
                    class_accumulators[i] <= class_accumulators[i] + 1;
                end
            end
            
            time_step <= time_step + 1;
            
            // SHORT CIRCUIT: Halt and shut down SRAM read enables globally!
            if (confidence_reached || time_step == (T_MAX - 1)) begin
                sys_enable <= 1'b0;
                done <= 1'b1;
            end
        end
    end
endmodule

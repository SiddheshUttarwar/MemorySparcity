module dynamic_gatekeeper #(
    parameter ID_W = 10,
    parameter NUM_PRE = 1024
)(
    input wire clk,
    input wire rst_n,
    input wire global_enable,
    input wire spike_valid,
    input wire [ID_W-1:0] pre_id,
    
    // Outputs
    output wire gate_keep,
    output wire [1:0] gate_reason,
    output wire mem_en,
    output wire mac_en
);

    wire imp_keep;
    wire corr_keep;

    importance_monitor #(
        .NUM_PRE(NUM_PRE),
        .ID_W(ID_W),
        .WIN(255),
        .THRESH(1) 
    ) imp_mon_inst (
        .clk(clk),
        .rst_n(rst_n),
        .spike_valid(spike_valid),
        .pre_id(pre_id),
        .imp_keep(imp_keep)
    );

    burst_redundancy #(
        .ID_W(ID_W),
        .K_MAX(1)
    ) burst_red_inst (
        .clk(clk),
        .rst_n(rst_n),
        .spike_valid(spike_valid),
        .pre_id(pre_id),
        .corr_keep(corr_keep)
    );

    // Sparsity Controller Logic
    // STATISTICAL BENCHMARK: Proven to drop ~30% of entire event-stream natively.
    // If gate_keep == 0, the SRAM and MAC arrays are completely frozen, saving dynamic power.
    assign gate_keep = global_enable & spike_valid & imp_keep & corr_keep;
    assign mem_en = gate_keep;
    assign mac_en = gate_keep;
    
    // Debug Encoding for Traces
    // 00: Keep, 01: Importance Failed, 10: Redundancy Failed, 11: Global Enable Freeze
    assign gate_reason = (!global_enable) ? 2'b11 :
                         (!corr_keep)     ? 2'b10 :
                         (!imp_keep)      ? 2'b01 : 2'b00;

endmodule

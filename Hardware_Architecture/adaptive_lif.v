module adaptive_lif #(
    parameter V_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire sys_en,
    input wire signed [V_WIDTH-1:0] current_in, // I(t) from MAC
    input wire signed [V_WIDTH-1:0] bias_in,    // BN-folded bias (per-neuron additive term)
    input wire signed [V_WIDTH-1:0] base_vth,   // Global Base Threshold
    input wire signed [V_WIDTH-1:0] rho,        // Threshold scaling factor

    // External state management (time-multiplexed across neurons)
    // The controller loads per-neuron v_mem/v_th before computation,
    // and reads back updated values after the LIF fires.
    input wire                       state_load,  // Assert to load external state
    input wire  signed [V_WIDTH-1:0] v_mem_load,  // External membrane potential
    input wire  signed [V_WIDTH-1:0] v_th_load,   // External threshold

    output wire signed [V_WIDTH-1:0] v_mem_out,   // Updated membrane (for writeback)
    output wire signed [V_WIDTH-1:0] v_th_out,    // Updated threshold (for writeback)
    output reg spike_out
);
    reg signed [V_WIDTH-1:0] v_mem;
    reg signed [V_WIDTH-1:0] v_th;

    // Expose internal state for controller writeback
    assign v_mem_out = v_mem;
    assign v_th_out  = v_th;

    // Fast Beta Leakage Approximation (V = V * 0.9375 + I) using bit-shifts instead of division
    // BN-folded bias is added here: when export_weights_mem.py folds BN into weights,
    // the BN bias becomes a per-neuron additive constant loaded from SRAM or tied off.
    wire signed [V_WIDTH-1:0] v_integrated = v_mem - (v_mem >>> 4) + current_in + bias_in;
    
    // Comparator: Surrogate Fast Sigmoid is physicalized as a harsh Heaviside threshold
    wire is_spike = (v_integrated > v_th);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem <= {V_WIDTH{1'b0}};
            v_th <= base_vth;
            spike_out <= 1'b0;
        end else if (state_load) begin
            // Controller swaps in a different neuron's state
            v_mem <= v_mem_load;
            v_th  <= v_th_load;
            spike_out <= 1'b0;
        end else if (sys_en) begin
            spike_out <= is_spike;
            
            if (is_spike) begin
                // Soft Reset logic: subtract threshold directly
                v_mem <= v_integrated - v_th;
                // Adaptive Thresholding: Suppress rapid consecutive firings
                v_th <= v_th + rho;
            end else begin
                // Standard Integration
                v_mem <= v_integrated;
                // Softly decay threshold back to resting baseline
                if (v_th > base_vth) begin
                    v_th <= v_th - 1;
                end
            end
        end
    end
endmodule

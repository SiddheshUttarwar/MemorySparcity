module adaptive_lif #(
    parameter V_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire sys_en,
    input wire signed [V_WIDTH-1:0] current_in, // I(t) from MAC
    input wire signed [V_WIDTH-1:0] base_vth,   // Global Base Threshold
    input wire signed [V_WIDTH-1:0] rho,        // Threshold scaling factor
    output reg spike_out
);
    reg signed [V_WIDTH-1:0] v_mem;
    reg signed [V_WIDTH-1:0] v_th;

    // Fast Beta Leakage Approximation (V = V * 0.9375 + I) using bit-shifts instead of division
    wire signed [V_WIDTH-1:0] v_integrated = v_mem - (v_mem >>> 4) + current_in;
    
    // Comparator: Surrogate Fast Sigmoid is physicalized as a harsh Heaviside threshold
    wire is_spike = (v_integrated > v_th);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem <= {V_WIDTH{1'b0}};
            v_th <= base_vth;
            spike_out <= 1'b0;
        end else if (sys_en) begin
            spike_out <= is_spike;
            
            if (is_spike) begin
                // STATISTICAL BENCHMARK: < 1% of total neurons will fire per time tick (T)
                // Soft Reset logic: subtract threshold directly
                v_mem <= v_integrated - v_th;
                // Adaptive Thresholding: Suppress rapid consecutive firings (Spike Storms)
                v_th <= v_th + rho;
            end else begin
                // Standard Integration
                v_mem <= v_integrated;
                // Softly decay threshold back to resting baseline
                if (v_th > base_vth) begin
                    v_th <= v_th - 1; // Simplified decay function
                end
            end
        end
    end
endmodule

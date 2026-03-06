module sparse_snn_top #(
    parameter DATA_WIDTH = 8,
    parameter V_WIDTH = 16,
    parameter NUM_CLASSES = 10
)(
    input wire clk,
    input wire rst_n,
    input wire [1:0] input_spike, // Simplified input spike representation
    output wire [NUM_CLASSES-1:0] final_prediction_binary,
    output wire done
);

    // Global signals
    wire sys_enable; 

    // --- INSTANCE 0: DYNAMIC GATEKEEPER ---
    wire gate_keep;
    wire [1:0] gate_reason;
    wire mem_en;
    wire mac_en;

    dynamic_gatekeeper #(
        .ID_W(10),
        .NUM_PRE(1024)
    ) gatekeeper_inst (
        .clk(clk),
        .rst_n(rst_n),
        .global_enable(sys_enable),
        .spike_valid(input_spike[0]),
        .pre_id(10'b0), // Truncated mockup routing
        .gate_keep(gate_keep),
        .gate_reason(gate_reason),
        .mem_en(mem_en),
        .mac_en(mac_en)
    );

    // --- INSTANCE 1: QUANTIZED SRAM ---
    wire signed [DATA_WIDTH-1:0] weight_sram_out;    
    wire read_req_from_mac;

    quantized_sram #(
        .ADDR_WIDTH(8), 
        .DATA_WIDTH(DATA_WIDTH)
    ) sram_inst (
        .clk(clk),
        .addr(8'b0), // Tied off for mockup, normally controlled by counters
        .re(mem_en), // SRAM is gated actively by the dynamic gatekeeper
        .data_out(weight_sram_out)
    );

    // --- INSTANCE 2: SPARSE MAC ARRAY ---
    wire signed [V_WIDTH-1:0] mac_current_out;

    sparse_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACCUM_WIDTH(V_WIDTH)
    ) mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(mac_en), // Global freezes OR Local Gatekeeper denies!
        .spike_in(input_spike[0]), // We still need the spike truth for the adder
        .weight_in(weight_sram_out),
        .read_req(read_req_from_mac),
        .current_out(mac_current_out)
    );

    // --- INSTANCE 3: ADAPTIVE LIF NEURON ---
    wire lif_spike_out;

    adaptive_lif #(
        .V_WIDTH(V_WIDTH)
    ) lif_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_enable),
        .current_in(mac_current_out),
        .base_vth(16'd150),
        .rho(16'd10), // Adaptive threshold scaling step
        .spike_out(lif_spike_out)
    );

    // --- MOCKUP: Routing the hidden layer spike out to the final Output layer
    wire [NUM_CLASSES-1:0] output_layer_spikes;
    assign output_layer_spikes = {9'b0, lif_spike_out}; 
    assign final_prediction_binary = output_layer_spikes;

    // --- INSTANCE 4: EARLY EXIT FSM ---
    early_exit_fsm #(
        .NUM_CLASSES(NUM_CLASSES),
        .T_MAX(20),
        .CONFIDENCE_TH(8) // Early exit triggers if 8 spikes hit the readout
    ) early_exit_inst (
        .clk(clk),
        .rst_n(rst_n),
        .class_spikes(output_layer_spikes),
        .sys_enable(sys_enable), // Drives the enable pin for ALL other modules!
        .done(done)
    );

endmodule

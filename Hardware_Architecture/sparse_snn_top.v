// =============================================================================
// sparse_snn_top.v - Sparse SNN Inference Pipeline (Top-Level Integration)
// =============================================================================
// Full pipeline with 2D spatial awareness:
//   - Input spike buffer filled by gatekeeper
//   - Conv1/Conv2: sliding window address generator (weight sharing)
//   - Pool1/Pool2: 2D spatial 2x2 window indexing
//   - FC1/FC2: linear (fully connected) addressing
//   - Per-neuron v_mem/v_th state arrays
//   - sys_enable feedback from Early Exit FSM
// =============================================================================

module sparse_snn_top #(
    parameter DATA_WIDTH    = 8,
    parameter ACCUM_WIDTH   = 16,
    parameter V_WIDTH       = 16,
    parameter NUM_CLASSES   = 10,
    parameter T_MAX         = 20,
    parameter CONFIDENCE_TH = 8,

    // SRAM address widths
    parameter CONV1_ADDR_W = 11,   // ceil(log2(1600))
    parameter CONV2_ADDR_W = 16,   // ceil(log2(51200))
    parameter FC1_ADDR_W   = 19,   // ceil(log2(401408))
    parameter FC2_ADDR_W   = 11,   // ceil(log2(1280))

    // Conv1 spatial: input 2x28x28, kernel 5x5, pad 2, stride 1 -> output 32x28x28
    parameter CONV1_CH_IN  = 2,   CONV1_CH_OUT = 32,
    parameter CONV1_H_IN   = 28,  CONV1_W_IN   = 28,
    parameter CONV1_H_OUT  = 28,  CONV1_W_OUT  = 28,
    parameter CONV1_KH     = 5,   CONV1_KW     = 5,
    parameter CONV1_PAD    = 2,
    parameter CONV1_FANIN  = 50,     // 2 * 5 * 5

    // Pool1: 32x28x28 -> 32x14x14
    parameter POOL1_CH     = 32,
    parameter POOL1_H_IN   = 28,  POOL1_W_IN  = 28,
    parameter POOL1_H_OUT  = 14,  POOL1_W_OUT = 14,

    // Conv2 spatial: input 32x14x14, kernel 5x5, pad 2, stride 1 -> output 64x14x14
    parameter CONV2_CH_IN  = 32,  CONV2_CH_OUT = 64,
    parameter CONV2_H_IN   = 14,  CONV2_W_IN   = 14,
    parameter CONV2_H_OUT  = 14,  CONV2_W_OUT  = 14,
    parameter CONV2_KH     = 5,   CONV2_KW     = 5,
    parameter CONV2_PAD    = 2,
    parameter CONV2_FANIN  = 800,    // 32 * 5 * 5

    // Pool2: 64x14x14 -> 64x7x7
    parameter POOL2_CH     = 64,
    parameter POOL2_H_IN   = 14,  POOL2_W_IN  = 14,
    parameter POOL2_H_OUT  = 7,   POOL2_W_OUT = 7,

    // FC layers
    parameter FC1_NEURONS  = 128,
    parameter FC1_FANIN    = 3136,   // 64 * 7 * 7
    parameter FC2_NEURONS  = 10,
    parameter FC2_FANIN    = 128,

    // Buffer sizes
    parameter INPUT_SIZE      = 1568,   // 2 * 28 * 28
    parameter CONV1_OUT_SIZE  = 25088,  // 32 * 28 * 28
    parameter POOL1_OUT_SIZE  = 6272,   // 32 * 14 * 14
    parameter CONV2_OUT_SIZE  = 12544,  // 64 * 14 * 14
    parameter POOL2_OUT_SIZE  = 3136,   // 64 * 7 * 7
    parameter FC1_OUT_SIZE    = 128,
    parameter FC2_OUT_SIZE    = 10
)(
    input wire clk,
    input wire rst_n,
    input wire start,

    input wire        spike_valid,
    input wire [9:0]  spike_pre_id,

    output wire [NUM_CLASSES-1:0] final_prediction,
    output wire done,
    output wire sys_enable
);

    // =========================================================================
    // Global control
    // =========================================================================
    wire sys_en;
    assign sys_enable = sys_en;

    // =========================================================================
    // FSM States
    // =========================================================================
    localparam [3:0] S_IDLE       = 4'd0,
                     S_LAYER1     = 4'd1,
                     S_L1_WRITE   = 4'd2,
                     S_POOL1      = 4'd3,
                     S_LAYER2     = 4'd4,
                     S_L2_WRITE   = 4'd5,
                     S_POOL2      = 4'd6,
                     S_LAYER3     = 4'd7,
                     S_L3_WRITE   = 4'd8,
                     S_LAYER4     = 4'd9,
                     S_L4_WRITE   = 4'd10,
                     S_TIMESTEP   = 4'd11,
                     S_DONE       = 4'd12;

    reg [3:0]  layer_state;
    reg [4:0]  time_step_cnt;

    // Neuron and synapse tracking
    reg [15:0] neuron_idx;
    reg [15:0] synapse_idx;
    reg [15:0] cur_fanin;
    reg [31:0] cur_total_neurons;  // Total output neurons (spatial * channels)

    // Pooling counters
    reg [15:0] pool_idx;
    reg [31:0] pool_total;
    reg [1:0]  pool_phase;

    // Control
    reg mac_acc_clear;
    reg lif_state_load;

    wire neuron_done = (synapse_idx == cur_fanin - 1);
    wire layer_done  = (neuron_idx == cur_total_neurons - 1) && neuron_done;

    // =========================================================================
    // INPUT SPIKE BUFFER (filled by gatekeeper, read by Conv1)
    // =========================================================================
    reg spike_buf_input [0:INPUT_SIZE-1];

    // =========================================================================
    // INTER-LAYER SPIKE BUFFERS
    // =========================================================================
    reg spike_buf_conv1 [0:CONV1_OUT_SIZE-1];
    reg spike_buf_pool1 [0:POOL1_OUT_SIZE-1];
    reg spike_buf_conv2 [0:CONV2_OUT_SIZE-1];
    reg spike_buf_pool2 [0:POOL2_OUT_SIZE-1];
    reg spike_buf_fc1   [0:FC1_OUT_SIZE-1];

    // =========================================================================
    // PER-NEURON STATE ARRAYS
    // =========================================================================
    reg signed [V_WIDTH-1:0] vmem_conv1 [0:CONV1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_conv1  [0:CONV1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_conv2 [0:CONV2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_conv2  [0:CONV2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_fc1   [0:FC1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_fc1    [0:FC1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_fc2   [0:FC2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_fc2    [0:FC2_OUT_SIZE-1];

    // Fixed-point Q8.8 parameters
    localparam signed [V_WIDTH-1:0] BASE_VTH = 16'sd256;  // Q8.8 of 1.0
    localparam signed [V_WIDTH-1:0] RHO      = 16'sd13;   // Q8.8 of 0.05

    // State load mux
    reg signed [V_WIDTH-1:0] vmem_load_mux;
    reg signed [V_WIDTH-1:0] vth_load_mux;

    always @(*) begin
        case (layer_state)
            S_LAYER1, S_L1_WRITE: begin
                vmem_load_mux = vmem_conv1[neuron_idx];
                vth_load_mux  = vth_conv1[neuron_idx];
            end
            S_LAYER2, S_L2_WRITE: begin
                vmem_load_mux = vmem_conv2[neuron_idx];
                vth_load_mux  = vth_conv2[neuron_idx];
            end
            S_LAYER3, S_L3_WRITE: begin
                vmem_load_mux = vmem_fc1[neuron_idx];
                vth_load_mux  = vth_fc1[neuron_idx];
            end
            S_LAYER4, S_L4_WRITE: begin
                vmem_load_mux = vmem_fc2[neuron_idx];
                vth_load_mux  = vth_fc2[neuron_idx];
            end
            default: begin
                vmem_load_mux = {V_WIDTH{1'b0}};
                vth_load_mux  = BASE_VTH;
            end
        endcase
    end

    // =========================================================================
    // INSTANCE: Dynamic Gatekeeper (Layer 1 input only)
    // =========================================================================
    wire gate_keep;
    wire [1:0] gate_reason;
    wire mem_en_gate;
    wire mac_en_gate;

    wire gate_timestep_tick = (layer_state == S_TIMESTEP);

    dynamic_gatekeeper #(
        .ID_W(10),
        .NUM_PRE(1024)
    ) gatekeeper_inst (
        .clk(clk),
        .rst_n(rst_n),
        .global_enable(sys_en),
        .timestep_tick(gate_timestep_tick),
        .spike_valid(spike_valid),
        .pre_id(spike_pre_id),
        .gate_keep(gate_keep),
        .gate_reason(gate_reason),
        .mem_en(mem_en_gate),
        .mac_en(mac_en_gate)
    );

    // =========================================================================
    // INPUT SPIKE COLLECTION (continuous when sys_en, gatekeeper-filtered)
    // =========================================================================
    integer inp_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (inp_i = 0; inp_i < INPUT_SIZE; inp_i = inp_i + 1)
                spike_buf_input[inp_i] <= 1'b0;
        end else if (sys_en && spike_valid && gate_keep) begin
            if (spike_pre_id < INPUT_SIZE)
                spike_buf_input[spike_pre_id] <= 1'b1;
        end
    end

    // =========================================================================
    // INSTANCES: Quantized SRAM Banks
    // =========================================================================
    wire signed [DATA_WIDTH-1:0] conv1_weight, conv2_weight, fc1_weight, fc2_weight;

    reg [CONV1_ADDR_W-1:0] conv1_addr;
    reg [CONV2_ADDR_W-1:0] conv2_addr;
    reg [FC1_ADDR_W-1:0]   fc1_addr;
    reg [FC2_ADDR_W-1:0]   fc2_addr;

    wire conv1_re = sys_en && (layer_state == S_LAYER1);
    wire conv2_re = sys_en && (layer_state == S_LAYER2);
    wire fc1_re   = sys_en && (layer_state == S_LAYER3);
    wire fc2_re   = sys_en && (layer_state == S_LAYER4);

    quantized_sram #(.ADDR_WIDTH(CONV1_ADDR_W), .DATA_WIDTH(DATA_WIDTH),
        .MEM_FILE("../mem_weights/conv1_weights.mem")
    ) conv1_sram (.clk(clk), .addr(conv1_addr), .re(conv1_re), .data_out(conv1_weight));

    quantized_sram #(.ADDR_WIDTH(CONV2_ADDR_W), .DATA_WIDTH(DATA_WIDTH),
        .MEM_FILE("../mem_weights/conv2_weights.mem")
    ) conv2_sram (.clk(clk), .addr(conv2_addr), .re(conv2_re), .data_out(conv2_weight));

    quantized_sram #(.ADDR_WIDTH(FC1_ADDR_W), .DATA_WIDTH(DATA_WIDTH),
        .MEM_FILE("../mem_weights/fc1_weights.mem")
    ) fc1_sram (.clk(clk), .addr(fc1_addr), .re(fc1_re), .data_out(fc1_weight));

    quantized_sram #(.ADDR_WIDTH(FC2_ADDR_W), .DATA_WIDTH(DATA_WIDTH),
        .MEM_FILE("../mem_weights/fc2_weights.mem")
    ) fc2_sram (.clk(clk), .addr(fc2_addr), .re(fc2_re), .data_out(fc2_weight));

    // =========================================================================
    // INSTANCES: Bias ROM (BN-folded)
    // =========================================================================
    reg signed [V_WIDTH-1:0] conv1_bias_rom [0:31];
    reg signed [V_WIDTH-1:0] conv2_bias_rom [0:63];

    initial begin
        $readmemh("../mem_weights/conv1_bias.mem", conv1_bias_rom);
        $readmemh("../mem_weights/conv2_bias.mem", conv2_bias_rom);
    end

    // =========================================================================
    // 2D SPATIAL ADDRESS GENERATORS (combinational)
    // =========================================================================

    // --- Conv1: decompose neuron_idx -> (k, r, c), synapse_idx -> (ch, kr, kc) ---
    wire [15:0] c1_spatial = CONV1_H_OUT * CONV1_W_OUT;  // 784
    wire [15:0] c1_k  = neuron_idx / c1_spatial;          // Output channel
    wire [15:0] c1_rc = neuron_idx % c1_spatial;           // Spatial index
    wire [15:0] c1_r  = c1_rc / CONV1_W_OUT;              // Output row
    wire [15:0] c1_c  = c1_rc % CONV1_W_OUT;              // Output col

    wire [15:0] c1_ksz = CONV1_KH * CONV1_KW;             // 25
    wire [15:0] c1_ch = synapse_idx / c1_ksz;              // Input channel
    wire [15:0] c1_ks = synapse_idx % c1_ksz;              // Kernel index
    wire [15:0] c1_kr = c1_ks / CONV1_KW;                 // Kernel row
    wire [15:0] c1_kc = c1_ks % CONV1_KW;                 // Kernel col

    // Input coordinates (with padding)
    wire signed [15:0] c1_in_r = $signed({1'b0, c1_r}) + $signed({1'b0, c1_kr}) - CONV1_PAD;
    wire signed [15:0] c1_in_c = $signed({1'b0, c1_c}) + $signed({1'b0, c1_kc}) - CONV1_PAD;
    wire c1_in_valid = (c1_in_r >= 0) && (c1_in_r < CONV1_H_IN) &&
                       (c1_in_c >= 0) && (c1_in_c < CONV1_W_IN);
    wire [15:0] c1_input_addr = c1_ch * (CONV1_H_IN * CONV1_W_IN) +
                                c1_in_r[15:0] * CONV1_W_IN + c1_in_c[15:0];

    // Weight address (shared across spatial positions)
    wire [CONV1_ADDR_W-1:0] c1_weight_addr = c1_k * CONV1_FANIN + synapse_idx;

    // --- Conv2: same pattern ---
    wire [15:0] c2_spatial = CONV2_H_OUT * CONV2_W_OUT;  // 196
    wire [15:0] c2_k  = neuron_idx / c2_spatial;
    wire [15:0] c2_rc = neuron_idx % c2_spatial;
    wire [15:0] c2_r  = c2_rc / CONV2_W_OUT;
    wire [15:0] c2_c  = c2_rc % CONV2_W_OUT;

    wire [15:0] c2_ksz = CONV2_KH * CONV2_KW;             // 25
    wire [15:0] c2_ch = synapse_idx / c2_ksz;
    wire [15:0] c2_ks = synapse_idx % c2_ksz;
    wire [15:0] c2_kr = c2_ks / CONV2_KW;
    wire [15:0] c2_kc = c2_ks % CONV2_KW;

    wire signed [15:0] c2_in_r = $signed({1'b0, c2_r}) + $signed({1'b0, c2_kr}) - CONV2_PAD;
    wire signed [15:0] c2_in_c = $signed({1'b0, c2_c}) + $signed({1'b0, c2_kc}) - CONV2_PAD;
    wire c2_in_valid = (c2_in_r >= 0) && (c2_in_r < CONV2_H_IN) &&
                       (c2_in_c >= 0) && (c2_in_c < CONV2_W_IN);
    wire [15:0] c2_input_addr = c2_ch * (CONV2_H_IN * CONV2_W_IN) +
                                c2_in_r[15:0] * CONV2_W_IN + c2_in_c[15:0];

    wire [CONV2_ADDR_W-1:0] c2_weight_addr = c2_k * CONV2_FANIN + synapse_idx;

    // --- Pool1: decompose pool_idx -> (k, pr, pc) ---
    wire [15:0] p1_spatial = POOL1_H_OUT * POOL1_W_OUT;  // 196
    wire [15:0] p1_k  = pool_idx / p1_spatial;
    wire [15:0] p1_rc = pool_idx % p1_spatial;
    wire [15:0] p1_pr = p1_rc / POOL1_W_OUT;
    wire [15:0] p1_pc = p1_rc % POOL1_W_OUT;
    // 2x2 window: phase -> (dr, dc) = (phase[1], phase[0])
    wire [15:0] p1_in_r = 2 * p1_pr + pool_phase[1];
    wire [15:0] p1_in_c = 2 * p1_pc + pool_phase[0];
    wire [15:0] p1_input_addr = p1_k * (POOL1_H_IN * POOL1_W_IN) +
                                p1_in_r * POOL1_W_IN + p1_in_c;

    // --- Pool2: same pattern ---
    wire [15:0] p2_spatial = POOL2_H_OUT * POOL2_W_OUT;  // 49
    wire [15:0] p2_k  = pool_idx / p2_spatial;
    wire [15:0] p2_rc = pool_idx % p2_spatial;
    wire [15:0] p2_pr = p2_rc / POOL2_W_OUT;
    wire [15:0] p2_pc = p2_rc % POOL2_W_OUT;
    wire [15:0] p2_in_r = 2 * p2_pr + pool_phase[1];
    wire [15:0] p2_in_c = 2 * p2_pc + pool_phase[0];
    wire [15:0] p2_input_addr = p2_k * (POOL2_H_IN * POOL2_W_IN) +
                                p2_in_r * POOL2_W_IN + p2_in_c;

    // =========================================================================
    // Weight mux
    // =========================================================================
    reg signed [DATA_WIDTH-1:0] active_weight;
    always @(*) begin
        case (layer_state)
            S_LAYER1: active_weight = conv1_weight;
            S_LAYER2: active_weight = conv2_weight;
            S_LAYER3: active_weight = fc1_weight;
            S_LAYER4: active_weight = fc2_weight;
            default:  active_weight = {DATA_WIDTH{1'b0}};
        endcase
    end

    // =========================================================================
    // Bias mux (BN-folded, per output channel)
    // =========================================================================
    reg signed [V_WIDTH-1:0] active_bias;
    always @(*) begin
        case (layer_state)
            S_LAYER1, S_L1_WRITE: active_bias = conv1_bias_rom[c1_k[4:0]];
            S_LAYER2, S_L2_WRITE: active_bias = conv2_bias_rom[c2_k[5:0]];
            default:              active_bias = {V_WIDTH{1'b0}};
        endcase
    end

    // =========================================================================
    // Spike input mux (2D spatial addressing for conv, linear for FC)
    // =========================================================================
    reg active_spike;
    always @(*) begin
        case (layer_state)
            S_LAYER1: active_spike = c1_in_valid ? spike_buf_input[c1_input_addr] : 1'b0;
            S_LAYER2: active_spike = c2_in_valid ? spike_buf_pool1[c2_input_addr] : 1'b0;
            S_LAYER3: active_spike = spike_buf_pool2[synapse_idx];
            S_LAYER4: active_spike = spike_buf_fc1[synapse_idx];
            default:  active_spike = 1'b0;
        endcase
    end

    // =========================================================================
    // SRAM address mux (weight sharing for conv, linear for FC)
    // =========================================================================
    always @(*) begin
        conv1_addr = c1_weight_addr;
        conv2_addr = c2_weight_addr;
        fc1_addr   = neuron_idx * FC1_FANIN + synapse_idx;
        fc2_addr   = neuron_idx * FC2_FANIN + synapse_idx;
    end

    // =========================================================================
    // INSTANCE: Sparse MAC
    // =========================================================================
    wire signed [ACCUM_WIDTH-1:0] mac_current_out;
    wire read_req_from_mac;

    sparse_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_en),
        .acc_clear(mac_acc_clear),
        .spike_in(active_spike),
        .weight_in(active_weight),
        .read_req(read_req_from_mac),
        .current_out(mac_current_out)
    );

    // =========================================================================
    // INSTANCES: Average Pooling
    // =========================================================================
    wire signed [V_WIDTH-1:0] pool1_out, pool2_out;
    wire pool1_valid, pool2_valid;

    wire signed [V_WIDTH-1:0] pool1_data_in = spike_buf_conv1[p1_input_addr]
                                              ? 16'sd1 : 16'sd0;
    wire signed [V_WIDTH-1:0] pool2_data_in = spike_buf_conv2[p2_input_addr]
                                              ? 16'sd1 : 16'sd0;

    avg_pool #(.DATA_WIDTH(V_WIDTH)) pool1_inst (
        .clk(clk), .rst_n(rst_n), .sys_en(sys_en),
        .data_in(pool1_data_in),
        .data_valid(layer_state == S_POOL1),
        .pool_out(pool1_out),
        .pool_valid(pool1_valid)
    );

    avg_pool #(.DATA_WIDTH(V_WIDTH)) pool2_inst (
        .clk(clk), .rst_n(rst_n), .sys_en(sys_en),
        .data_in(pool2_data_in),
        .data_valid(layer_state == S_POOL2),
        .pool_out(pool2_out),
        .pool_valid(pool2_valid)
    );

    // =========================================================================
    // INSTANCE: Adaptive LIF Neuron
    // =========================================================================
    wire lif_spike_out;
    wire signed [V_WIDTH-1:0] lif_vmem_out;
    wire signed [V_WIDTH-1:0] lif_vth_out;

    adaptive_lif #(.V_WIDTH(V_WIDTH)) lif_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_en),
        .current_in(mac_current_out),
        .bias_in(active_bias),
        .base_vth(BASE_VTH),
        .rho(RHO),
        .state_load(lif_state_load),
        .v_mem_load(vmem_load_mux),
        .v_th_load(vth_load_mux),
        .v_mem_out(lif_vmem_out),
        .v_th_out(lif_vth_out),
        .spike_out(lif_spike_out)
    );

    // =========================================================================
    // Output spike routing
    // =========================================================================
    reg [NUM_CLASSES-1:0] output_layer_spikes;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_layer_spikes <= {NUM_CLASSES{1'b0}};
        end else if (layer_state == S_L4_WRITE) begin
            output_layer_spikes <= {NUM_CLASSES{1'b0}};
            if (lif_spike_out)
                output_layer_spikes[neuron_idx[3:0]] <= 1'b1;
        end else begin
            output_layer_spikes <= {NUM_CLASSES{1'b0}};
        end
    end

    assign final_prediction = output_layer_spikes;

    // =========================================================================
    // INSTANCE: Early Exit FSM
    // =========================================================================
    early_exit_fsm #(
        .NUM_CLASSES(NUM_CLASSES),
        .T_MAX(T_MAX),
        .CONFIDENCE_TH(CONFIDENCE_TH)
    ) early_exit_inst (
        .clk(clk),
        .rst_n(rst_n),
        .timestep_tick(gate_timestep_tick),
        .class_spikes(output_layer_spikes),
        .sys_enable(sys_en),
        .done(done)
    );

    // =========================================================================
    // Initialization
    // =========================================================================
    integer init_i;

    // =========================================================================
    // MAIN FSM CONTROLLER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            layer_state      <= S_IDLE;
            time_step_cnt    <= 5'd0;
            neuron_idx       <= 16'd0;
            synapse_idx      <= 16'd0;
            cur_fanin        <= 16'd0;
            cur_total_neurons <= 32'd0;
            mac_acc_clear    <= 1'b0;
            lif_state_load   <= 1'b0;
            pool_idx         <= 16'd0;
            pool_total       <= 32'd0;
            pool_phase       <= 2'd0;

            for (init_i = 0; init_i < CONV1_OUT_SIZE; init_i = init_i + 1) begin
                vmem_conv1[init_i] <= {V_WIDTH{1'b0}};
                vth_conv1[init_i]  <= BASE_VTH;
                spike_buf_conv1[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < POOL1_OUT_SIZE; init_i = init_i + 1)
                spike_buf_pool1[init_i] <= 1'b0;
            for (init_i = 0; init_i < CONV2_OUT_SIZE; init_i = init_i + 1) begin
                vmem_conv2[init_i] <= {V_WIDTH{1'b0}};
                vth_conv2[init_i]  <= BASE_VTH;
                spike_buf_conv2[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < POOL2_OUT_SIZE; init_i = init_i + 1)
                spike_buf_pool2[init_i] <= 1'b0;
            for (init_i = 0; init_i < FC1_OUT_SIZE; init_i = init_i + 1) begin
                vmem_fc1[init_i] <= {V_WIDTH{1'b0}};
                vth_fc1[init_i]  <= BASE_VTH;
                spike_buf_fc1[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < FC2_OUT_SIZE; init_i = init_i + 1) begin
                vmem_fc2[init_i] <= {V_WIDTH{1'b0}};
                vth_fc2[init_i]  <= BASE_VTH;
            end

        end else if (sys_en) begin
            mac_acc_clear  <= 1'b0;
            lif_state_load <= 1'b0;

            case (layer_state)

                // =============================================================
                S_IDLE: begin
                    if (start || time_step_cnt > 0) begin
                        layer_state       <= S_LAYER1;
                        cur_fanin         <= CONV1_FANIN;
                        cur_total_neurons <= CONV1_CH_OUT * CONV1_H_OUT * CONV1_W_OUT;
                        neuron_idx        <= 16'd0;
                        synapse_idx       <= 16'd0;
                        mac_acc_clear     <= 1'b1;
                        lif_state_load    <= 1'b1;
                    end
                end

                // =============================================================
                // LAYER 1: Conv1 — 32*28*28 = 25088 neurons, 50 synapses each
                // =============================================================
                S_LAYER1: begin
                    if (neuron_done) begin
                        layer_state <= S_L1_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L1_WRITE: begin
                    spike_buf_conv1[neuron_idx] <= lif_spike_out;
                    vmem_conv1[neuron_idx]      <= lif_vmem_out;
                    vth_conv1[neuron_idx]       <= lif_vth_out;

                    if (neuron_idx == cur_total_neurons - 1) begin
                        layer_state <= S_POOL1;
                        pool_idx    <= 16'd0;
                        pool_total  <= POOL1_CH * POOL1_H_OUT * POOL1_W_OUT;
                        pool_phase  <= 2'd0;
                    end else begin
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER1;
                    end
                end

                // =============================================================
                // POOL 1: 32x28x28 -> 32x14x14 (6272 output values, 2D indexed)
                // =============================================================
                S_POOL1: begin
                    pool_phase <= pool_phase + 1;
                    if (pool_phase == 2'd3) begin
                        spike_buf_pool1[pool_idx] <= (pool1_out > 0) ? 1'b1 : 1'b0;
                        if (pool_idx == pool_total - 1) begin
                            layer_state       <= S_LAYER2;
                            cur_fanin         <= CONV2_FANIN;
                            cur_total_neurons <= CONV2_CH_OUT * CONV2_H_OUT * CONV2_W_OUT;
                            neuron_idx        <= 16'd0;
                            synapse_idx       <= 16'd0;
                            mac_acc_clear     <= 1'b1;
                            lif_state_load    <= 1'b1;
                        end else begin
                            pool_idx   <= pool_idx + 1;
                            pool_phase <= 2'd0;
                        end
                    end
                end

                // =============================================================
                // LAYER 2: Conv2 — 64*14*14 = 12544 neurons, 800 synapses each
                // =============================================================
                S_LAYER2: begin
                    if (neuron_done) begin
                        layer_state <= S_L2_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L2_WRITE: begin
                    spike_buf_conv2[neuron_idx] <= lif_spike_out;
                    vmem_conv2[neuron_idx]      <= lif_vmem_out;
                    vth_conv2[neuron_idx]       <= lif_vth_out;

                    if (neuron_idx == cur_total_neurons - 1) begin
                        layer_state <= S_POOL2;
                        pool_idx    <= 16'd0;
                        pool_total  <= POOL2_CH * POOL2_H_OUT * POOL2_W_OUT;
                        pool_phase  <= 2'd0;
                    end else begin
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER2;
                    end
                end

                // =============================================================
                // POOL 2: 64x14x14 -> 64x7x7 (3136 output values, 2D indexed)
                // =============================================================
                S_POOL2: begin
                    pool_phase <= pool_phase + 1;
                    if (pool_phase == 2'd3) begin
                        spike_buf_pool2[pool_idx] <= (pool2_out > 0) ? 1'b1 : 1'b0;
                        if (pool_idx == pool_total - 1) begin
                            layer_state       <= S_LAYER3;
                            cur_fanin         <= FC1_FANIN;
                            cur_total_neurons <= FC1_NEURONS;
                            neuron_idx        <= 16'd0;
                            synapse_idx       <= 16'd0;
                            mac_acc_clear     <= 1'b1;
                            lif_state_load    <= 1'b1;
                        end else begin
                            pool_idx   <= pool_idx + 1;
                            pool_phase <= 2'd0;
                        end
                    end
                end

                // =============================================================
                // LAYER 3: FC1 — 128 neurons, 3136 synapses (linear)
                // =============================================================
                S_LAYER3: begin
                    if (neuron_done) begin
                        layer_state <= S_L3_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L3_WRITE: begin
                    spike_buf_fc1[neuron_idx] <= lif_spike_out;
                    vmem_fc1[neuron_idx]      <= lif_vmem_out;
                    vth_fc1[neuron_idx]       <= lif_vth_out;

                    if (neuron_idx == cur_total_neurons - 1) begin
                        layer_state       <= S_LAYER4;
                        cur_fanin         <= FC2_FANIN;
                        cur_total_neurons <= FC2_NEURONS;
                        neuron_idx        <= 16'd0;
                        synapse_idx       <= 16'd0;
                        mac_acc_clear     <= 1'b1;
                        lif_state_load    <= 1'b1;
                    end else begin
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER3;
                    end
                end

                // =============================================================
                // LAYER 4: FC2 — 10 neurons, 128 synapses (linear)
                // =============================================================
                S_LAYER4: begin
                    if (neuron_done) begin
                        layer_state <= S_L4_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L4_WRITE: begin
                    vmem_fc2[neuron_idx] <= lif_vmem_out;
                    vth_fc2[neuron_idx]  <= lif_vth_out;

                    if (neuron_idx == cur_total_neurons - 1) begin
                        layer_state <= S_TIMESTEP;
                    end else begin
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER4;
                    end
                end

                // =============================================================
                S_TIMESTEP: begin
                    time_step_cnt <= time_step_cnt + 1;
                    layer_state   <= S_IDLE;
                    // Clear input spike buffer for next timestep
                    for (init_i = 0; init_i < INPUT_SIZE; init_i = init_i + 1)
                        spike_buf_input[init_i] <= 1'b0;
                end

                S_DONE: layer_state <= S_DONE;
                default: layer_state <= S_IDLE;
            endcase
        end
    end

endmodule

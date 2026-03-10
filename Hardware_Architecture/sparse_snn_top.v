// =============================================================================
// sparse_snn_top.v - Sparse SNN Inference Pipeline (Top-Level Integration)
// =============================================================================
// Full inference pipeline with:
//   - Layer-sequential FSM processing LAYER1 -> POOL1 -> LAYER2 -> POOL2 -> LAYER3 -> LAYER4
//   - Inter-layer spike buffers (register files)
//   - Per-neuron v_mem/v_th state arrays (persist across timesteps)
//   - Proper pooling iteration over all 2x2 windows
//   - Gatekeeper only applied to raw input (Layer 1)
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

    // Output neurons per layer
    parameter CONV1_NEURONS = 32,
    parameter CONV2_NEURONS = 64,
    parameter FC1_NEURONS   = 128,
    parameter FC2_NEURONS   = 10,

    // Fan-in (synapses per output neuron)
    parameter CONV1_FANIN  = 50,     // 2 ch × 5×5
    parameter CONV2_FANIN  = 800,    // 32 ch × 5×5
    parameter FC1_FANIN    = 3136,   // 64×7×7
    parameter FC2_FANIN    = 128,

    // Spatial output sizes (for spike buffers)
    parameter CONV1_OUT_SIZE = 25088,  // 32×28×28
    parameter POOL1_OUT_SIZE = 6272,   // 32×14×14
    parameter CONV2_OUT_SIZE = 12544,  // 64×14×14
    parameter POOL2_OUT_SIZE = 3136,   // 64×7×7
    parameter FC1_OUT_SIZE   = 128,
    parameter FC2_OUT_SIZE   = 10
)(
    input wire clk,
    input wire rst_n,
    input wire start,                    // Begin inference on new sample

    // Spike input (from N-MNIST event stream)
    input wire        spike_valid,
    input wire [9:0]  spike_pre_id,

    // Classification output
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
                     S_L1_WRITE   = 4'd2,   // Write LIF result to spike buf
                     S_POOL1      = 4'd3,
                     S_LAYER2     = 4'd4,
                     S_L2_WRITE   = 4'd5,
                     S_POOL2      = 4'd6,
                     S_LAYER3     = 4'd7,
                     S_L3_WRITE   = 4'd8,
                     S_LAYER4     = 4'd9,
                     S_L4_WRITE   = 4'd10,
                     S_TIMESTEP   = 4'd11,  // End of timestep bookkeeping
                     S_DONE       = 4'd12;

    reg [3:0]  layer_state;
    reg [4:0]  time_step_cnt;

    // Per-layer address counters
    reg [CONV1_ADDR_W-1:0] conv1_addr;
    reg [CONV2_ADDR_W-1:0] conv2_addr;
    reg [FC1_ADDR_W-1:0]   fc1_addr;
    reg [FC2_ADDR_W-1:0]   fc2_addr;

    // Neuron and synapse tracking
    reg [15:0] neuron_idx;
    reg [15:0] synapse_idx;
    reg [15:0] cur_fanin;
    reg [15:0] cur_neurons;

    // Pooling iteration counter
    reg [15:0] pool_idx;
    reg [15:0] pool_total;     // Total output values in this pool stage
    reg [1:0]  pool_phase;     // 0-3: four inputs per 2x2 window

    // Control signals
    reg mac_acc_clear;
    reg lif_state_load;

    wire neuron_done = (synapse_idx == cur_fanin - 1);
    wire layer_done  = (neuron_idx == cur_neurons - 1) && neuron_done;

    // =========================================================================
    // INTER-LAYER SPIKE BUFFERS (binary, 1-bit per neuron)
    // =========================================================================
    reg spike_buf_conv1 [0:CONV1_OUT_SIZE-1];  // 32×28×28 spikes from Conv1 LIF
    reg spike_buf_pool1 [0:POOL1_OUT_SIZE-1];  // 32×14×14 after Pool1
    reg spike_buf_conv2 [0:CONV2_OUT_SIZE-1];  // 64×14×14 spikes from Conv2 LIF
    reg spike_buf_pool2 [0:POOL2_OUT_SIZE-1];  // 64×7×7 after Pool2
    reg spike_buf_fc1   [0:FC1_OUT_SIZE-1];    // 128 spikes from FC1 LIF

    // =========================================================================
    // PER-NEURON STATE ARRAYS (persist across timesteps)
    // =========================================================================
    reg signed [V_WIDTH-1:0] vmem_conv1 [0:CONV1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_conv1  [0:CONV1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_conv2 [0:CONV2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_conv2  [0:CONV2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_fc1   [0:FC1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_fc1    [0:FC1_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vmem_fc2   [0:FC2_OUT_SIZE-1];
    reg signed [V_WIDTH-1:0] vth_fc2    [0:FC2_OUT_SIZE-1];

    // Muxed state load values for the LIF
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
                vth_load_mux  = 16'd150;
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

    dynamic_gatekeeper #(
        .ID_W(10),
        .NUM_PRE(1024)
    ) gatekeeper_inst (
        .clk(clk),
        .rst_n(rst_n),
        .global_enable(sys_en),
        .spike_valid(spike_valid),
        .pre_id(spike_pre_id),
        .gate_keep(gate_keep),
        .gate_reason(gate_reason),
        .mem_en(mem_en_gate),
        .mac_en(mac_en_gate)
    );

    // =========================================================================
    // INSTANCES: Quantized SRAM Banks
    // =========================================================================
    wire signed [DATA_WIDTH-1:0] conv1_weight, conv2_weight, fc1_weight, fc2_weight;

    // SRAM read-enable: gatekeeper gates Layer 1 only; hidden layers
    // bypass the gatekeeper and read whenever sys_en is active.
    wire conv1_re = mem_en_gate && (layer_state == S_LAYER1);
    wire conv2_re = sys_en      && (layer_state == S_LAYER2);
    wire fc1_re   = sys_en      && (layer_state == S_LAYER3);
    wire fc2_re   = sys_en      && (layer_state == S_LAYER4);

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

    // Weight mux
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
    // Spike input mux: gatekeeper for L1, spike buffer for hidden layers
    // =========================================================================
    reg active_spike;
    always @(*) begin
        case (layer_state)
            S_LAYER1: active_spike = spike_valid & gate_keep;
            S_LAYER2: active_spike = spike_buf_pool1[synapse_idx];
            S_LAYER3: active_spike = spike_buf_pool2[synapse_idx];
            S_LAYER4: active_spike = spike_buf_fc1[synapse_idx];
            default:  active_spike = 1'b0;
        endcase
    end

    // MAC enable: gatekeeper for L1, always-on for hidden layers
    wire mac_en_active = (layer_state == S_LAYER1) ? mac_en_gate : sys_en;

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
        .sys_en(mac_en_active),
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

    // Pool data input: read from the pre-pool spike buffer
    // For pooling, spikes are treated as 16-bit values (0 or 1 sign-extended)
    wire signed [V_WIDTH-1:0] pool1_data_in = spike_buf_conv1[pool_idx * 4 + pool_phase]
                                              ? 16'sd1 : 16'sd0;
    wire signed [V_WIDTH-1:0] pool2_data_in = spike_buf_conv2[pool_idx * 4 + pool_phase]
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
    // INSTANCE: Adaptive LIF Neuron (time-multiplexed)
    // =========================================================================
    wire lif_spike_out;
    wire signed [V_WIDTH-1:0] lif_vmem_out;
    wire signed [V_WIDTH-1:0] lif_vth_out;

    adaptive_lif #(.V_WIDTH(V_WIDTH)) lif_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sys_en(sys_en),
        .current_in(mac_current_out),
        .bias_in({V_WIDTH{1'b0}}),
        .base_vth(16'd150),
        .rho(16'd10),
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
        .class_spikes(output_layer_spikes),
        .sys_enable(sys_en),
        .done(done)
    );

    // =========================================================================
    // State array initialization index (shared for reset loops)
    // =========================================================================
    integer init_i;

    // =========================================================================
    // MAIN FSM CONTROLLER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            layer_state   <= S_IDLE;
            time_step_cnt <= 5'd0;
            neuron_idx    <= 16'd0;
            synapse_idx   <= 16'd0;
            cur_fanin     <= 16'd0;
            cur_neurons   <= 16'd0;
            mac_acc_clear <= 1'b0;
            lif_state_load <= 1'b0;
            conv1_addr    <= {CONV1_ADDR_W{1'b0}};
            conv2_addr    <= {CONV2_ADDR_W{1'b0}};
            fc1_addr      <= {FC1_ADDR_W{1'b0}};
            fc2_addr      <= {FC2_ADDR_W{1'b0}};
            pool_idx      <= 16'd0;
            pool_total    <= 16'd0;
            pool_phase    <= 2'd0;

            // Zero all state arrays
            for (init_i = 0; init_i < CONV1_OUT_SIZE; init_i = init_i + 1) begin
                vmem_conv1[init_i] <= {V_WIDTH{1'b0}};
                vth_conv1[init_i]  <= 16'd150;
                spike_buf_conv1[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < POOL1_OUT_SIZE; init_i = init_i + 1) begin
                spike_buf_pool1[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < CONV2_OUT_SIZE; init_i = init_i + 1) begin
                vmem_conv2[init_i] <= {V_WIDTH{1'b0}};
                vth_conv2[init_i]  <= 16'd150;
                spike_buf_conv2[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < POOL2_OUT_SIZE; init_i = init_i + 1) begin
                spike_buf_pool2[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < FC1_OUT_SIZE; init_i = init_i + 1) begin
                vmem_fc1[init_i] <= {V_WIDTH{1'b0}};
                vth_fc1[init_i]  <= 16'd150;
                spike_buf_fc1[init_i] <= 1'b0;
            end
            for (init_i = 0; init_i < FC2_OUT_SIZE; init_i = init_i + 1) begin
                vmem_fc2[init_i] <= {V_WIDTH{1'b0}};
                vth_fc2[init_i]  <= 16'd150;
            end

        end else if (sys_en) begin
            // Defaults
            mac_acc_clear  <= 1'b0;
            lif_state_load <= 1'b0;

            case (layer_state)

                // =============================================================
                S_IDLE: begin
                    if (start || time_step_cnt > 0) begin
                        layer_state    <= S_LAYER1;
                        cur_fanin      <= CONV1_FANIN;
                        cur_neurons    <= CONV1_NEURONS;
                        neuron_idx     <= 16'd0;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;  // Load neuron 0's state
                    end
                end

                // =============================================================
                // LAYER 1: Conv1 — 32 neurons × 50 synapses
                // =============================================================
                S_LAYER1: begin
                    conv1_addr <= conv1_addr + 1;

                    if (neuron_done) begin
                        // MAC done for this neuron; let LIF fire next cycle
                        layer_state <= S_L1_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L1_WRITE: begin
                    // Write back LIF result: spike and updated v_mem/v_th
                    spike_buf_conv1[neuron_idx] <= lif_spike_out;
                    vmem_conv1[neuron_idx]      <= lif_vmem_out;
                    vth_conv1[neuron_idx]       <= lif_vth_out;

                    if (neuron_idx == cur_neurons - 1) begin
                        // All Conv1 neurons done → go to pooling
                        layer_state <= S_POOL1;
                        pool_idx    <= 16'd0;
                        pool_total  <= POOL1_OUT_SIZE;
                        pool_phase  <= 2'd0;
                    end else begin
                        // Next neuron: clear MAC, load next neuron's state
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER1;
                    end
                end

                // =============================================================
                // POOL 1: 32×28×28 → 32×14×14  (6272 output values)
                // =============================================================
                S_POOL1: begin
                    pool_phase <= pool_phase + 1;

                    if (pool_phase == 2'd3) begin
                        // avg_pool outputs on this cycle (pool1_valid = 1)
                        // Store result into pooled spike buffer
                        // (threshold: if average >= 0.5 → spike)
                        spike_buf_pool1[pool_idx] <= (pool1_out > 0) ? 1'b1 : 1'b0;

                        if (pool_idx == pool_total - 1) begin
                            // Pooling done → Layer 2
                            layer_state    <= S_LAYER2;
                            cur_fanin      <= CONV2_FANIN;
                            cur_neurons    <= CONV2_NEURONS;
                            neuron_idx     <= 16'd0;
                            synapse_idx    <= 16'd0;
                            mac_acc_clear  <= 1'b1;
                            lif_state_load <= 1'b1;
                        end else begin
                            pool_idx   <= pool_idx + 1;
                            pool_phase <= 2'd0;
                        end
                    end
                end

                // =============================================================
                // LAYER 2: Conv2 — 64 neurons × 800 synapses
                // =============================================================
                S_LAYER2: begin
                    conv2_addr <= conv2_addr + 1;

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

                    if (neuron_idx == cur_neurons - 1) begin
                        layer_state <= S_POOL2;
                        pool_idx    <= 16'd0;
                        pool_total  <= POOL2_OUT_SIZE;
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
                // POOL 2: 64×14×14 → 64×7×7  (3136 output values)
                // =============================================================
                S_POOL2: begin
                    pool_phase <= pool_phase + 1;

                    if (pool_phase == 2'd3) begin
                        spike_buf_pool2[pool_idx] <= (pool2_out > 0) ? 1'b1 : 1'b0;

                        if (pool_idx == pool_total - 1) begin
                            layer_state    <= S_LAYER3;
                            cur_fanin      <= FC1_FANIN;
                            cur_neurons    <= FC1_NEURONS;
                            neuron_idx     <= 16'd0;
                            synapse_idx    <= 16'd0;
                            mac_acc_clear  <= 1'b1;
                            lif_state_load <= 1'b1;
                        end else begin
                            pool_idx   <= pool_idx + 1;
                            pool_phase <= 2'd0;
                        end
                    end
                end

                // =============================================================
                // LAYER 3: FC1 — 128 neurons × 3136 synapses
                // =============================================================
                S_LAYER3: begin
                    fc1_addr <= fc1_addr + 1;

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

                    if (neuron_idx == cur_neurons - 1) begin
                        layer_state    <= S_LAYER4;
                        cur_fanin      <= FC2_FANIN;
                        cur_neurons    <= FC2_NEURONS;
                        neuron_idx     <= 16'd0;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                    end else begin
                        neuron_idx     <= neuron_idx + 1;
                        synapse_idx    <= 16'd0;
                        mac_acc_clear  <= 1'b1;
                        lif_state_load <= 1'b1;
                        layer_state    <= S_LAYER3;
                    end
                end

                // =============================================================
                // LAYER 4: FC2 — 10 neurons × 128 synapses
                // =============================================================
                S_LAYER4: begin
                    fc2_addr <= fc2_addr + 1;

                    if (neuron_done) begin
                        layer_state <= S_L4_WRITE;
                    end else begin
                        synapse_idx <= synapse_idx + 1;
                    end
                end

                S_L4_WRITE: begin
                    vmem_fc2[neuron_idx] <= lif_vmem_out;
                    vth_fc2[neuron_idx]  <= lif_vth_out;
                    // output_layer_spikes is handled in its own always block

                    if (neuron_idx == cur_neurons - 1) begin
                        // All layers done for this timestep
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
                // TIMESTEP: End-of-timestep bookkeeping
                // =============================================================
                S_TIMESTEP: begin
                    time_step_cnt <= time_step_cnt + 1;
                    // Reset SRAM addresses for next timestep
                    conv1_addr <= {CONV1_ADDR_W{1'b0}};
                    conv2_addr <= {CONV2_ADDR_W{1'b0}};
                    fc1_addr   <= {FC1_ADDR_W{1'b0}};
                    fc2_addr   <= {FC2_ADDR_W{1'b0}};
                    // Go back to IDLE (will start next timestep if sys_en)
                    layer_state <= S_IDLE;
                end

                // =============================================================
                S_DONE: begin
                    layer_state <= S_DONE;
                end

                default: layer_state <= S_IDLE;
            endcase
        end
    end

endmodule

`timescale 1ns / 1ps

module snn_tile #(
    parameter TILE_ID = 8'd1,
    parameter NEURON_COUNT = 256,
    parameter WEIGHT_WIDTH = 8,
    parameter ADDR_WIDTH = 11
)(
    input wire clk,
    input wire rst_n,
    input wire timestep_tick,
    
    // Input interface (from NoC or Ping-Pong Buffer)
    input wire rx_spike_valid,
    input wire [ADDR_WIDTH-1:0] rx_spike_id,
    
    // Output interface (to NoC or Ping-Pong Buffer)
    output wire tx_spike_valid,
    output wire [ADDR_WIDTH-1:0] tx_spike_id
);

    // 1. Local SRAM Bank (Weights)
    // In practice, this outputs a wide bus containing weights for all neurons in this tile.
    wire [1023:0] sram_weight_bus; 
    
    sram_weight_bank #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .BUS_WIDTH(1024)
    ) local_memory (
        .clk(clk),
        .read_en(rx_spike_valid),
        .read_addr(rx_spike_id),
        .read_data(sram_weight_bus)
    );
    
    // 2. Sparse MAC Unit (Adder Tree)
    // Accumulates the read weights into the target membrane potentials
    wire [15:0] mac_updates [0:NEURON_COUNT-1];
    
    sparse_mac_array #(
        .NEURON_COUNT(NEURON_COUNT),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) mac_unit (
        .clk(clk),
        .enable(rx_spike_valid),
        .weight_bus(sram_weight_bus),
        .v_updates(mac_updates)
    );
    
    // 3. LIF Neuron Array
    // Leaks, Integrates, and Fires
    wire [NEURON_COUNT-1:0] fire_vector;
    
    lif_neuron_array #(
        .NEURON_COUNT(NEURON_COUNT)
    ) lif_array (
        .clk(clk),
        .rst_n(rst_n),
        .timestep_tick(timestep_tick),
        .v_updates(mac_updates),
        .fire_out(fire_vector)
    );
    
    // 4. Output Router / Encoder
    // Converts the parallel fire vector into sequential spike packets
    spike_encoder #(
        .NEURON_COUNT(NEURON_COUNT),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) output_router (
        .clk(clk),
        .rst_n(rst_n),
        .fire_vector(fire_vector),
        .tx_valid(tx_spike_valid),
        .tx_spike_id(tx_spike_id)
    );

endmodule

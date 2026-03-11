`timescale 1ns / 1ps

module ping_pong_spike_buffer #(
    parameter ADDR_WIDTH = 11,    // 11 bits can address up to 2048 neurons
    parameter BUFFER_DEPTH = 256  // Max spikes per timestep per tile
)(
    input wire clk,
    input wire rst_n,
    
    // Global synchronization
    input wire timestep_tick,     // Toggles ping-pong state (T -> T+1)
    
    //---------------------------------------------------------
    // WRITE PORT (From Upstream Layer, e.g. Conv1 Tile)
    //---------------------------------------------------------
    input wire write_en,
    input wire [ADDR_WIDTH-1:0] write_spike_id,
    output wire buffer_full,
    
    //---------------------------------------------------------
    // READ PORT (To Downstream Layer, e.g. Conv2 Tile SRAM)
    //---------------------------------------------------------
    input wire read_en,
    output reg [ADDR_WIDTH-1:0] read_spike_id,
    output wire buffer_empty
);

    // Two physical memory arrays for Ping-Pong operation
    reg [ADDR_WIDTH-1:0] mem_A [0:BUFFER_DEPTH-1];
    reg [ADDR_WIDTH-1:0] mem_B [0:BUFFER_DEPTH-1];
    
    // Pointers for Memory A
    reg [7:0] head_A; // Write pointer
    reg [7:0] tail_A; // Read pointer
    
    // Pointers for Memory B
    reg [7:0] head_B; // Write pointer
    reg [7:0] tail_B; // Read pointer
    
    // State toggle: 0 = Write A / Read B, 1 = Write B / Read A
    reg active_bank; 

    //---------------------------------------------------------
    // TIMESTEP TOGGLE & POINTER RESET LOGIC
    //---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
            head_A <= 8'd0; tail_A <= 8'd0;
            head_B <= 8'd0; tail_B <= 8'd0;
        end else if (timestep_tick) begin
            // Toggle the active bank
            active_bank <= ~active_bank;
            
            // Reset the pointers for the bank that is ABOUT TO BE WRITTEN to
            if (active_bank == 1'b0) begin
                // Transitioning to Write B, Read A
                head_B <= 8'd0;
                // Leave tail_A alone so it can be read from 0 to head_A
                tail_A <= 8'd0; 
            end else begin
                // Transitioning to Write A, Read B
                head_A <= 8'd0;
                // Leave tail_B alone so it can be read from 0 to head_B
                tail_B <= 8'd0;
            end
        end else begin
            //---------------------------------------------------------
            // WRITE LOGIC (Routing based on active_bank)
            //---------------------------------------------------------
            if (write_en) begin
                if (active_bank == 1'b0 && !buffer_full) begin // Write to A
                    mem_A[head_A] <= write_spike_id;
                    head_A <= head_A + 1;
                end else if (active_bank == 1'b1 && !buffer_full) begin // Write to B
                    mem_B[head_B] <= write_spike_id;
                    head_B <= head_B + 1;
                end
            end
            
            //---------------------------------------------------------
            // READ LOGIC (Routing based on active_bank)
            //---------------------------------------------------------
            if (read_en && !buffer_empty) begin
                if (active_bank == 1'b0) begin // Read from B
                    read_spike_id <= mem_B[tail_B];
                    tail_B <= tail_B + 1;
                end else begin // Read from A
                    read_spike_id <= mem_A[tail_A];
                    tail_A <= tail_A + 1;
                end
            end
        end
    end

    //---------------------------------------------------------
    // STATUS FLAGS (Combinational routing)
    //---------------------------------------------------------
    assign buffer_full = (active_bank == 1'b0) ? (head_A == BUFFER_DEPTH) : (head_B == BUFFER_DEPTH);
    assign buffer_empty = (active_bank == 1'b0) ? (tail_B == head_B) : (tail_A == head_A);

endmodule

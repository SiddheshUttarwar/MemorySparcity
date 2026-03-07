// Auto-generated Verilog weight loading snippet
// Include this in your testbench or top-level initial block

initial begin
    $readmemh("conv1_weights.mem", conv1_sram.mem);
    $readmemh("conv2_weights.mem", conv2_sram.mem);
    $readmemh("fc1_weights.mem", fc1_sram.mem);
    $readmemh("fc2_weights.mem", fc2_sram.mem);
end

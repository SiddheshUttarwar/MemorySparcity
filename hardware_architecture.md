# Neuromorphic Hardware Architecture: CSNN Inference Engine to Verilog

This document provides directly translatable hardware block diagrams (RTL-level concepts) for both the Baseline Convolutional Spiking Neural Network (CSNN) and the Sparsity-Optimized CSNN (Sparse-SNN). 

**Important Hardware Note:** This architecture describes a **pure inference engine**. Training occurs entirely off-chip (e.g., in PyTorch on a GPU). Once trained, the stationary INT8 weights are permanently locked into the SRAM instances. The hardware is designed *solely* for forward-pass execution, heavily optimized to physically trigger SRAM read enables *only* when an incoming spike explicitly demands it, guaranteeing extreme memory sparsity.

---

## 1. Baseline CSNN Hardware Architecture

The Baseline model relies on synchronous iterations over fixed time steps ($T=20$). Every layer strictly reads from its dedicated SRAM block, integrates the membrane potential ($V$), and computes a binary step function.

```mermaid
graph TD
    %% Global Control
    CLK[System Clock] --> CU[Global FSM / Control Unit]
    CU --> |T=1 to 20| IB[Input Spike Buffer]
    
    %% Memory Blocks (Inference Only)
    subgraph Memory_Hierarchy ["Memory Hierarchy (Trained Weights)"]
        SRAM1[(SRAM: Conv1 Weights)]
        SRAM2[(SRAM: Conv2 Weights)]
        SRAM3[(SRAM: FC1 Weights)]
        SRAM4[(SRAM: FC2 Weights)]
    end

    %% Layer 1 Execution
    subgraph Layer_1 ["Layer 1: Conv + LIF"]
        MAC1[MAC Array]
        V_MEM1[(Registers: Mem. Potential V)]
        LIF_GEN1[Spike Generator <br> if V > V_th, S=1, V=V-V_th]
    end

    %% Pipeline Flow
    IB --> |Input Spikes| MAC1
    SRAM1 --> |Weight Fetch| MAC1
    MAC1 --> |Current I| V_MEM1
    V_MEM1 --> |Voltage| LIF_GEN1
    
    %% Layer 2 Execution
    subgraph Layer_2 ["Layer 2: Pool + Conv + LIF"]
        POOL1[Avg Pool / Subsampler]
        MAC2[MAC Array]
        V_MEM2[(Registers: Mem. Potential V)]
        LIF_GEN2[Spike Generator]
    end

    LIF_GEN1 --> POOL1
    POOL1 --> MAC2
    SRAM2 --> MAC2
    MAC2 --> V_MEM2
    V_MEM2 --> LIF_GEN2
    
    %% Readout Buffer
    OUT_ACC[(Output Spike Accumulator)]
    LIF_GEN2 -.-> |...FC Layers...| OUT_ACC
    CU --> |T=20 Reached| OUT_EVAL[ArgMax Evaluator]
    OUT_ACC --> OUT_EVAL
```

### Verilog Submodule Definitions (Baseline):
*   **SRAM**: Standard IP core with `ADDR`, `WE` (Write Enable), and `DATA_OUT`. Read Enables are synchronized heavily with the MAC Array.
*   **V_MEM**: Array of D-Flip Flops storing `V_m` across $T$ steps, retaining state between clock cycles.
*   **Spike Generator**: A simple Verilog `always` block comparator: `assign spike = (V_m > V_th) ? 1'b1 : 1'b0;` followed by `if (spike) V_nxt = V_m - V_th;`.

---

## 2. Sparsity-Optimized SNN (Sparse-SNN) Architecture

The Sparse-SNN requires dynamic control logic. It introduces an **Adaptive Thresholding ALU** inside every LIF Core, **INT8 Arithmetic** to limit bus width, and a **Global Early-Exit FSM** to forcefully shut down the system clock before $T=20$ if an answer is confidently reached.

```mermaid
graph TD
    %% Advanced Global Control
    CLK[System Clock] --> CU_EE[Dynamic Control Unit + Early Exit FSM]
    CU_EE --> |Enable| IB[Input Spike Buffer]
    
    %% Sparse Memory Blocks (INT8)
    subgraph Quantized_SRAM ["Quantized SRAM (8-bit Data Bus)"]
        SRAM_Q1[(SRAM Conv1 INT8)]
        SRAM_Q2[(SRAM Conv2 INT8)]
    end

    %% Sparse Layer Execution
    subgraph Sparse_Layer ["Sparse Layer: LIF with Adaptive V_th"]
        MAC_Q[INT8 Sparse MAC Array <br> Triggered ONLY on Spike In]
        V_MEM_Q[(Membrane Reg V)]
        VTH_MEM_Q[(Threshold Reg V_th)]
        
        ALU_V[V Integrator = V*Beta + I]
        ALU_VTH[Adaptive V_th ALU <br> V_th = V_th + rho*S]
        
        CMP[Comparator: S = V > V_th]
        RST[Soft Reset: V = V - S*V_th]
    end

    %% Pipeline Flow
    IB --> |S_in = 1| MAC_Q
    SRAM_Q1 --> |INT8 Read| MAC_Q
    MAC_Q --> ALU_V
    
    ALU_V --> V_MEM_Q
    V_MEM_Q --> CMP
    VTH_MEM_Q --> CMP
    
    CMP --> |Spike Generated S| RST
    CMP --> |Spike Generated S| ALU_VTH
    ALU_VTH --> VTH_MEM_Q
    RST --> V_MEM_Q
    
    %% Early Exit Logic
    subgraph Output_Classification_Stage ["Output Classification Stage"]
        OUT_ACCUM[Spike Confidence Integrator]
        PROB_CHK[Threshold Checker <br> Max Prob > 0.9]
    end
    
    CMP --> |Output Spikes| OUT_ACCUM
    OUT_ACCUM --> PROB_CHK
    
    %% Early Exit Disconnect
    PROB_CHK --> |Confidence Reached!| CU_EE
    CU_EE --> |SHUTDOWN CLOCK| MAC_Q
```

### Verilog Submodule Updates (Sparse):
1.  **Sparse MAC Array (Triggered Reads)**:
    *   In Verilog, wrap the SRAM `Read_Enable` pin with `if (Spike_In == 1'b1)`. If no spike arrives from the previous layer, the SRAM address is never pulsed, saving massive dynamic power.
    *   Arithmetic shifts from floating-point DSP slices to simple **INT8 Adders/Multipliers** (`signed [7:0]`).
2.  **Adaptive V_th Registers**:
    *   Instead of hardcoding `V_th = 200` globally, each neuron requires its own register.
    *   Logic: `always @(posedge clk) begin if (spike_out) V_th <= V_th + RHO_CONST; else V_th <= V_th - DECAY; end`
3.  **Early Exit FSM**:
    *   The `Control Unit` tracks `Time_Step`. If the `Output_Accumulator` signals the `Confidence_Flag`, the global FSM jumps to the `IDLE/DONE` state immediately, cutting off the `Enable` signal to all Memory and MAC blocks to freeze power consumption.

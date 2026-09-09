# VectorCore Compiler — MLIR Dialect Skeleton

> **Status:** This directory is a *skeleton for a future native C++ MLIR
> dialect build.* The current shipping path is the **textual MLIR** layer
> under `src/vector_core_sim/mlir/` (Python emitter + parser + verifier).
> The skeletons here describe the C++ ODS / CMake / `vc-opt` structure that
> the textual layer is forward-compatible with, so that a future native build
> can drop in without changing the IR contract.

The textual layer emits and parses MLIR text of the form:

```mlir
// VectorCore MLIR (vcworkload dialect), schema_version: 1.0

func.func @attention_block(
    %q : tensor<1x8x128x32xbf16>,
    %k : tensor<1x8x128x32xbf16>,
    %v : tensor<1x8x128x32xbf16>
) -> tensor<1x8x128x32xbf16> {
  %out = vcworkload.attention %q, %k, %v {
    causal = true, scale = 0.176000,
    query_tile = 32 : i64, key_tile = 32 : i64
  } : (tensor<1x8x128x32xbf16>, tensor<1x8x128x32xbf16>, tensor<1x8x128x32xbf16>)
      -> tensor<1x8x128x32xbf16>
  vcworkload.return %out : tensor<1x8x128x32xbf16>
}
```

---

## 1. `vcworkload` Dialect (high-level workload IR)

Namespace: `vcworkload`
Parent dialect: `func` (uses `func.func` for function containers).

### ODS skeleton — `vcworkload.td`

```tablegen
#ifndef VCWORKLOAD_OPS
#define VCWORKLOAD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"

def VCWorkload_Dialect : Dialect {
  let name = "vcworkload";
  let summary = "VectorCore high-level workload dialect";
  let description = [{
    Models the workload-level ops (matmul, attention, softmax, rmsnorm,
    elementwise, ...) that sit between PyTorch export and the cycle simulator.
    Lowered to the `vcmapping` dialect by the mapper pass.
  }];
  let cppNamespace = "::vector_core_sim::vcworkload";
}

// Common base class for vcworkload ops.
class VCWorkload_Op<string mnemonic, list<Trait> traits = []> :
    Op<VCWorkload_Dialect, mnemonic, traits>;

// ---- Op: matmul ----
def VCWorkload_MatMulOp : VCWorkload_Op<"matmul",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Matrix multiply, optional transpose_b";
  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    BoolAttr:$transpose_b
  );
  let results = (outs AnyTensor:$c);
  let assemblyFormat = "$a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($c)";
}

// ---- Op: attention ----
def VCWorkload_AttentionOp : VCWorkload_Op<"attention", [Pure]> {
  let summary = "Flash-style scaled-dot-product attention";
  let arguments = (ins
    AnyTensor:$q,
    AnyTensor:$k,
    AnyTensor:$v,
    BoolAttr:$causal,
    F64Attr:$scale,
    I64Attr:$query_tile,
    I64Attr:$key_tile
  );
  let results = (outs AnyTensor:$out);
  let assemblyFormat =
    "$q `,` $k `,` $v attr-dict `:` "
    "`(` type($q) `,` type($k) `,` type($v) `)` `->` type($out)";
}

// ---- Op: softmax ----
def VCWorkload_SoftmaxOp : VCWorkload_Op<"softmax", [Pure]> {
  let arguments = (ins AnyTensor:$x, I64Attr:$axis);
  let results = (outs AnyTensor:$out);
}

// ---- Op: rmsnorm ----
def VCWorkload_RMSNormOp : VCWorkload_Op<"rmsnorm", [Pure]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$gamma, F32Attr:$eps);
  let results = (outs AnyTensor:$out);
}

// ---- Op: elementwise ----
def VCWorkload_ElementwiseOp : VCWorkload_Op<"elementwise", [Pure]> {
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    StrAttr:$kind            // "add" | "mul" | "relu" | "gelu" | ...
  );
  let results = (outs AnyTensor:$out);
}

// ---- Op: return ----
def VCWorkload_ReturnOp : VCWorkload_Op<"return",
    [Terminator, HasParent<"FuncOp">]> {
  let arguments = (ins Variadic<AnyTensor>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

#endif // VCWORKLOAD_OPS
```

### Op summary

| Op                  | Operands                   | Results | Key attrs                                   |
|---------------------|----------------------------|---------|---------------------------------------------|
| `vcworkload.matmul` | `%a, %b`                   | `%c`    | `transpose_b: bool`                        |
| `vcworkload.attention` | `%q, %k, %v`            | `%out`  | `causal, scale, query_tile, key_tile`      |
| `vcworkload.softmax`   | `%x`                    | `%out`  | `axis: i64`                                 |
| `vcworkload.rmsnorm`   | `%x, %gamma`            | `%out`  | `eps: f32`                                  |
| `vcworkload.elementwise` | variadic `%inputs`    | `%out`  | `kind: str`                                 |
| `vcworkload.return`  | variadic                  | —       | —                                           |

---

## 2. `vcmapping` Dialect (hardware-mapped IR)

Namespace: `vcmapping`
Parent dialect: standalone (uses its own `vcmapping.func` or plain region).

This dialect is produced by the **mapper pass** `vcworkload-to-vcmapping`,
which tiles each workload op onto the cube/SIMD/SIMT clusters, allocates
unified-buffer scratch, and inserts DMAs, double-buffering and barriers.

### ODS skeleton — `vcmapping.td`

```tablegen
#ifndef VCMAPPING_OPS
#define VCMAPPING_OPS

include "mlir/IR/OpBase.td"

def VCMapping_Dialect : Dialect {
  let name = "vcmapping";
  let summary = "VectorCore hardware-mapped dialect";
  let description = [{
    Models ops after hardware mapping: UB allocations, DMAs, cube GEMM
    regions, SIMD/SIMT regions, double-buffer, barriers and vector-function
    launches with explicit dependency edges.
  }];
  let cppNamespace = "::vector_core_sim::vcmapping";
}

class VCMapping_Op<string mnemonic, list<Trait> traits = []> :
    Op<VCMapping_Dialect, mnemonic, traits>;

// ---- alloc_ub: allocate a scratch tensor in the unified buffer ----
def VCMapping_AllocUbOp : VCMapping_Op<"alloc_ub", [Pure]> {
  let arguments = (ins I64Attr:$bytes, StrAttr:$bank_policy);
  let results = (outs AnyRankedTensor:$ref);
}

// ---- dma: async DMA copy (Ub<->DRAM or UB<->UB) ----
def VCMapping_DmaOp : VCMapping_Op<"dma"> {
  let arguments = (ins
    AnyTensor:$src,
    AnyTensor:$dst,
    I64Attr:$bytes,
    I64Attr:$dst_offset,
    I64Attr:$src_offset,
    StrAttr:$mode,             // "dram_to_ub" | "ub_to_dram" | "ub_to_ub"
    OptionalAttr<I64Attr>:$issue_cycle
  );
  let results = (outs AnyTensor:$done_token);
}

// ---- cube_gemm: matrix multiply on cube cluster ----
def VCMapping_CubeGemmOp : VCMapping_Op<"cube_gemm"> {
  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    AnyTensor:$c,
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    I64Attr:$m, I64Attr:$n, I64Attr:$k,
    StrAttr:$dtype,
    BoolAttr:$accumulate
  );
  let results = (outs AnyTensor:$c_out);
}

// ---- simd_region: region of elementwise SIMD work ----
def VCMapping_SimdRegionOp : VCMapping_Op<"simd_region",
    [RegionOpInterface, SingleBlockImplicitTerminator<"VCMapping_YieldOp">]> {
  let arguments = (ins Variadic<AnyTensor>:$inputs, StrAttr:$kind);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (ins Region:$body);
}

// ---- simt_region: region of SIMT (warp) work ----
def VCMapping_SimtRegionOp : VCMapping_Op<"simt_region",
    [RegionOpInterface, SingleBlockImplicitTerminator<"VCMapping_YieldOp">]> {
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$num_warps,
    I64Attr:$lane_width
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (ins Region:$body);
}

// ---- double_buffer: marks an iteration variable as double-buffered ----
def VCMapping_DoubleBufferOp : VCMapping_Op<"double_buffer",
    [RegionOpInterface]> {
  let arguments = (ins
    Variadic<AnyTensor>:$buffers,
    I64Attr:$depth,           // typically 2
    BoolAttr:$overlap_compute
  );
  let results = (outs Variadic<AnyTensor>:$out_buffers);
  let regions = (ins Region:$body);
}

// ---- barrier: execution barrier (producer/consumer sync) ----
def VCMapping_BarrierOp : VCMapping_Op<"barrier", [HasParent<"FuncOp">]> {
  let arguments = (ins
    Variadic<AnyTensor>:$wait_tokens,
    I64Attr:$barrier_id,
    I64Attr:$arrival_count
  );
  let results = (outs AnyTensor:$release_token);
}

// ---- vf: vector-function kernel launch ----
def VCMapping_VfOp : VCMapping_Op<"vf",
    [HasParent<"FuncOp">]> {
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    StrAttr:$kernel,           // e.g. "layernorm_fused"
    I64Attr:$m, I64Attr:$n,
    StrAttr:$dtype
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
}

// ---- dependency: explicit RAW/WAR/WAW dependency edge ----
def VCMapping_DependencyOp : VCMapping_Op<"dependency"> {
  let arguments = (ins
    Variadic<AnyTensor>:$from,
    Variadic<AnyTensor>:$to,
    StrAttr:$kind               // "raw" | "war" | "waw" | "rar"
  );
  let assemblyFormat = "$from `->` $to attr-dict";
}

// ---- return ----
def VCMapping_ReturnOp : VCMapping_Op<"return",
    [Terminator, HasParent<"FuncOp">]> {
  let arguments = (ins Variadic<AnyTensor>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def VCMapping_YieldOp : VCMapping_Op<"yield",
    [Terminator, HasParent<"VCMapping_SimdRegionOp", "VCMapping_SimtRegionOp",
                            "VCMapping_DoubleBufferOp">]> {
  let arguments = (ins Variadic<AnyTensor>:$operands);
}

#endif // VCMAPPING_OPS
```

### vcmapping op summary

| Op                      | Role                                                    |
|-------------------------|---------------------------------------------------------|
| `vcmapping.alloc_ub`    | scratch allocation in unified buffer                    |
| `vcmapping.dma`         | async DMA copy, returns a `done_token`                 |
| `vcmapping.cube_gemm`   | matrix multiply on the cube cluster                    |
| `vcmapping.simd_region` | region of elementwise SIMD work                        |
| `vcmapping.simt_region` | region of SIMT (warp) work                              |
| `vcmapping.double_buffer` | double-buffered iteration region                      |
| `vcmapping.barrier`     | producer/consumer sync, returns a `release_token`       |
| `vcmapping.vf`          | vector-function kernel launch                           |
| `vcmapping.dependency`  | explicit RAW/WAR/WAW dependency edge                    |
| `vcmapping.return`      | function terminator                                     |
| `vcmapping.yield`       | region terminator                                       |

---

## 3. CMakeLists.txt structure

```
compiler/
├── CMakeLists.txt
├── include/
│   └── vector_core_sim/
│       ├── vcworkload/
│       │   ├── Dialect.h
│       │   ├── Ops.h
│       │   └── Types.h
│       └── vcmapping/
│           ├── Dialect.h
│           └── Ops.h
└── lib/
    ├── vcworkload/
    │   ├── Dialect.cpp
    │   ├── Ops.cpp
    │   └── Types.cpp
    ├── vcmapping/
    │   ├── Dialect.cpp
    │   └── Ops.cpp
    ├── Conversion/
    │   └── VCWorkloadToVCMapping.cpp
    └── Pass/
        └── Mapper.cpp
└── tools/
    └── vc-opt/
        └── vc-opt.cpp
```

### `CMakeLists.txt` (skeleton)

```cmake
cmake_minimum_required(VERSION 3.20)
project(vector_core_sim_compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddMLIR)
include(TableGen)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- vcworkload dialect ---
add_mlir_dialect_library(VCWorkloadOps
  lib/vcworkload/Dialect.cpp
  lib/vcworkload/Ops.cpp
  lib/vcworkload/Types.cpp
  DEPENDS
    VCWorkloadOpsIncGen
)

mlir_tablegen(VCWorkloadOps.h.inc    -gen-op-decls     -I${CMAKE_CURRENT_SOURCE_DIR}/include VCWorkloadOps.td)
mlir_tablegen(VCWorkloadOps.cpp.inc -gen-op-defs      -I${CMAKE_CURRENT_SOURCE_DIR}/include VCWorkloadOps.td)
add_public_tablegen_target(VCWorkloadOpsIncGen)

# --- vcmapping dialect ---
add_mlir_dialect_library(VCMappingOps
  lib/vcmapping/Dialect.cpp
  lib/vcmapping/Ops.cpp
  DEPENDS
    VCMappingOpsIncGen
)

mlir_tablegen(VCMappingOps.h.inc    -gen-op-decls -I${CMAKE_CURRENT_SOURCE_DIR}/include VCMappingOps.td)
mlir_tablegen(VCMappingOps.cpp.inc -gen-op-defs  -I${CMAKE_CURRENT_SOURCE_DIR}/include VCMappingOps.td)
add_public_tablegen_target(VCMappingOpsIncGen)

# --- conversion pass ---
add_mlir_library(VCWorkloadToVCMapping
  lib/Conversion/VCWorkloadToVCMapping.cpp
  lib/Pass/Mapper.cpp
  LINK_LIBS PUBLIC
    VCWorkloadOps VCMappingOps
    MLIRIR MLIRPass
)

# --- vc-opt tool ---
add_executable(vc-opt tools/vc-opt/vc-opt.cpp)
target_link_libraries(vc-opt PRIVATE
  VCWorkloadOps VCMappingOps
  VCWorkloadToVCMapping
  MLIRIR MLIRParser MLIRPass MLIRSupport
)
```

---

## 4. `vc-opt` tool skeleton

```cpp
// tools/vc-opt/vc-opt.cpp
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "vector_core_sim/vcworkload/Dialect.h"
#include "vector_core_sim/vcworkload/Ops.h"
#include "vector_core_sim/vcmapping/Dialect.h"
#include "vector_core_sim/vcmapping/Ops.h"

using namespace vector_core_sim;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<vcworkload::VCWorkloadDialect>();
  registry.insert<vcmapping::VCMappingDialect>();
  // Pull in the func dialect for function containers.
  registry.insert<mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VectorCore optimizer\n",
                        registry));
}
```

### Pass registration (skeleton)

```cpp
// lib/Pass/Mapper.cpp
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vector_core_sim/vcworkload/Ops.h"
#include "vector_core_sim/vcmapping/Ops.h"

using namespace mlir;

namespace {
struct VCWorkloadToVCMappingPass
    : PassWrapper<VCWorkloadToVCMappingPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
}  // namespace

void VCWorkloadToVCMappingPass::runOnOperation() {
  // For each vcworkload.matmul: emit vcmapping.alloc_ub + dma + cube_gemm.
  // For each vcworkload.attention: tiled loop nest of cube_gemm + simd_region.
  // For each pair of dependent tiles: vcmapping.dependency + barrier.
}

void registerMapperPass() {
  PassRegistration<VCWorkloadToVCMappingPass>(
      "vcworkload-to-vcmapping",
      "Map high-level workload ops to hardware-mapped vcmapping ops");
}
```

---

## 5. Migration path

1. **Today** — `src/vector_core_sim/mlir/` provides `emit_mlir`, `parse_mlir`,
   `verify_mlir`, `roundtrip`. The simulator consumes the parsed dict.
2. **Next** — generate the C++ dialects from the ODS in this file, build
   `vc-opt`, and round-trip the *same textual MLIR* through `vc-opt --mlir-print-ir`.
   The textual contract (op names, attr spellings, type syntax) is already
   aligned with the ODS above.
3. **Later** — add the `vcworkload-to-vcmapping` conversion pass and replace
   the Python mapper with the C++ pass output.

---

*Generated 2026-08-04. Maintained alongside
`src/vector_core_sim/mlir/`.*

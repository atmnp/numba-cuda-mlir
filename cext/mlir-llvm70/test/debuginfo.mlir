// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_80">] {
    llvm.func @debug_kernel(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %arg1, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @debug_kernel2(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c = llvm.mlir.constant(42 : i32) : i32
      llvm.store %c, %arg0 : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// Function definitions come first in the IR — verify !dbg on instructions.
// CHECK-IR: define ptx_kernel void @debug_kernel(
// CHECK-IR: call i32 asm "mov.u32 $0, %tid.x;", "=r"(){{.*}}!dbg
// CHECK-IR: store i32{{.*}}!dbg

// CHECK-IR: define ptx_kernel void @debug_kernel2(
// CHECK-IR: store i32 42{{.*}}!dbg

// Verify Debug Info Version uses Error behavior so nvJitLink can merge it with
// NVRTC-generated CUDA-source LTOIR.
// CHECK-IR-DAG: !llvm.module.flags = !{![[DBG_VERSION:[0-9]+]]}
// CHECK-IR-DAG: ![[DBG_VERSION]] = !{i32 1, !"Debug Info Version", i32 3}
// CHECK-IR-DAG: !nvvmir.version = !{![[NVVMIR_VERSION:[0-9]+]]}
// CHECK-IR-DAG: ![[NVVMIR_VERSION]] = !{i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 {{[0-9]+}}}

// Metadata comes after functions — verify compile unit, subprograms, and file.
// CHECK-IR-DAG: !DICompileUnit(language: DW_LANG_C,{{.*}}producer: "llvm7"{{.*}}isOptimized: false

// CHECK-IR-DAG: !DISubprogram(name: "debug_kernel",
// CHECK-IR-SAME: file: ![[FILE:[0-9]+]]
// CHECK-IR-SAME: isDefinition: true
// CHECK-IR-DAG: ![[FILE]] = !DIFile(filename: "debuginfo.mlir"

// CHECK-IR-DAG: !DISubprogram(name: "debug_kernel2",

// Verify PTX compiles successfully with debug info present in the IR.
// CHECK-PTX: .visible .entry debug_kernel
// CHECK-PTX: .visible .entry debug_kernel2

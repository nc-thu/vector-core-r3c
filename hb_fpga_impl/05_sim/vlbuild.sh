#!/usr/bin/env bash
# vlbuild.sh — Verilator 两步构建包装（2026-08-30 迁移新增）
# 背景：conda-forge Verilator 5.050 的 verilated.mk 把 CXX 硬编码到 conda 的
#   x86_64-conda-linux-gnu-c++（gcc 16.2），与服务器 Ubuntu 20.04/glibc 2.31 组合会撞
#   'timespec_get has not been declared'（conda gcc16 的 c++ <ctime> 与老 glibc 不兼容）。
# 做法：照常 verilator --binary 生成 obj_dir（其内嵌 make 失败没关系），然后用
#   系统 g++ 9.4 在 make 命令行覆盖 CXX 重跑链接，产物 obj_dir/tb_ae_v。
# 用法：bash vlbuild.sh <verilator 额外参数...>   （在含 tb_ae_v.sv 的目录里跑）
V=~/.conda/envs/vsim/bin
set -e
$V/verilator --binary --timing -j 0 -Wno-fatal "$@" > /tmp/vlbuild.log 2>&1 || true
if [ ! -f obj_dir/Vtb_ae_v.mk ]; then
  echo "vlbuild: Verilator 生成阶段失败（SV 错误），日志："; cat /tmp/vlbuild.log; exit 1
fi
# g++-10：系统默认 g++9 不认 verilated.mk 的 -fcoroutines；AR/LINK 同样绕开
#   conda 硬编码的工具链名（x86_64-conda-linux-gnu-ar / c++）
make -C obj_dir -f Vtb_ae_v.mk -j 32 CXX=/usr/bin/g++-10 AR=/usr/bin/ar \
     LINK=/usr/bin/g++-10 >> /tmp/vlbuild.log 2>&1 || {
  echo "vlbuild: 系统 g++-10 make 失败，日志尾部："; tail -30 /tmp/vlbuild.log; exit 1; }
echo "vlbuild: obj_dir/tb_ae_v 构建完成（系统 g++-10）"

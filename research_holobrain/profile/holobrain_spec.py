# -*- coding: utf-8 -*-
"""HB-GD (HoloBrain-0 GD 0.2B 变体, arXiv 2602.12062) 部署态推理工作负载规格模型。

口径与 swiftvla_spec.py 完全同源（Item(name,m,n,k,count,kind,causal,stage)、
MAC=乘加、FLOP=2*MAC、W 流=INT8 权重按次装载），维度全部对到本地核验过的
结构数字（robo_orchard_lab 源码 + hf_cfg/pretrain_model.config.json），非猜测：

  - 视觉主干 2D Swin-T 96/192/384/768, depths[2,2,6,2], heads[3,6,12,24], win7,
    patch4, 320x256 → 每视角 8.484 GMAC / 27.52M（含窗口 padding，padded token
    5880/1470/441/196）——四路调研+核验员逐 stage 复核通过
    (swin_transformer.py:614-618 corner padding, :661-687 整窗算完才裁剪)
  - depth 分支 (backbone_3d Swin 16/32/64/128 heads[4,8,8,16] + neck_3d +
    DepthFusionSpatialEnhancer) = 2.28M / 2.63 GMAC/视角，每次推理仅 1 次
    (structure.py:131-157, pre_spatial_enhancer=false)
  - BERT-base 12L/768/12h/3072, pad_to_max=false → L=指令实际长度，主档 16
    (bert.py:184-192 padding='longest')；每 chunk 只跑 1 次 (structure.py:131-133)
  - 特征增强 6 层 TextImageDeformable2DEnhancer：
      bi-attn 只作用于第 4 级 stride64 (每视角 20 token, 4head x dhead256,
      feature_enhancer.py:188-193 level_start_index[-1] 切的是 extra conv 输出)
      deformable img 块在全部 4 级 1700 token/视角 (MSDeformAttn h8/lv4/pt4
      = layers.py:89/:91 默认值；投影 4 条全进 GEMM 账，采样本身非 GEMM)
      text 自注意块 4head，每 chunk 一次
  - 动作专家 (HoloBrainActionDecoder, 论文 Table2 20.8M = 20.85M 对账差 0.3%)：
      每 chunk: robot_encoder 4 层 JointGraphAttention ×1；去噪 10 步
      DPMSolver++ 每步重复 input/t_embed/6 层 decoder/head（K/V 每步全量重算，
      无缓存——action_decoder.py:654-666 + layers.py:181-185，核验员确认）
      hoist_kv=True 变体：img/text K/V 投影 + 关节距离 ScalarEmbedder 提到
      循环外（数学等价，属软件优化）
  - token 结构：动作 token = N_j x 16 chunk (pred_steps=64 / chunk_size=4)；
    temp_joint KV = N_j x (1 hist + 16)；img_cross KV = 视角数 x 400
    (feature_level=[1,2] → stride16 320 + stride32 80)
  - LIBERO 主档（口径要求）：2 视图 (agentview + eye_in_hand) + N_j=8
    (Franka 7+1, envs/libero/env.py:310)，敏感度行给 3/4 视图与 N_j=14

非 GEMM 算子（deformable 采样、3D 反投影、图注意力 einsum、上采样插值等）
不进 gemm_items，量级估算在 holobrain_hw.py 的 not_modeled 里逐项列出。

输出: holobrain_spec.json (reference：忠实上游代码，K/V 每步重算) + 控制台表。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

# ----------------------------------------------------------------------------
# 结构常量（出处见模块 docstring；R=已核验的四路调研结论）
# ----------------------------------------------------------------------------
IMG_W, IMG_H, PATCH, WIN = 320, 256, 4, 7          # dst_wh=(320,256), patch4, win7
SW2_C, SW2_D, SW2_H = [96, 192, 384, 768], [2, 2, 6, 2], [3, 6, 12, 24]
SW3_C, SW3_D, SW3_H = [16, 32, 64, 128], [2, 2, 6, 2], [4, 8, 8, 16]
NECK2_IN, NECK3_IN, NECK_OUT = [192, 384, 768], [32, 64, 128], 256

BERT_H, BERT_HEADS, BERT_INTER, BERT_LAYERS = 768, 12, 3072, 12
ENH_LAYERS, ENH_D, ENH_INTER = 6, 256, 2048        # feature_enhancer.num_layers=6
BI_HEADS, BI_PROJ = 4, 1024                        # text_img_attn_block: 4h, 256->1024
TXT_HEADS = 4                                      # text_attn_block.self_attn 4h
DEF_NP = 128                                       # 8 heads x 4 levels x 4 points

DEPTH_BINS, PSE_FUSION_IN = 128, 512               # num_depth=128; 256+128+128=512

DEC_LAYERS, DEC_D, DEC_INTER, DEC_HEADS = 6, 256, 2048, 8
DENOISE_STEPS, PRED_STEPS, CHUNK_SIZE = 10, 64, 4
STATE_DIM = 8                                      # [关节角,x,y,z,qw,qx,qy,qz]
N_VIEWS, N_JOINTS, LANG_TOKENS = 2, 8, 16          # LIBERO 主档
HIST_CHUNKS = 1                                    # hist_steps=1
NUM_CHUNK = PRED_STEPS // CHUNK_SIZE               # 16


def ceil7(x: int) -> int:
    return int(math.ceil(x / WIN)) * WIN


def grids(img_w: int = IMG_W, img_h: int = IMG_H):
    """4 个 stage 的 (H,W) 网格：stride4/8/16/32。"""
    gw, gh = img_w // PATCH, img_h // PATCH
    return [(gh >> i, gw >> i) for i in range(4)]


@dataclass
class Item:
    """一条可独立映射到 Cube/Vector 的计算项（与 swiftvla_spec 同口径）。"""
    name: str
    m: int
    n: int = 1
    k: int = 1
    count: int = 1
    kind: str = "gemm"        # gemm | attn | elementwise
    causal: bool = False
    stage: str = ""

    @property
    def macs(self) -> int:
        if self.kind == "elementwise":
            return self.m * self.count
        f = 0.5 if self.causal else 1.0
        return self.m * self.n * self.k * self.count * f


def gemm(name, m, n, k, count=1, stage="") -> Item:
    return Item(name, m, n, k, count, "gemm", stage=stage)


def attn(name, m, n, k, count=1, causal=False, stage="") -> Item:
    """m=Lq, n=Lk, k=dhead*heads（头折进 k）；QK^T+PV 共 2 GEMM。"""
    return Item(name, 2 * m, n, k, count, "attn", causal=causal, stage=stage)


def elem(name, elems, count=1, stage="") -> Item:
    return Item(name, elems, 1, 1, count, "elementwise", stage=stage)


# ----------------------------------------------------------------------------
# Swin（2D 与 depth 共用；窗口注意力按 attn 条目逐窗等价计入）
# ----------------------------------------------------------------------------
def swin_items(cs, ds, hs, in_ch, g, views, prefix, img_wh=(IMG_W, IMG_H)):
    """一个 Swin 主干的 GEMM/attn 清单。count 已含 views。
    qkv/proj 用 padded token 数（corner padding 后整窗计算）、FFN/PM 用真实 token。"""
    T = [h * w for (h, w) in g]
    Tp = [ceil7(h) * ceil7(w) for (h, w) in g]
    NW = [ceil7(h) * ceil7(w) // WIN // WIN for (h, w) in g]
    items = [gemm(f"patch_embed{prefix}(4x4x{in_ch} conv)", T[0], cs[0],
                  PATCH * PATCH * in_ch, views, f"{prefix}/s1")]
    for i in range(4):
        st, c, d, hh = f"{prefix}/s{i + 1}", cs[i], ds[i], hs[i]
        items += [
            gemm("qkv", Tp[i], 3 * c, c, d * views, st),
            attn("win_attn", WIN * WIN, WIN * WIN, c, d * views * NW[i], stage=st),
            gemm("proj", Tp[i], c, c, d * views, st),
            gemm("ffn_fc1", T[i], 4 * c, c, d * views, st),
            gemm("ffn_fc2", T[i], c, 4 * c, d * views, st),
        ]
        if i < 3:                                   # PatchMerging: 4C_in -> C_next
            items.append(gemm("patch_merge", T[i + 1], cs[i + 1], 4 * c, views, st))
    return items


def swin_elem_items(cs, ds, hs, g, views, prefix):
    T = [h * w for (h, w) in g]
    Tp = [ceil7(h) * ceil7(w) for (h, w) in g]
    NW = [ceil7(h) * ceil7(w) // WIN // WIN for (h, w) in g]
    out = []
    for i in range(4):
        st = f"{prefix}/s{i + 1}"
        out += [
            elem("ln", T[i] * cs[i], 2 * ds[i] * views, st),
            elem("gelu_ff", T[i] * 4 * cs[i], ds[i] * views, st),
            elem("softmax_exp", hs[i] * Tp[i] * WIN * WIN, ds[i] * views * NW[i], st),
        ]
    return out


def neck_items(in_cs, out_c, g, views, prefix):
    """ChannelMapper：3 个 1x1 + extra 3x3 s2（im2col 后 K=in*9）。"""
    tok = [g[1][0] * g[1][1], g[2][0] * g[2][1], g[3][0] * g[3][1],
           g[3][0] // 2 * (g[3][1] // 2)]
    items = [
        gemm("neck_l1", tok[0], out_c, in_cs[0], views, prefix),
        gemm("neck_l2", tok[1], out_c, in_cs[1], views, prefix),
        gemm("neck_l3", tok[2], out_c, in_cs[2], views, prefix),
        gemm("neck_extra3x3s2", tok[3], out_c, in_cs[2] * 9, views, prefix),
    ]
    return items


# ----------------------------------------------------------------------------
# BERT / 特征增强 / PSE / 动作专家
# ----------------------------------------------------------------------------
def bert_items(l: int):
    st = "Text/BERT"
    return [
        gemm("qkv", l, 3 * BERT_H, BERT_H, BERT_LAYERS, st),
        attn("attn_qk+pv", l, l, BERT_H, BERT_LAYERS, stage=st),
        gemm("o_proj", l, BERT_H, BERT_H, BERT_LAYERS, st),
        gemm("ffn_fc1", l, BERT_INTER, BERT_H, BERT_LAYERS, st),
        gemm("ffn_fc2", l, BERT_H, BERT_INTER, BERT_LAYERS, st),
        gemm("text_feat_map(768->256)", l, ENH_D, BERT_H, 1, st),
    ]


def bert_elem_items(l: int):
    st = "Text/BERT"
    return [
        elem("gelu_ff", l * BERT_INTER, BERT_LAYERS, st),
        elem("ln", l * BERT_H, 2 * BERT_LAYERS, st),
        elem("softmax_exp", BERT_HEADS * l * l, BERT_LAYERS, st),
        elem("embed_gather", l * BERT_H, 1, st),
    ]


def lvl_tokens(g):
    """4 级金字塔 (stride 8/16/32/64) 每视角 token：g1+g2+g3+g3/2 = 1700@320x256。
    第 4 级来自 ChannelMapper extra 3x3 s2 → (4,5)=20，bi-attn 只吃这一级。"""
    tok = lambda gg: gg[0] * gg[1]
    lvl4 = tok((g[3][0] // 2, g[3][1] // 2))
    return tok(g[1]) + tok(g[2]) + tok(g[3]) + lvl4, lvl4


def enhancer_items(views, l, g):
    """6 层 TextImageDeformable2DEnhancer。
    lvl_tok=4 级总 token/视角(1700@320x256)；lvl4=stride64 级 token/视角(20)。
    bi-attn 的 N_v = views*lvl4（各视角拼接后单次联合注意力，跨视角耦合）。"""
    lvl_tok, lvl4 = lvl_tokens(g)
    nv = views * lvl4
    bi, img, txt = "Fusion/bi", "Fusion/img", "Fusion/txt"
    items = [
        gemm("bi_v+values_proj", nv, 2 * BI_PROJ, ENH_D, ENH_LAYERS, bi),
        gemm("bi_out_v", nv, ENH_D, BI_PROJ, ENH_LAYERS, bi),
        gemm("bi_l+values_proj", l, 2 * BI_PROJ, ENH_D, ENH_LAYERS, bi),
        gemm("bi_out_l", l, ENH_D, BI_PROJ, ENH_LAYERS, bi),
        attn("bi_attn_qk+av_v", nv, l, BI_PROJ, ENH_LAYERS, stage=bi),
        gemm("bi_attn_av_l", l, BI_PROJ, nv, ENH_LAYERS, bi),
        gemm("def_sampling_offsets", lvl_tok, ENH_D, ENH_D, ENH_LAYERS * views, img),
        gemm("def_attn_weights", lvl_tok, 128, ENH_D, ENH_LAYERS * views, img),
        gemm("def_value_proj", lvl_tok, ENH_D, ENH_D, ENH_LAYERS * views, img),
        gemm("def_output_proj", lvl_tok, ENH_D, ENH_D, ENH_LAYERS * views, img),
        gemm("img_ffn_fc1", lvl_tok, ENH_INTER, ENH_D, ENH_LAYERS * views, img),
        gemm("img_ffn_fc2", lvl_tok, ENH_D, ENH_INTER, ENH_LAYERS * views, img),
        gemm("txt_qkv", l, 3 * ENH_D, ENH_D, ENH_LAYERS, txt),
        attn("txt_attn_qk+pv", l, l, ENH_D, ENH_LAYERS, stage=txt),
        gemm("txt_out", l, ENH_D, ENH_D, ENH_LAYERS, txt),
        gemm("txt_ffn_fc1", l, 1024, ENH_D, ENH_LAYERS, txt),
        gemm("txt_ffn_fc2", l, ENH_D, 1024, ENH_LAYERS, txt),
    ]
    return items, lvl_tok, lvl4


def enhancer_elem_items(views, l, lvl_tok, lvl4):
    nv = views * lvl4
    return [
        elem("ln", lvl_tok * ENH_D, 2 * ENH_LAYERS * views, "Fusion/img"),
        elem("silu_ff", lvl_tok * ENH_INTER, ENH_LAYERS * views, "Fusion/img"),
        elem("softmax_exp", BI_HEADS * nv * l * 2, ENH_LAYERS, "Fusion/bi"),
        elem("ln", l * ENH_D, 2 * ENH_LAYERS, "Fusion/txt"),
        elem("softmax_exp", TXT_HEADS * l * l, ENH_LAYERS, "Fusion/txt"),
    ]


def pse_items(views, lvl_tok):
    """DepthFusionSpatialEnhancer（每次推理 1 次，不进去噪循环）。
    pts_fc 的 217600=1700*128 个 3D 点：相机固定时可整体离线缓存（not_modeled 注）。"""
    st = "PSE"
    return [
        gemm("pts_prob_pre_fc", lvl_tok, 128, ENH_D, views, st),
        gemm("pts_prob_fc1", lvl_tok, ENH_D, ENH_D, views, st),
        gemm("pts_prob_fc2", lvl_tok, 128, ENH_D, views, st),
        gemm("pts_fc(3->128)", lvl_tok * DEPTH_BINS, 128, 3, views, st),
        gemm("fusion_fc1", lvl_tok, 1024, PSE_FUSION_IN, views, st),
        gemm("fusion_fc2", lvl_tok, 512, 1024, views, st),
        gemm("fusion_out", lvl_tok, ENH_D, 512, views, st),
    ]


def pse_elem_items(views, lvl_tok):
    return [
        elem("softmax_depth128", lvl_tok * DEPTH_BINS, views, "PSE"),
        elem("silu_ff", lvl_tok * 1024, views, "PSE"),
        elem("ln", lvl_tok * ENH_D, 2 * views, "PSE"),
    ]


def decoder_items(nj, n_img, l, hoist_kv):
    """每去噪步条目（stage="ActionHead/step"，account 自动 ×denoise_steps）。
    Q=nj*16 动作 token；temp_joint KV=nj*17；AV 按头拆 8x6=48 条 bmm；
    temp_joint 的 QK^T 是 6D einsum（非 GEMM，not_modeled）。"""
    q, kv = nj * NUM_CHUNK, nj * (HIST_CHUNKS + NUM_CHUNK)
    st = "ActionHead/step"
    per_step = [
        gemm("input_proj(32->256)", q, DEC_D, CHUNK_SIZE * STATE_DIM, 1, st),
        gemm("input_fc2-5", q, DEC_D, DEC_D, 4, st),
        gemm("t_embed", 1, DEC_D, DEC_D, 2, st),
        gemm("adarm_proj(256->1536)", 1, 1536, DEC_D, DEC_LAYERS, st),
        gemm("temp_q", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("temp_kv", kv, 2 * DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("temp_out", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("temp_joint_dist_embed", nj * nj, DEC_D, DEC_D, 2 * DEC_LAYERS, st),
        gemm("temp_av", q, DEC_D // DEC_HEADS, kv, DEC_LAYERS * DEC_HEADS, st),
        gemm("img_q", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("img_kv", n_img, 2 * DEC_D, DEC_D, DEC_LAYERS, st),
        attn("img_attn_qk+pv", q, n_img, DEC_D, DEC_LAYERS, stage=st),
        gemm("img_out", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("txt_q", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("txt_kv", l, 2 * DEC_D, DEC_D, DEC_LAYERS, st),
        attn("txt_attn_qk+pv", q, l, DEC_D, DEC_LAYERS, stage=st),
        gemm("txt_out", q, DEC_D, DEC_D, DEC_LAYERS, st),
        gemm("ffn_fc1", q, DEC_INTER, DEC_D, DEC_LAYERS, st),
        gemm("ffn_fc2", q, DEC_D, DEC_INTER, DEC_LAYERS, st),
        gemm("head_conv0(3x1d)", nj * 32, 128, DEC_D * 3, 1, st),
        gemm("head_conv1(3x1d)", nj * 64, 64, 128 * 3, 1, st),
        gemm("head_out_mlp", nj * 64, 64, 64, 2, st),
        gemm("head_out(64->8)", nj * 64, STATE_DIM, 64, 1, st),
    ]
    once = robot_encoder_items(nj)
    if not hoist_kv:
        return per_step, once
    hoisted = {it.name for it in per_step if it.name in
               ("img_kv", "txt_kv", "temp_joint_dist_embed")}
    once += [Item(i.name, i.m, i.n, i.k, i.count, i.kind, stage="ActionHead/once")
             for i in per_step if i.name in hoisted]
    return [i for i in per_step if i.name not in hoisted], once


def robot_encoder_items(nj):
    """HoloBrainRobotStateEncoder：每 chunk 1 次（operation_order 前 4 层）。"""
    st = "ActionHead/once"
    return [
        gemm("enc_input_fc(8->256)", nj, DEC_D, STATE_DIM, 1, st),
        gemm("enc_input_fc2-6", nj, DEC_D, DEC_D, 5, st),
        gemm("enc_jga_qkv", nj, 3 * DEC_D, DEC_D, 4, st),
        gemm("enc_jga_out", nj, DEC_D, DEC_D, 4, st),
        gemm("enc_jga_dist_embed", nj * nj, DEC_D, DEC_D, 2 * 4, st),
        attn("enc_jga_attn", nj, nj, DEC_D, 4, stage=st),
        gemm("enc_ffn_fc1", nj, DEC_INTER, DEC_D, 4, st),
        gemm("enc_ffn_fc2", nj, DEC_D, DEC_INTER, 4, st),
    ]


def decoder_elem_items(nj, n_img, l):
    q, kv = nj * NUM_CHUNK, nj * (HIST_CHUNKS + NUM_CHUNK)
    st = "ActionHead/step"
    return [
        elem("rmsnorm", q * DEC_D, 3 * DEC_LAYERS, st),
        elem("silu_ff", q * DEC_INTER, DEC_LAYERS, st),
        elem("adarm_modulate", q * DEC_D, 3 * DEC_LAYERS, st),
        elem("softmax_exp", DEC_HEADS * q * n_img, DEC_LAYERS, st),
        elem("softmax_exp", DEC_HEADS * q * kv, DEC_LAYERS, st),
        elem("softmax_exp", DEC_HEADS * q * l, DEC_LAYERS, st),
        elem("rope", 2 * q * DEC_D, 2 * DEC_LAYERS, st),
        elem("act_update(dpmsolver+fk)", q * STATE_DIM, 3, st),
    ]


# ----------------------------------------------------------------------------
# 参数量（核验过的代码级加总；GEMM 权重之外的 LN/rel-pos 等一并计入）
# ----------------------------------------------------------------------------
def params_of():
    swin2d = 27_520_506       # Swin-T(96/192/384/768)，逐 stage 加总（核验通过）
    swin3d = 790_648          # backbone_3d 精确重算（核验采信第三路）
    return {
        "backbone_2d(Swin-T)": {"params": swin2d, "basis": "R2 逐 stage 加总+核验员复核"},
        "neck_2d(ChannelMapper)": {"params": 2_116_608, "basis": "3x1x1+extra3x3s2+GN"},
        "backbone_3d(Swin micro)": {"params": swin3d, "basis": "R3 精确重算（核验采信）"},
        "neck_3d(ChannelMapper)": {"params": 177_664, "basis": "核验修正：extra 是 3x3s2"},
        "spatial_enhancer(PSE头)": {"params": 1_314_048,
                                    "basis": "pre_fc 32,896+MLP 98,688+pts_fc 512+fusion 1,181,952+LN"},
        "bert_base": {"params": 108_891_648,
                      "basis": "embedding 23,837,184 + encoder 12x7,087,872，无 pooler"},
        "text_feat_map": {"params": 196_864, "basis": "Linear(768->256)"},
        "feature_enhancer(6L)": {"params": 21_907_840,
                                 "basis": "bi 1.579M + img 1.282M + txt 0.790M + level_embed"},
        "action_expert": {"params": 20_853_960,
                          "basis": "论文 Table2 20.8M 对账差 0.3%（decoder 14.20M+encoder 6.12M+io 0.54M）"},
    }


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
def build_spec(views: int = N_VIEWS, nj: int = N_JOINTS, lang: int = LANG_TOKENS,
               steps: int = DENOISE_STEPS, depth: bool = True,
               img_w: int = IMG_W, img_h: int = IMG_H, hoist_kv: bool = False) -> dict:
    g = grids(img_w, img_h)
    gemm_items = swin_items(SW2_C, SW2_D, SW2_H, 3, g, views, "Swin2D")
    elem_items = swin_elem_items(SW2_C, SW2_D, SW2_H, g, views, "Swin2D")
    gemm_items += neck_items(NECK2_IN, NECK_OUT, g, views, "Neck2D")
    if depth:
        gemm_items += swin_items(SW3_C, SW3_D, SW3_H, 1, g, views, "Swin3D")
        elem_items += swin_elem_items(SW3_C, SW3_D, SW3_H, g, views, "Swin3D")
        gemm_items += neck_items(NECK3_IN, 128, g, views, "Neck3D")
        lvl_tok, _ = lvl_tokens(g)
        gemm_items += pse_items(views, lvl_tok)
        elem_items += pse_elem_items(views, lvl_tok)
    gemm_items += bert_items(lang)
    elem_items += bert_elem_items(lang)
    enh, lvl_tok, lvl4 = enhancer_items(views, lang, g)
    gemm_items += enh
    elem_items += enhancer_elem_items(views, lang, lvl_tok, lvl4)
    per_step, once = decoder_items(nj, views * n_img_per_view(g), lang, hoist_kv)
    gemm_items += per_step + once
    elem_items += decoder_elem_items(nj, views * n_img_per_view(g), lang)

    step_macs = sum(i.macs for i in per_step)
    stage_macs = {
        "Vision2D(Swin+neck)": sum(i.macs for i in gemm_items
                                   if i.stage.startswith(("Swin2D", "Neck2D"))),
        "Vision3D+PSE(depth)": sum(i.macs for i in gemm_items
                                   if i.stage.startswith(("Swin3D", "Neck3D", "PSE"))),
        "Text(BERT+map)": sum(i.macs for i in gemm_items if i.stage.startswith("Text")),
        "Fusion(enhancer6L)": sum(i.macs for i in gemm_items
                                  if i.stage.startswith("Fusion")),
        "ActionHead(once)": sum(i.macs for i in once),
        "ActionHead(x10)": step_macs * steps + sum(i.macs for i in once),
    }
    total = sum(i.macs for i in gemm_items) + step_macs * (steps - 1)
    return {
        "config": {
            "model": "HB-GD 0.2B (HoloBrain GD, arXiv 2602.12062) LIBERO 主档",
            "img_wh": [img_w, img_h], "views": views, "num_joints": nj,
            "lang_tokens": lang, "denoise_steps": steps, "hoist_kv": hoist_kv,
            "with_depth": depth, "pred_steps": PRED_STEPS,
            "num_chunk": NUM_CHUNK, "state_dim": STATE_DIM,
            "lvl_tokens_per_view": lvl_tok, "lvl4_tokens_per_view": lvl4,
            "n_img_tokens": views * n_img_per_view(g),
            "action_tokens": nj * NUM_CHUNK, "temp_kv": nj * (HIST_CHUNKS + NUM_CHUNK),
        },
        "params": params_of(),
        "gemm_items": [asdict(i) | {"macs": i.macs} for i in gemm_items],
        "elem_items": [asdict(i) | {"macs": i.macs} for i in elem_items],
        "totals": {
            "gemm_macs_per_step": step_macs,
            "gemm_macs_total": total,
            "gemm_flops_total": 2 * total,
            "elementwise_ops_total": sum(i.macs for i in elem_items),
            "stage_macs": stage_macs,
        },
    }


def n_img_per_view(g):
    """decoder img_cross 的 K/V：feature_level=[1,2] → stride16+stride32 token。"""
    return (g[2][0] * g[2][1]) + (g[3][0] * g[3][1])


def w_bytes_of(spec: dict) -> int:
    """全部 GEMM 权重字节（int8；attention 的 B 来自 COPY 不走 WRAM）。
    ActionHead/step 的权重每步重装 ×denoise_steps（忠实上游：无 K/V 缓存）。"""
    tot = 0
    for it in spec["gemm_items"]:
        mul = spec["config"]["denoise_steps"] if it["stage"] == "ActionHead/step" else 1
        if it["kind"] == "gemm":
            tot += it["k"] * it["n"] * it["count"] * mul
    return tot


def main() -> None:
    out_dir = Path(__file__).parent
    spec = build_spec()
    t, cfg, p = spec["totals"], spec["config"], spec["params"]
    print("=" * 78)
    print(f"HB-GD 0.2B · LIBERO 主档 {cfg['views']} 视图 {cfg['img_wh'][0]}x{cfg['img_wh'][1]}"
          f" N_j={cfg['num_joints']} lang={cfg['lang_tokens']}"
          f" 去噪{cfg['denoise_steps']}步 chunk={cfg['pred_steps']}")
    print("=" * 78)
    tot_params = sum(v["params"] for v in p.values())
    for k, v in p.items():
        print(f"  {k:<28}: {v['params']/1e6:8.2f} M   ({v['basis']})")
    print(f"  {'合计':<28}: {tot_params/1e6:8.2f} M  (论文口径 0.2B 量级 ✓)")
    for k, v in t["stage_macs"].items():
        print(f"  {k:<28}: {v/1e9:8.3f} GMAC")
    print(f"  每去噪步                : {t['gemm_macs_per_step']/1e9:8.3f} GMAC")
    print(f"  TOTAL GEMM              : {t['gemm_macs_total']/1e9:.2f} GMAC / chunk"
          f" ({t['gemm_flops_total']/1e12:.2f} TFLOP)")
    print(f"  Elementwise(粗账)       : {t['elementwise_ops_total']/1e9:.3f} G-ops")
    print(f"  W 流量/chunk            : {w_bytes_of(spec)/1e6:.0f} MB"
          f" (int8, 动作专家每步重装 x{cfg['denoise_steps']})")
    opt = build_spec(hoist_kv=True)
    print(f"  [opt] K/V+dist hoist 后 : {opt['totals']['gemm_macs_total']/1e9:.2f} GMAC"
          f" / W {w_bytes_of(opt)/1e6:.0f} MB"
          f"  (省 {100*(1-opt['totals']['gemm_macs_total']/t['gemm_macs_total']):.0f}% MAC,"
          f" {100*(1-w_bytes_of(opt)/w_bytes_of(spec)):.0f}% W)")
    (out_dir / "holobrain_spec.json").write_text(json.dumps(spec, indent=1),
                                                 encoding="utf-8")
    print("  -> holobrain_spec.json")


if __name__ == "__main__":
    main()

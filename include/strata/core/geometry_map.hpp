// include/strata/core/geometry_map.hpp - Stratic: describe a model instead of compiling it in.
//
// Two steps live here, and they are separate on purpose:
//
//   apply_arch_geometry   the FILE's numbers onto the engine's ModelGeometry
//   qsa_shapes_from       the kernel shapes the attention path needs
//
// The capability flags on ArchSpec are what make the first step safe. "The file does not declare
// hc" cannot by itself mean "zero the gated residual", because a future qwen4exp file might simply
// omit an optional key; only the family knows whether the component exists.
//
// On the attention side, see Â§15.10 of the white paper: `qsa_selection_width(n_kv, s)` is
// min(n_kv, idx_top_k + idx_block - 1), and below that bound the selection is the IDENTITY. So a
// model with no indexer does NOT need a new attention kernel -- it needs idx_top_k at the kernel's
// cap, which makes every context up to kTopkMaxCells exact plain attention.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/qsa.hpp"

namespace strata::core {

/// Apply the file's numbers to `g`. Fields this architecture does not have are ZEROED rather than
/// left at the engine's compiled-in 125B defaults.
inline void apply_arch_geometry(const strata::ArchGeometry& a, ModelGeometry& g) {
    if (a.n_layers > 0)   g.n_layers = a.n_layers;
    if (a.n_embd > 0)     g.n_embd = a.n_embd;
    if (a.qsa_interval > 0) g.qsa_interval = a.qsa_interval;
    if (a.n_head > 0)     g.n_head = a.n_head;
    if (a.n_head_kv > 0)  g.n_head_kv = a.n_head_kv;
    if (a.head_dim > 0)   g.head_dim = a.head_dim;

    if (a.ssm_state_size)    g.ssm_state_size = a.ssm_state_size;
    if (a.ssm_k_heads)       g.ssm_k_heads = a.ssm_k_heads;
    if (a.ssm_v_heads)       g.ssm_v_heads = a.ssm_v_heads;
    if (a.ssm_d_conv)        g.ssm_d_conv = a.ssm_d_conv;
    if (a.ssm_conv_channels) g.ssm_conv_channels = a.ssm_conv_channels;
    if (a.ssm_value_dim)     g.ssm_value_dim = a.ssm_value_dim;

    // `n_ff` is the DENSE feed-forward width. For a MoE model this field is the PER-EXPERT width,
    // which the expert table's own blob geometry describes, so it is not taken from
    // feed_forward_length there -- overwriting it would change the 125B's expert geometry.
    if (a.n_ff > 0 && a.dense()) g.n_ff = a.n_ff;

    // A dense feed-forward is a one-expert mixture: the router has a single choice, softmax over
    // one logit is 1.0, and the expert path computes the FFN. The runtime already accepts this
    // (router_top10 has no lower bound on n_expert; prefill.cpp checks `n_expert < 1`).
    // A dense feed-forward is a one-expert mixture. A MoE file declares its own count; only override
    // when the file actually spoke, so a MoE file that omitted the key keeps the compiled-in shape
    // instead of silently becoming dense.
    if (a.dense()) g.n_expert = 1;
    else if (a.n_expert > 0) g.n_expert = a.n_expert;

    // absent components are zeroed, not inherited
    if (!a.has_gated_residual) { g.hc = 0; g.hc_lr = 0; }
    if (!a.has_sparse_indexer) { g.idx_q_heads = 0; g.idx_key_dim = 0; }
}

/// The attention shapes for this model.
///
/// A model WITH an indexer keeps the family's sparse selection budget. A model WITHOUT one gets
/// idx_top_k at the kernel cap, which the selection rule turns into the identity -- exact plain
/// attention over every cached cell, up to kTopkMaxCells (32768). Past that the cap must be
/// raised; upstream tracks that as its own Phase 3 / Phase 7.
inline strata::kernels::QsaShapes qsa_shapes_from(const ModelGeometry& g, const strata::ArchGeometry& a) {
    strata::kernels::QsaShapes s;      // NOT qsa_real_shapes(): those are the 125B's numbers
    s.n_head    = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim  = g.head_dim;
    s.n_rot     = a.rope_dims > 0 ? a.rope_dims : strata::kernels::qsa_real_shapes().n_rot;
    s.page_size = 4;                   // one page = one indexer block, as the artifact ships

    if (a.has_sparse_indexer) {
        s.idx_n_head = g.idx_q_heads;
        s.idx_dim    = g.idx_key_dim;
        s.idx_block  = a.idx_block  > 0 ? a.idx_block  : 4;
        s.idx_top_k  = a.idx_top_k  > 0 ? a.idx_top_k  : 2048;
    } else {
        // no indexer: no indexer heads, and a budget large enough that the selection is the identity
        // for every context the selection kernel can serve at all.
        s.idx_n_head = 0;
        s.idx_dim    = 0;
        s.idx_block  = 4;
        s.idx_top_k  = strata::kernels::kTopkMaxCells;
    }
    return s;
}

/// Convenience: the number of cached cells attention will actually read at `n_kv`.
inline int64_t attention_width(int64_t n_kv, const strata::kernels::QsaShapes& s) {
    return strata::kernels::qsa_selection_width(n_kv, s);
}

}  // namespace strata::core

"""Fill the Triton kernel and then enable its launcher; no PyTorch fallback."""
import triton
import triton.language as tl


@triton.jit
def attention_forward(Q, K, V, O, S: tl.constexpr, D: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    # TODO(A5.1): load a Q tile with guarded sequence-tail indices.
    # TODO(A5.2): keep running max m_i, normalizer l_i, FP32 output accumulator.
    # TODO(A5.3): iterate causal K/V tiles and update via online softmax.
    # TODO(A5.4): normalize once and store FP16, with no S*S global workspace.
    # Each first valid row has at least its diagonal key: avoid -inf - -inf.
    pass


def launch_attention(q, k, v, out, cfg):
    # After implementing attention_forward, replace the exception with:
    # grid = (triton.cdiv(q.shape[2], cfg.tile_m), q.shape[0] * q.shape[1])
    # attention_forward[grid](q, k, v, out, q.shape[2], 128,
    #     cfg.tile_m, cfg.tile_n, num_warps=cfg.warps, num_stages=cfg.stages)
    raise NotImplementedError("A5: implement tiled causal attention with online softmax")

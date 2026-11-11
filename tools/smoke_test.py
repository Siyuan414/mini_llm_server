"""
smoke_test.py — S1.1: Verify both engine profiles execute and produce finite outputs.

Tests:
  1. PREFILL (profile 0): seq=128, past=0  → hidden [1,128,2048] finite
  2. DECODE  (profile 1): seq=1,  past=512 → hidden [1,1,2048]  finite

Run from llm_server/:  python tools/smoke_test.py
"""

from pathlib import Path

import numpy as np
import tensorrt as trt
try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 12.6/13.x
except ImportError:
    from cuda import cudart  # older cuda-python

NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN = 2048

ENGINE = Path(__file__).parent.parent / "engine" / "llama1b_fp16.trt"


def check(err):
    e = err[0] if isinstance(err, tuple) else err
    if e != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {e}")
    return err[1:] if isinstance(err, tuple) and len(err) > 1 else None


def alloc(nbytes):
    (ptr,) = check(cudart.cudaMalloc(nbytes))
    return ptr

def run_phase(engine, profile, lens, seq, tokens, kv_data, poison=1e4):
    """
    lens    : real cached length per slot, e.g. [500, 30, 200, 1]
    seq     : new tokens per slot this step (1 for decode)
    tokens  : [B, seq] int64 — the new token ids
    kv_data : list of B arrays, kv_data[i] is [H, lens[i], D] fp16 — real KV
    poison  : value written into the dead padding region
    returns : hidden states [B, seq, HIDDEN]
    """
    B    = len(lens)
    past = max(lens)
    total = past + seq
    print(f"\n── profile {profile}: B={B} lens={lens} past={past} seq={seq} ──")

    ctx = engine.create_execution_context()
    (stream,) = check(cudart.cudaStreamCreate())
    ctx.set_optimization_profile_async(profile, stream)
    check(cudart.cudaStreamSynchronize(stream))

    buffers = {}

    def upload(name, arr):
        arr = np.ascontiguousarray(arr)
        ptr = alloc(max(arr.nbytes, 2))
        if arr.nbytes:
            check(cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        buffers[name] = ptr
        ctx.set_input_shape(name, arr.shape)
        ctx.set_tensor_address(name, ptr)

    # Row i's new token sits at its OWN next position, not at `past`.
    pos = np.array([np.arange(l, l + seq) for l in lens], dtype=np.int64)

    # real tokens | dead padding | new token(s).  TRT appends new KV at index
    # `past`, so the new columns are always the last `seq`.
    mask = np.zeros((B, total), dtype=np.int64)
    for i, l in enumerate(lens):
        mask[i, :l]    = 1
        mask[i, past:] = 1

    # Poison the padding so a masking failure blows up instead of drifting.
    kv = np.full((B, NUM_KV_HEADS, past, HEAD_DIM), poison, dtype=np.float16)
    for i, l in enumerate(lens):
        if l:
            kv[i, :, :l, :] = kv_data[i]

    upload("input_ids",      tokens.astype(np.int64))
    upload("position_ids",   pos)
    upload("attention_mask", mask)
    for i in range(NUM_LAYERS):
        upload(f"past_key_values.{i}.key",   kv)
        upload(f"past_key_values.{i}.value", kv)

    out_ptrs, hidden_name = {}, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(ctx.get_tensor_shape(name))
        ptr = alloc(max(int(np.prod(shape)) * 2, 2))
        out_ptrs[name] = (ptr, shape)
        ctx.set_tensor_address(name, ptr)
        if "present" not in name:
            hidden_name = name

    assert ctx.execute_async_v3(stream), "enqueue failed"
    check(cudart.cudaStreamSynchronize(stream))

    ptr, shape = out_ptrs[hidden_name]
    host = np.empty(shape, dtype=np.float16)
    check(cudart.cudaMemcpy(host.ctypes.data, ptr, host.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    for p in list(buffers.values()) + [p for p, _ in out_ptrs.values()]:
        cudart.cudaFree(p)
    cudart.cudaStreamDestroy(stream)

    print(f"  hidden {shape}  finite={np.isfinite(host.astype(np.float32)).all()}")
    return host


def main():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())
    print(f"Engine: {ENGINE.name}  profiles={engine.num_optimization_profiles}")

    lens = [500, 30, 200, 1]
    rng  = np.random.default_rng(0)
    kv_all = [(rng.standard_normal((NUM_KV_HEADS, l, HEAD_DIM)) * 0.1).astype(np.float16)
              for l in lens]
    tok = rng.integers(0, 128000, (len(lens), 1), dtype=np.int64)

    # 1. prefill still works (past must be 0 → profile 0)
    run_phase(engine, 0, [0, 0, 0, 0], 128,
              rng.integers(0, 128000, (4, 128), dtype=np.int64), [None]*4)

    # 2. ragged batched decode
    h_batch = run_phase(engine, 1, lens, 1, tok, kv_all)

    # 3. invariance A — slot 1 alone, no padding at all
    h_alone = run_phase(engine, 1, [lens[1]], 1, tok[1:2], [kv_all[1]])
    diff = np.abs(h_batch[1].astype(np.float32) - h_alone[0].astype(np.float32)).max()
    print(f"\nA. slot1 batched vs alone : max|diff| = {diff:.5f}")

    # 4. invariance B — same batch, opposite poison; must be unchanged
    h_flip = run_phase(engine, 1, lens, 1, tok, kv_all, poison=-1e4)
    diff2 = np.abs(h_batch[1].astype(np.float32) - h_flip[1].astype(np.float32)).max()
    print(f"B. slot1 poison +1e4 vs -1e4: max|diff| = {diff2:.5f}")

    ok = diff < 5e-2 and diff2 == 0.0
    print("\n" + ("PASSED — padding is correctly masked"
                  if ok else "FAILED — masked positions are leaking into the output"))
    
    # C. same batch size, same slot-1 data, but no padding anywhere
    lens_c = [30, 30, 30, 30]
    kv_c   = [kv_all[1]] * 4          # slot 1's data in every slot
    h_nopad = run_phase(engine, 1, lens_c, 1, tok, kv_c)
    diff3 = np.abs(h_batch[1].astype(np.float32) - h_nopad[1].astype(np.float32)).max()
    print(f"C. slot1 past=500 vs past=30 : max|diff| = {diff3:.5f}")


    print(np.abs(h_alone[0].astype(np.float32) - h_nopad[1].astype(np.float32)).max())  # expect ~0
    print(np.abs(h_batch[1].astype(np.float32)).max())                                   # signal scale

    return 0 if ok else 1
    


if __name__ == "__main__":
    main()

# CANNJudge SqrtSoftplusTopK — best kernel (v8b)

Ascend C kernel for the CANNJudge `SqrtSoftplusTopK` operator
(score(x) = sqrt(softplus(x)), per-row top-k of router logits).

## Result (2026-09-14)

- Platform: 15/15 PASS, best recorded score 40.12 → current ~38.5 under
  continuously-drifting live targets (competitor targets improved during the
  campaign, most notably t03's tbest falling to 4.36us).
- t15 (E=1024, k=64, fp32, T=131072): **1719us → 851us** (2.02x), stable
  across three submissions.
- t03 (E=1024/k64/T~16K): 128us (filter does not pay at small T on the
  platform hardware), t14 (E=512/k32/fp16): 590us.

## Kernel structure

- Entry dispatch (`run_kernel`) gates the filter to
  `(fp16, E=512, k=32)` and `(fp32, E=1024, k=64)`; every other shape runs
  the proven pruned-pow2 baseline (`sqrtsoftplus_pruned_pow2`).
- Filter path (`sqrtsoftplus_fixedcap_{half,float}`):
  1. two 32-value samples per row, one batched `Sort32`; the lower of the two
     4th-largest sample values thresholds the row (scale-free, robust to the
     value distribution for iid rows);
  2. per-row `CompareScalar` + counter-mode `GatherMask` compact candidates
     (sentinel-padded candidate buffer);
  3. per-row compaction into a fixed CAP layout (CAP=128 for E512/k32,
     CAP=256 for E1024/k64);
  4. **one batched `Sort32` for the whole batch** followed by the proven
     batched cross-block merge chain (4-way/2-way `MrgSort` + compaction);
  5. rows with count<k or count>CAP take an exact per-row variable fallback;
  6. math (softplus/sqrt in fp32) runs only on the selected top-k.
- **DMA prefetch**: depth-2 input queue with the software-pipelined
  CopyIn/Compute/CopyOut loop; the sample and fallback paths reuse the fast
  chain buffers so batchRows=8 (E1024) fits UB.

## Rejected variants (evidence in project experiments/)

- E256/k16 filter: 25-30% slower than baseline (mask/gather overhead) — gated off.
- tiered CAP (256/512): ~0% gain even on crafted count∈(256,512] data.
- CAP96/CAP192 with 3-way merges: device traps (validBit=7 unreliable).
- single-sample rank5, output-queue depth-2, T-gated tiered entry: neutral or worse.

## Build / eval

Compiles with CANN 8.5.0 (dav-2201 / A2). The local eval pipeline
(88→97-case correctness + benchmark proxies, fingerprint dedup, auto-rollback)
lives in the main project's `tools/local_eval/`.

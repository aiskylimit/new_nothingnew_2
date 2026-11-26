# Các run cần chạy cho rebuttal: chạy gì, ở file nào, report ra đâu

Ngày soạn: 2026-11-26. Mọi lệnh chạy từ thư mục `SpectralGuidedLearning/` trên server.

## 0. Chạy như thế nào

```bash
cd SpectralGuidedLearning
bash project_commands.sh --dry-run          # chỉ in lệnh, không chạy: kiểm tra đường dẫn trước
bash project_commands.sh                    # chạy cả 6 nhóm, lần lượt
bash project_commands.sh positional w4      # chỉ chạy các nhóm được nêu, theo thứ tự đó
```

| Nhóm | Script | Nội dung | Output chính |
|---|---|---|---|
| `rebuttal` | `project_commands_rebuttal.sh` | answer gain trên s1K-1.1, 3 model, so với SFT và zero-shot | `rebuttal/` |
| `answer_steps` | `project_commands_answer_steps.sh` | chỉ tính loss trên các step chứa đáp án | `rebuttal/` |
| `vcore_asft` | `project_commands_vcore_asft.sh` | baseline VCORE và ASFT, 1.5B, 966 trace | `rebuttal/` |
| `positional` | `project_commands_positional.sh` | kiểm tra thiên lệch vị trí của answer gain | `results/positional/` |
| `w4` | `project_commands_w4.sh` | độ bền theo probe và cách chia bước (reviewer HHj8, W4) | `results/w4/` |
| `hard` | `project_commands_hard.sh` | ALG-hard: giữ/bỏ cứng theo answer gain so với trọng số mềm của ALG | `results/hard/` |

**Chung cho mọi nhóm**

- Venv `/mnt/local/uvenvs/spectral_guided_learning`.
- Model `/mnt/local/_models/aiskylimit_new_nothingnew_2/`, benchmark `/mnt/local/_data/aiskylimit_new_nothingnew_2/`, chạy offline.
- Script tự `source` venv, kiểm tra model, dataset, CUDA, vLLM, DeepSpeed rồi mới chạy. Thiếu gì thì dừng ngay.
- Mặc định dùng GPU 0 và 1, đổi bằng `GPUS="0 1"`. Mỗi run train trên **1 GPU**, hai GPU chạy song song hai run khác nhau, batch hiệu dụng vẫn là 32.
- Bước nào đã có output thì được bỏ qua: crash thì chạy lại đúng lệnh cũ.
- Một nhóm lỗi không chặn nhóm sau; cuối cùng in `FAILED: ...`.
- Toàn bộ console của mỗi lần chạy được ghi vào `driver-<thời gian>.log` (`rebuttal/logs/`, `results/positional/meta/`, `results/w4/meta/`).

**Giao thức đánh giá chung** (`configs/common/eval/palign-protocol.yaml`):
- vLLM, thinking off, n=3, T=0.6, top-p 0.9, repetition penalty 1.05, 4096 token, seed 42.
- Benchmark AIME25, AIME24, AMC12, MATH500; chấm bằng math_verify.

**Công thức train của track 1.5B / 966 trace** (`r1-qwen-1.5b-palign`):
- Full FT, 3 epoch, lr 5e-5 cosine về 1e-5, warm-up 10%, batch hiệu dụng 32, max 32,768.
- DeepSpeed ZeRO-2 có fp32 master weights.
- ALG/ALERD công bố = arm `e7-alg-tau1` (τ=1, λ=0.5, c=2).

Thứ tự trong `project_commands.sh`: rebuttal → answer_steps → vcore_asft → positional → w4 → hard. Nhóm `rebuttal` lâu nhất (3 model trên s1K). Cần kết quả 1.5B sớm thì chạy `bash project_commands.sh positional w4 vcore_asft` trước.

---

## 1. `rebuttal`: answer gain trên s1K-1.1 (3 model)

**File:** `project_commands_rebuttal.sh`; config `configs/sgl/{r1-qwen-1.5b,qwen25-7b,qwen3-8b}-s1k/`.

**Model:**
- `r1-qwen-1.5b`: full FT.
- `qwen25-7b`, `qwen3-8b`: LoRA r16.

Mỗi model chạy trên một GPU, lần lượt các bước sau.

| Run (config) | Bước | Làm gì |
|---|---|---|
| `iwc-gain-l05[-lora]` | prepare, capture, answer_gain, gain_signal, weights, answer_stats, train, eval | chia bước theo câu → chấm answer gain (probe P0, max-model-len 32768) → trọng số IWC-Stable λ=0.5 → thống kê step chứa đáp án → train → eval |
| `sft-nll[-lora]` | prepare, vanilla, train, eval | SFT thường, NLL đều trên mọi token |
| `zeroshot` | eval | model gốc, không train |

**Report:**
- `rebuttal/results/<run>/summary.json` và `raw/*.jsonl`: điểm và output từng sample.
- `rebuttal/results/comparison-table.md`: bảng so sánh.
- `rebuttal/answer-steps/<model>-s1k/`: số token trong step chứa đáp án, tỉ lệ token đáp án / token của step, bản xem HTML.
- `rebuttal/efficiency/table.md`: thời gian và bộ nhớ GPU đỉnh của từng bước (trả lời C1 về hiệu năng).
- `rebuttal/logs/`: log từng bước.

## 2. `answer_steps`: chỉ học trên step chứa đáp án

**File:** `project_commands_answer_steps.sh`; config `configs/sgl/<model>-s1k/answer-steps-only[-lora].yaml`.

**Chạy gì:**
- Cùng 3 model, cùng dữ liệu s1K, cùng lịch train và eval như nhóm 1.
- Loss NLL chỉ tính trên các step chứa đáp án; mọi step khác bị che khỏi loss.
- `ANSWER_DEF=mention` (mặc định): chuỗi đáp án xuất hiện ở bất kỳ đâu trong step. `ANSWER_DEF=boxed`: chỉ step viết `\boxed{đáp án}`.

**Report:** `rebuttal/results/ansstep-<định nghĩa>[-lora]-<model>-s1k/`, nằm chung bảng `comparison-table.md` với nhóm 1.

Phải chạy sau nhóm 1, không chạy đồng thời: cả hai ghi vào `data/<model>-s1k/`.

## 3. `vcore_asft`: baseline VCORE và ASFT

**File:**
- `project_commands_vcore_asft.sh`
- Config `configs/sgl/r1-qwen-1.5b-palign/sft-{asft,vcore}.yaml`
- Code `src/sgl/training/baselines.py`

| Run | Làm gì |
|---|---|
| `sft-asft` | ASFT: loss = sg(p)·NLL + λ·KL(model gốc ‖ student), λ=0.05 |
| `sft-vcore` | VCORE theo code đã phát hành |

Chi tiết VCORE:
- Bước thăm dò θ−ε·g trên một batch ngẫu nhiên.
- Trọng số token = softmax(gain/(ε·T))·N, với ε=1e-4, T=5000.
- Loss nhân thêm min(1, uniform/weighted).
- Chạy 1 GPU, không DeepSpeed, fp32.

Lệnh `prepare,vanilla` chạy trước, sau đó mỗi GPU một arm. Có thể truyền override, ví dụ `bash project_commands_vcore_asft.sh asft_kl_lambda=0.03`.

**Report:**
- `rebuttal/results/sft-{asft,vcore}-r1-qwen-1.5b-palign/`
- `comparison-table.md`
- `rebuttal/efficiency/`
- Trong `checkpoints/<run>/run-summary.json`: `method`. Với VCORE có thêm `weight_stats` (`effective_token_fraction`: nếu ≈1 thì trọng số gần như đều).

**Cần để ý:**
- Port VCORE thăm dò trên toàn bộ trọng số; upstream chỉ dùng LoRA.
- Chưa tune; tác giả grid 4 cấu hình.
- VRAM ước tính khoảng 40 GB, thời gian 3–4 lần SFT.

## 4. `positional`: thiên lệch vị trí của answer gain

**File:**
- `project_commands_positional.sh`
- Code `src/sgl/positional/`
- Config `configs/sgl/r1-qwen-1.5b-palign/pos-r{6,7,8}.yaml`

**Chế độ:**
- `bash project_commands_positional.sh analyze`: chỉ phân tích, vài phút, CPU.
- `... report`: dựng lại report từ đĩa.
- `RUNS_EXTRA=r7`: chạy thêm R7.
- `PALIGN_RUN=<tag> ALERD_RUN=<tag>`: chỉ định run baseline.

**Các bước:**

1. **Dữ liệu và gain.**
   - Dùng `experiments/answer_gain/scores/answer_gain.json`, file gain ALERD đã train.
   - Nếu không có, script chấm lại và in cảnh báo, vì kết quả có thể lệch bản gốc.
2. **Bảng và trọng số** (`sgl.positional.tables`).
   - Tạo step_table / trace_table, trọng số R6/R7/R8, file train `data/r1-qwen-1.5b-palign/train-positional-r{6,7,8}.jsonl`.
   - Kiểm tra `analysis/sanity_weights.json`: bảo toàn khối lượng, R6 đơn điệu, R8 hoán vị đúng, so với Table 8 (0.595 / 0.761 / 1.233). Có kiểm tra nào hỏng thì **không train**.
3. **Phân tích A1–A4** (`analysis/`, csv/json/pdf/png).
   - A1: khối lượng trọng số theo decile vị trí. A1b: profile.
   - A2: ρ giữa gain và vị trí.
   - A3: vị trí của step có gain lớn nhất.
   - A4: theo đoạn prefix / continuation.
4. **Baseline.** Output từng sample của P-ALIGN (40.76 / 52.03, tự tìm trong `results/`) và ALERD: dùng lại `results/<tag>/raw` hoặc eval lại từ checkpoint.
5. **R6** = pipeline ALERD với điểm = vị trí tương đối của step; train và eval.
6. **Quyết định** (`sgl.positional.decide`), theo quy tắc chốt trước (ngưỡng median|ρ| 0.3, uniform 40.76±1.0, ALERD 42.78/54.70). Nếu quy tắc yêu cầu thì chạy thêm:
   - R7 = trọng số trung bình theo decile.
   - R8 = hoán vị z trong cùng decile.

**Report:**
- `results/positional/report.md`, `decision.json`, `analysis/`.
- `runs/<r6|r7|r8>/`:
  - `config.json`, `data_check.json`, `train_log.jsonl` (từng optimizer step), `train_summary.json`.
  - `eval/generations.jsonl`, `metrics.json`, `paired_vs_baselines.json` (paired bootstrap).

## 5. `w4`: độ bền theo probe và cách chia bước (reviewer HHj8, W4)

**File:**
- `project_commands_w4.sh`
- Code `src/sgl/robustness/` (`segment`, `token_logp`, `signals`, `stability`, `runs`)
- Config `configs/sgl/r1-qwen-1.5b-palign/w4.yaml` + `configs/sgl/methods/robustness.yaml`

**Chế độ:**

```bash
bash project_commands_w4.sh score       # chỉ chấm điểm + độ ổn định thứ hạng (không train), làm trước
bash project_commands_w4.sh train A     # train/eval nhóm A (thêm B, C tùy ý: train A B C)
bash project_commands_w4.sh report      # dựng lại bảng từ đĩa
bash project_commands_w4.sh             # tất cả: score → A, B, C → report
```

Biến môi trường:
- `KEEP_EPOCH_CKPT=0`: xóa checkpoint epoch 1–2 sau khi eval xong, tiết kiệm khoảng 6 GB mỗi run.
- `SEED42_UNIFORM`, `SEED42_ALG`: tag của run seed 42 có sẵn, mặc định `e0-uniform-…` và `e7-alg-tau1-…`.

### 5.1 Biến thể

**Probe** (cùng cách chia S0). Mọi probe đều kết thúc bằng `\boxed{`:

| Mã | Probe |
|---|---|
| P0 | `\n\nTherefore, the final answer is \boxed{` (mặc định) |
| P1 | `\n\nThe answer is \boxed{` |
| P2 | `\n\nFinal Answer: \boxed{` |
| P3 | `\n\n**Final Answer**\n\n\boxed{` |
| Pavg | trung bình u_k của P0–P3, không chấm thêm |

**Cách chia bước** (cùng probe P0):

| Mã | Cách chia |
|---|---|
| S0 | theo câu (mặc định) |
| S1 | theo đoạn `\n\n` |
| S2a / S2b | khối 64 / 128 token |
| S3 | theo câu nhưng không cắt trong `$…$`, `$$…$$`, `\[…\]`, `\(…\)` và không cắt sau số thứ tự danh sách (`1.`, `**2.**`) |

Số bước trên 966 trace (đã chạy thử trên máy này; tokenizer test nên S2a/S2b lệch nhẹ so với tokenizer R1):

| | S0 | S1 | S2a | S2b | S3 |
|---|---|---|---|---|---|
| Tổng số bước | **47,532** (khớp paper) | **82,559** | 47,275 | 23,867 | 46,777 |
| Median bước / trace | 34 | 71 | 42 | 21 | 34 |
| Trace chỉ có 1 bước | 0 | 0 | 0 | 0 | 0 |

> **Khác với plan:** S1 cho **nhiều bước hơn** S0, không phải "ít hơn nhiều": trace R1 xuống dòng đôi rất thường xuyên. Mục 7 của plan (c = 2 ít bị chạm với S1) phải đảo lại: với S1, mỗi bước ngắn hơn và nhiều hơn. Chấm S1 cần khoảng 83.5k probe, gần gấp đôi S0.

### 5.2 Bước 1: chấm điểm và độ ổn định (không train)

1. `prepare` (dùng lại `train-segmented.jsonl` = S0).
   - `runs hashes`: SHA-256 của file 966 trace, tokenizer và trọng số model gốc → `results/w4/meta/hashes.json`.
2. `sgl.robustness.segment` × 5: tạo `data/r1-qwen-1.5b-palign/w4/segmented-<S>.jsonl` kèm `.check.json`.
   - Σℓ_k = số token phản hồi; EOS nằm ngoài các bước; phân phối số bước; số trace 1 bước.
   - S0 phải tái tạo đúng `train-segmented.jsonl`.
   - Lỗi bất kỳ thì dừng.
3. Chấm điểm trên GPU 0 và 1, mỗi GPU một job:
   - `token_logp`: log-prob từng token dưới student đóng băng, chạy 1 lần (cột `predictability`).
   - 8 job `answer_gain`: P0/S0, P1/S0, P2/S0, P3/S0, P0/S1, P0/S2a, P0/S2b, P0/S3.
   - Mỗi job lưu `signals/<…>/raw/{gains,logp,meta}.json`. `meta`: chuỗi probe, token id, ngữ cảnh dài nhất mỗi trace, các bước vượt 16,384, thời gian.
   - Kèm `gpu_mem.txt` (bộ nhớ GPU lấy mẫu mỗi giây) và `score.log`.
4. `sgl.robustness.signals` × 9 (8 cặp + Pavg) → `results/w4/signals/r1q1.5b__<probe>__<seg>/`:
   - `steps.parquet`, một dòng / bước:
     - `tok_start/end`, `ell`, `char_start/end`, `text`
     - `region` (prefix / continuation / crosses_marker), `has_final_boxed`
     - `b`, `u`, `z`, `v`, `clipped`, `r`, `w_hat`, `w`
     - `contains_answer_exact/normalized`, `predictability`
   - `traces.parquet`, một dòng / trace:
     - `K`, số token, số token prefix, đáp án vàng + token id + `N`
     - `b0`, kiểm tra Σu = b_K − b_0, `mu_u`, `sigma_u`, `uniform_fallback`, `B` (phải = 1)
     - ngữ cảnh probe dài nhất, `exceeds_context`, `top1_step`, `top1_contains_answer`, `top3_pos_gain_share`
   - `summary.json`:
     - số bước, phân phối K, % trace về trọng số đều, % bước bị clip
     - khối lượng của 10% bước nặng nhất (paper 39.5%), % bước tăng/giảm > 0.1 nat (paper 43%/39%)
     - % gain dương ở bước chứa đáp án (paper 28.3%), % trace có top-step chứa đáp án (paper 66%)
     - mọi kiểm tra; thời gian chấm; bộ nhớ GPU đỉnh
   - `manifest.json`: git commit + diff, phiên bản thư viện, phần cứng, hash, probe và token id, hàm chia bước, hash `steps.parquet`.
   - **Hồi quy P0/S0:** so với `experiments/answer_gain/scores/answer_gain.json` (sai số tối đa, Spearman), và so trọng số với `train-e7-alg-tau1.jsonl` nếu còn.
   - File train `data/r1-qwen-1.5b-palign/train-w4-<probe>-<seg>.jsonl` (cộng `train-w4-uniform.jsonl` từ S0):
     - trọng số IWC-Stable τ=1, λ=0.5, c=2, đúng hàm của arm đã công bố
     - thêm `token_tags` để log NLL theo vùng
     - kiểm tra phủ token hoặc bất biến ngân sách hỏng thì **không tạo file train**
5. `sgl.robustness.stability`, so với P0/S0 → Bảng R-W4a:
   - **Probe:** Spearman ρ của u_k trong từng trace, top-3 Jaccard, tỉ lệ gain dương trong top-3.
   - **Cách chia:** trải w_k xuống token, Spearman giữa hai vector trọng số token, độ chồng khối lượng top-10% token.
   - **Đường chuẩn ngẫu nhiên:** 50 lần xáo mỗi trace (seed 42), % trace vượt phân vị 95% của ngẫu nhiên.
   - Output:
     - `results/w4/analysis/rank_stability/<probe>__<seg>.parquet` (một dòng / trace)
     - `rank_stability_summary.csv` (median, IQR, hai chỉ số W3)
     - `table_r_w4a.md`

### 5.3 Bước 2: train và eval (12 run)

Tên run: `r1q1.5b__<method>__<probe>__<seg>__tau1_lam0.5_c2__seed<s>`.

| Nhóm | Run | Seed |
|---|---|---|
| A (bắt buộc) | ALG P1/S0, ALG Pavg/S0, ALG P0/S1, ALG P0/S3 | 42 |
| B | ALG P2/S0, ALG P3/S0, ALG P0/S2a, ALG P0/S2b | 42 |
| C (đo nhiễu) | uniform ×2, ALG P0/S0 ×2 | 43, 44 |

Seed 42 của uniform và ALG mặc định là các run E0 / E7 có sẵn. Với nhóm C, `report` đọc `results/<tag>/summary.json` của chúng để tính mean ± std trên 3 seed.

Mỗi run nằm trong `results/w4/runs/<run_id>/`:

- `config.yaml`: config đã resolve đầy đủ.
- `train/`:
  - `stdout.log`, `stderr.log`
  - `config.json`, `data_check.json` (trọng số token đúng với file train)
  - `train_log.jsonl`, một dòng / optimizer step:
    - `loss_weighted`, `loss_nll_unweighted`
    - NLL không trọng số theo vùng: `nll_prefix`, `nll_cont`, `nll_junction` (8 token đầu sau `<End_of_Prefix>`), `nll_answer_steps`, `nll_non_answer_steps`, `nll_eos`
    - `Z`, `sum_w_over_Z` (≈ 1), `w_mean`, `w_max`, `grad_norm`, `lr`, `tokens_per_sec`, `max_mem_gb`
  - `epoch_log.jsonl`: trung bình các cột trên theo epoch, cộng NLL không trọng số trên **toàn bộ 966 trace** với model cuối mỗi epoch (toàn bộ và theo vùng).
  - `health.jsonl`:
    - tỉ lệ tham số đã đổi ở step 64 (cảnh báo nếu < 50%: lỗi bf16 không có fp32 master ở Appendix A)
    - loss hoặc grad_norm là NaN/Inf, grad_norm > 10 lần median, Σw/Z lệch 1
  - `train_summary.json`
- `checkpoints/epoch1`, `epoch2`: checkpoint bf16, chỉ trọng số. `checkpoints/epoch3` = checkpoint cuối dùng để eval; state optimizer bị xóa khi train xong.
- `raw/*.jsonl`, `summary.json`: output thô của vLLM.
- `eval/samples.jsonl`, một dòng / generation, 643 bài × 3 = 1,929 dòng:
  - `benchmark`, `problem_id`, `sample_idx`, `sampling_seed`, `generation`, `num_tokens`, `finish_reason`
  - `extracted_answer`, `reference_answer`, `correct_mathverify`, `correct_oat`, `correct` (OR)
  - `truncated`, `repetition_loop`, `emits_marker`, `has_boxed`
- `eval/metrics.json`, theo benchmark và trung bình:
  - pass@1, pass@3 (math_verify), pass@1/3 khi OR với OAT
  - tỉ lệ bị cắt, độ dài trung bình/median, tỉ lệ 3 sample đồng ý, tỉ lệ lặp vòng, tỉ lệ sinh marker
  - số lần Math-Verify và OAT bất đồng, cấu hình vLLM
- `timing.json`: thời gian train và eval.
- `manifest.json`: git, phiên bản, phần cứng, seed, hash, thư mục signals + hash `steps.parquet`, thời gian.
- `run_check.json`: checklist 5.8 tự động:
  - manifest có đủ; step log không NaN; Σw/Z ≈ 1 ở mọi step; tham số đã đổi
  - đủ 1,929 sample; tính lại được metrics từ samples
  - `git_clean`

### 5.4 Báo cáo tổng

- `results/w4/analysis/results_master.csv`, một dòng / run:
  - method, probe, seg, τ, λ, c, seed
  - p@1, p@3 của 4 benchmark và trung bình
  - tỉ lệ cắt, đồng ý, lặp vòng
  - NLL cuối (toàn bộ / prefix / continuation)
  - thời gian chấm, train, eval; thư mục signals; commit
- `results/w4/analysis/table_r_w4b.md` (Bảng R-W4b):
  - số của paper
  - mean ± std 3 seed của uniform và ALG mặc định
  - từng biến thể
  - danh sách biến thể rơi dưới uniform 40.76 (tiêu chí ở mục 0 của plan)
- `results/w4/analysis/table_r_w4a.md` (Bảng R-W4a).

Mọi con số trong rebuttal nên lấy từ hai file trên, không ghi tay.

### 5.5 Kiểm tra trước khi chạy W4 và chi phí

- Chạy `score` trước, đọc các file sau rồi mới train:
  - `signals/r1q1.5b__P0__S0/summary.json`:
    - `checks.regression_vs_reference_gains`: lệch ở mức nhiễu bf16, Spearman ≈ 1
    - `default_reference`: 47,532 bước, 48,498 probe, 28.3%, 66%, 39.5%
  - `n_traces_exceeding_context`
- Chấm điểm: plan ước tính khoảng 12 phút / cặp trên H200. S1 gần gấp đôi; tổng khoảng 1.5–2 giờ cho 8 cặp trên 2 GPU.
- Train: 12 run. Nếu giữ checkpoint epoch 1–2 thì khoảng 3 GB × 3 × 12 ≈ 110 GB; `KEEP_EPOCH_CKPT=0` để chỉ giữ epoch 3.
- NLL cuối epoch trên 966 trace thêm vài phút mỗi epoch.

---

## 6. Những gì chưa kiểm chứng được

Máy soạn script không có GPU. Đã làm:
- Test CPU (tiny Qwen2): 330 pass, 2 fail (2 test này fail sẵn từ trước khi sửa).
- Dry-run cả 5 nhóm.
- Chạy chia bước W4 và phân tích positional trên đủ 966 trace thật, với tokenizer test.

Chưa chạy thật lần nào: train DeepSpeed, vLLM eval, chấm answer gain trên GPU.

Những điểm cần xem khi chạy trên server:

1. **File gain ALERD** `experiments/answer_gain/scores/answer_gain.json` cần có trên server. Thiếu thì:
   - positional chấm lại (có thể lệch);
   - W4 mất bước hồi quy P0/S0.
2. **W4, ALG mặc định seed 43/44** train trên gain P0/S0 **chấm lại**, không phải file gain gốc của E7; chỉ lệch ở mức nhiễu bf16 nếu hồi quy đạt.
3. **W4 `gold_matches_reference`** để trống: file 966 trace không có đáp án tham chiếu của s1K-1.1, và `problem_id` = chỉ số trong file 966 trace.
4. **W4 `samples.jsonl`** không có prompt / hash prompt, vì output thô của `sgl.eval.evaluate` không lưu prompt. Prompt dựng lại được từ benchmark + cấu hình.
5. **W4 không có `epoch_log.jsonl` cho seed 42 của uniform/ALG:** đó là các run E0/E7 cũ, chỉ có điểm.
6. Hai định nghĩa trong `summary.json` của W4 chưa chắc trùng với paper: "% gain dương ở bước chứa đáp án" (khối lượng gain dương) và "khối lượng top-10%". Định nghĩa được ghi ngay trong file; hồi quy P0/S0 sẽ cho thấy có khớp số paper không.
7. **Positional:** quy tắc quyết định có một trường hợp không phủ (in `NOT COVERED`). A1b "khác nhiều" cần đọc tay; nếu cần thì `RUNS_EXTRA=r7`.
8. **VCORE:** trọng số có thể gần đều; xem `effective_token_fraction` trong `run-summary.json`.

---

## ALG-hard (`hard`): chọn cứng với cùng answer gain

**File:** `project_commands_hard.sh`; code `src/sgl/hardsel/`; config `configs/sgl/r1-qwen-1.5b-palign/hard-{h59,r59,h30,alg-h59}.yaml`. 1.5B, 966 trace, recipe E7.

```bash
bash project_commands_hard.sh --dry-run
bash project_commands_hard.sh prepare       # CPU: chọn bước, kiểm tra, thống kê, ghi quy tắc quyết định
bash project_commands_hard.sh train A       # H59, R59 (2 GPU)
bash project_commands_hard.sh train B       # H30, ALG∘H59
bash project_commands_hard.sh nll_posthoc   # NLL theo vùng cho uniform / ALG
bash project_commands_hard.sh report        # bảng, hình, decision.json
bash project_commands_hard.sh train C       # H59 seed 43, 44, chỉ chạy khi decision.json yêu cầu
bash project_commands_hard.sh               # tất cả: prepare, A, nll_posthoc, report, C nếu cần, B, report
```

| Run (`r1q1.5b__<arm>__P0__S0__seed<s>`) | Nhóm | Token được giám sát | Trọng số |
|---|---|---|---|
| `h59` | A | bước gain cao nhất cho đến khi đủ ngân sách | 1 |
| `r59` | A | thứ tự bước ngẫu nhiên (`Random(42 + trace id)`), đủ số token của H59 ở trace đó | 1 |
| `h30` | B | như H59, p = 0.30 | 1 |
| `alg-h59` | B | mask H59 | ALG (τ=1, λ=0.5, c=2) tính lại trên bước giữ |
| `h59` seed 43, 44 | C | như H59 | 1 |

**Đầu vào phải có trên server**
- Gain: `experiments/answer_gain/scores/answer_gain.json`; không có thì dùng `results/w4/signals/r1q1.5b__P0__S0` (chạy `bash project_commands_w4.sh score` trước).
- Run seed 42 của uniform / ALG: `results/e0-uniform-r1-qwen-1.5b-palign/raw`, `results/e7-alg-tau1-r1-qwen-1.5b-palign/raw` (đổi bằng `SEED42_UNIFORM`, `SEED42_ALG`). Thiếu thì không có paired bootstrap so với ALG và `decision.json` ở trạng thái `pending`.
- Checkpoint E0 / E7 (`checkpoints/<tag>`) và checkpoint từng epoch của W4 nhóm C: cho `nll_posthoc`. Thiếu cái nào thì bỏ qua cái đó và in cảnh báo.
- Tùy chọn: `data/r1-qwen-1.5b-palign/train-spectral.jsonl` (gate SGL). Có file này thì mặc định (`BUDGET=auto`) ngân sách của H59/R59 là số token gate SGL giữ ở từng trace (bảng ghi `H-SGL`); `BUDGET=fixed` để luôn dùng 59%.

**Report** (`results/hard/`)
- `selection/<arm>/`: `steps.parquet`, `traces.parquet`, `select_check.json`, `selection_stats.json`, `manifest.json`. Kiểm tra nào hỏng thì không ghi file train.
- `runs/<run_id>/`: `config.yaml`; `train/` (`train_log.jsonl` mỗi optimizer step, `epoch_log.jsonl` epoch 0–3, `health.jsonl`, `data_check.json`, `config.json`, `train_summary.json`); `checkpoints/epoch1..3`; `raw/`; `eval/` (`samples.jsonl`, `metrics.json`, `paired_vs_baselines.json`); `timing.json`, `manifest.json`, `run_check.json`.
- `posthoc_nll/<method>__seed<s>/epoch_nll.json`.
- `analysis/`: `table_r_h1.md`, `table_r_h2.md`, `fig_h1_dropped_nll`, `fig_h2_kept_by_position`, `results_master.csv`, `decision.json`, `selection_overlap.json`.

**Lưu ý khi đọc kết quả**
- Bước vượt ngưỡng được giữ nguyên, nên tỉ lệ giữ thực tế cao hơn p (có trace cao hơn nhiều vì một bước rất dài). Xem `selection_stats.json` sau `prepare`; R59 cũng vượt thêm tối đa một bước so với H59.
- Số chính là Math-Verify OR OAT (theo plan); `table_r_h1.md` có thêm bảng chỉ Math-Verify, và `GRADER=mathverify` để quyết định theo Math-Verify. Số công bố 40.76 / 42.78 trong repo được ghi là chấm bằng Math-Verify.
- Dòng SGL (s1K) trong Bảng R-H1 phải điền tay qua `SGL_ROW="..."`.
- `w` và tỉ trọng loss trong Bảng R-H2 tính trên token của các bước (không tính EOS); "top-10% gain" là 10% bước có `u` cao nhất trong từng trace.
- Trong `train_log.jsonl`, `n_tok_total` / `n_tok_kept` không tính EOS; `nll_kept` và `Z` có tính EOS.
- Nhóm C của W4 phải giữ checkpoint epoch (`KEEP_EPOCH_CKPT=1`, mặc định) cho đến khi chạy xong `nll_posthoc`. Đĩa: khoảng 9 GB mỗi run.


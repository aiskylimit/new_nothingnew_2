# Ablation Experiment Plan — Answer-Gain Allocation

**Paper:** *Decisive, Not Difficult: Allocating Learning Effort across Reasoning Steps in Long-CoT Distillation*  
**Mục tiêu:** Kiểm tra (i) tính cần thiết của answer-likelihood gain, (ii) nguy cơ ưu tiên các bước chỉ nêu đáp án, (iii) vai trò của mức phân bổ learning effort, và (iv) độ ổn định của kết quả.  
**Phạm vi pilot:** DeepSeek-R1-Distill-Qwen-1.5B trên 966 P-ALIGN stitched traces.  
**Trạng thái:** Kế hoạch thí nghiệm — chưa phải kết quả đo được.

## 1. Research questions và ưu tiên

| ID | Research question | Thí nghiệm | Ưu tiên |
|---|---|---|---|
| RQ3 | Tại sao dùng **answer-likelihood gain** thay vì **step predictability**? | Easy, Hard, Random Assignment vs ALG | **P0** |
| RQ4 | ALG có chủ yếu thưởng cho các step **nêu lại gold answer** không? | ALG-NoAnswerUpweight | **P0** |
| RQ5 | Mức độ soft allocation có quan trọng không? | Sweep `lambda` | **P1** |
| RQ6 | ALG có nhạy với temperature của trọng số không? | Sweep `tau` | **P1** |
| Robustness | Cải thiện có nhất quán qua các training seeds không? | Repeated training seeds | **P1** |

> **Quy tắc ưu tiên:** Hoàn thành RQ3 và answer-leakage trước; sau đó làm `lambda`, `tau`, rồi mở rộng training seeds cho những phép so sánh quan trọng. Tất cả kết luận cần phụ thuộc kết quả thực nghiệm, không giả định ALG chắc chắn thắng.

## 2. Protocol chung (giữ cố định)

| Thành phần | Cấu hình |
|---|---|
| Student | DeepSeek-R1-Distill-Qwen-1.5B |
| Dữ liệu | 966 stitched traces từ P-ALIGN; cố định mọi trace và segmentation |
| Step segmentation | Sentence boundaries như bản thảo (47,532 steps) |
| Scoring model | Frozen initial student `q0` trước fine-tuning |
| Fine-tuning | Full-parameter SFT, 3 epochs |
| Learning rate | `5e-5`, cosine decay xuống `1e-5`; warmup 10% |
| Effective batch size | 32 |
| Training seed pilot | 42 |
| Các benchmarks | AIME24, AIME25, AMC12, MATH500 |
| Metrics chính | pass@1, pass@3, macro-average qua bốn benchmarks |
| Evaluation | 3 samples / problem, temperature 0.6, top-p 0.9, repetition penalty 1.05, max 4,096 generated tokens; grading bằng Math-Verify |
| ALG default | `lambda = 0.5`, `tau = 2`, `c = 2`, `epsilon = 1e-8` |

**Fair comparison checklist**

- [ ] Tất cả arms khởi tạo từ **cùng checkpoint**, data và step boundaries.
- [ ] Mọi score phụ thuộc `q0` phải được tính **offline trước training**, không cập nhật theo fine-tuned model.
- [ ] Với RQ3, giữ nguyên công thức score-to-weight; **chỉ thay step score**.
- [ ] Giữ cùng prompt formatting, EOS treatment, loss denominator và training recipe.
- [ ] Kiểm tra `sum_k(length_k * weight_k) = sum_k(length_k)` cho **từng trace** (sai số floating-point nhỏ được chấp nhận).
- [ ] Dùng cùng evaluation questions, sampling settings, và evaluation seeds giữa các arms.
- [ ] Baseline Uniform SFT cần được **match trong cùng training pipeline**; không tự động coi kết quả P-ALIGN từ pipeline khác là fully matched.

### 2.1. Pipeline biến step score thành trọng số

Với step score tổng quát `v_k`:

\[
 z_k = \frac{v_k - \mu_v}{\sigma_v + \epsilon}, \quad
 r_k = \exp\left(\frac{\operatorname{clip}(z_k,-c,c)}{\tau}\right),
\]

\[
 \hat w_k = r_k \frac{\sum_j \ell_j}{\sum_j \ell_j r_j}, \qquad
 w_k = (1-\lambda)+\lambda \hat w_k.
\]

Các steps vẫn được supervised; `lambda < 1` bảo đảm một sàn uniform weight. Nếu độ lệch chuẩn score của một trace quá nhỏ, dùng uniform weights cho trace đó. Giữ EOS weight bằng 1 như paper.

## 3. Run matrix — các cấu hình pilot

| Run | Method | Điểm khác biệt | Ưu tiên | Tình trạng dự kiến |
|---|---|---|---|---|
| **E0** | Uniform SFT | `w_k = 1` | P0 | Tái sử dụng nếu matched; nếu không, train lại |
| **E1** | ALG (ours) | `u_k = b_k - b_(k-1)`, default parameters | P0 | Tái sử dụng nếu matched |
| **E2** | Predictability-Easy | `v_k = d_k` | P0 | **Mới** |
| **E3** | Predictability-Hard | `v_k = -d_k` | P0 | **Mới** |
| **E4** | Random Assignment | Hoán vị `u_k` trong từng trace | P0 | **Mới** |
| **E5** | ALG-NoAnswerUpweight | Không tăng trọng số answer-only steps | P0 | **Mới** |
| **E6** | ALG Full Allocation | `lambda = 1` (`tau=2`, `c=2`) | P1 | **Mới** |
| **E7** | ALG Sharper Allocation | `tau = 1` (`lambda=0.5`, `c=2`) | P1 | **Mới** |
| **E8** | ALG Smoother Allocation | `tau = 4` (`lambda=0.5`, `c=2`) | P1 | **Mới** |

**Pilot budget:** 9 configurations tổng cộng, gồm **7 configurations mới (E2–E8)** nếu E0/E1 đã có trong pipeline matched. Đây là số *configurations*, chưa tính các lần training lặp lại cho nhiều seeds.

## 4. RQ3 — Answer gain so với predictability (P0)

**Giả thuyết cần kiểm định:** Một step nên nhận learning effort theo mức nó giúp student phục hồi gold answer, thay vì theo mức nó dễ/khó được student dự đoán.

### 4.1. Định nghĩa score

**ALG** (answer gain):

\[
 b_k=\log q_0(a^*\mid x,s_{1:k},\pi), \qquad
 u_k=b_k-b_{k-1}.
\]

`pi` là fixed answer probe, `a*` là gold answer; đánh giá bằng teacher forcing.

**Predictability** (mean log-likelihood của step):

\[
 d_k=\frac{1}{\ell_k}\sum_{t\in s_k}
 \log q_0(y_t\mid x,y_{<t}).
\]

- **E2 — Easy:** `v_k = d_k`: trọng số cao hơn cho step dễ dự đoán.
- **E3 — Hard:** `v_k = -d_k`: trọng số cao hơn cho step khó dự đoán.
- **E4 — Random Assignment:** shuffle dãy `u_k` **trong mỗi trace**, rồi đưa scores đã shuffle qua **cùng pipeline score-to-weight**. Dùng random seed cố định và log lại phép hoán vị. Không cần chạy thêm independent random-weight baseline.

**Lưu ý về DFT:** Appendix F của bản thảo đã thử DFT token-level với detached probabilities, nhưng đây **không phải** matched step-score ablation. Nếu còn compute, có thể thêm optional step-level probability score `mean_t q0(y_t | x,y_<t)` qua cùng score-to-weight pipeline; báo cáo tách biệt với reproduction của original DFT.

### 4.2. Triển khai

1. [ ] Chạy frozen `q0` teacher forcing trên toàn bộ traces để tính `d_k`.
2. [ ] Tái sử dụng bảng `u_k` hiện có; tính score statistics và kiểm tra `NaN`, `inf`.
3. [ ] Tạo weights cho E2, E3, E4 với `lambda=0.5`, `tau=2`, `c=2`.
4. [ ] Xác thực token-mass conservation và histogram weight theo arm.
5. [ ] Train E2–E4, cùng seed 42 và recipe với E1.
6. [ ] Evaluate bốn benchmarks, so sánh E0–E4.

### 4.3. Analyses không cần training mới

- [ ] Spearman correlation `corr(u, d)` trong mỗi trace; báo cáo median/IQR qua các traces.
- [ ] Top-10% score/weight overlap giữa ALG và Predictability-Easy.
- [ ] Tỷ lệ step `high answer gain / low predictability` và chiều ngược lại.
- [ ] Đồ thị scatter hoặc 2D-binned heatmap giữa `u_k` và `d_k`.

### 4.4. Diễn giải kết quả

- **ALG > Easy, Hard, Random:** hỗ trợ việc dùng answer-directed score trong setting này.
- **ALG ≈ Easy/Hard:** chưa có bằng chứng đủ mạnh rằng answer gain tốt hơn predictability; cần kiểm tra noise/variance và giới hạn claim.
- **ALG > Random nhưng ≈ Easy:** score alignment có ích nhưng chưa chứng minh được ưu thế riêng của answer gain.
- **ALG ≈ Random:** cần xem xét lại tính hữu ích của signal hoặc normalization.

## 5. Answer leakage — ALG đang thưởng cho điều gì? (P0)

**Động cơ từ paper:** Ở 1.5B, các steps chứa gold answer string ít nhưng chiếm tỷ trọng đáng kể trong positive answer gain; top-gain step thường chứa answer string (Section 7.2).

### E5 — ALG-NoAnswerUpweight

1. [ ] Tạo nhãn sơ bộ step **answer-only** (chỉ nêu/lặp lại đáp án), phân biệt với step **có derivation và đồng thời ghi đáp án**.
2. [ ] Kiểm tra thủ công mẫu được gắn nhãn để ước lượng lỗi phân loại.
3. [ ] Đặt `w_k = 1` cho các answer-only steps; vẫn **giữ nguyên tokens trong trace và loss**.
4. [ ] Phân bổ ALG trong tập còn lại và **renormalize token mass trên tập còn lại**, sao cho tổng weight mass mỗi trace bằng uniform.
5. [ ] Train E5 và so sánh E0/E1/E5.

**Edge case:** Nếu một trace không còn step không phải answer-only, dùng uniform weights cho trace đó. Giữ rõ quy tắc nhận dạng answer-only trong báo cáo.

**Cần báo cáo:** Performance, tỷ lệ token/steps bị neutralize, tỷ trọng mass trước/sau neutralization. Nếu E5 vẫn > E0, đây là bằng chứng lợi ích không chỉ đến từ việc tăng trọng số các step chỉ nêu đáp án; không được xem đây là bằng chứng causal đầy đủ về reasoning correctness.

## 6. Allocation mechanism — sweep lambda (P1)

Giữ `tau=2`, `c=2`; chạy:

| Lambda | Ý nghĩa | Run |
|---|---|---|
| `0` | Uniform SFT | E0 |
| `0.5` | ALG default, shrink về uniform | E1 |
| `1` | Full normalized allocation | **E6** |

> Với `tau` hữu hạn và clipped scores, `lambda=1` **vẫn cho weights dương**, không tự động trở thành hard selection.

**Deliverables:** Performance và weight statistics theo `lambda`. Nếu khác biệt không rõ, không kết luận `lambda=0.5` tối ưu. Chỉ mở rộng `lambda ∈ {0,0.25,0.5,0.75,1}` nếu kết quả pilot cần thêm chi tiết.

## 7. Temperature sensitivity — sweep tau (P1)

Giữ `lambda=0.5`, `c=2`; chạy:

| Tau | Hiệu ứng dự kiến | Run |
|---|---|---|
| `1` | Allocation sắc hơn | **E7** |
| `2` | Default | E1 |
| `4` | Allocation đều hơn | **E8** |

**Deliverables:** Performance, P10/median/P90 của weights, top-10% mass concentration, tỷ lệ steps có `w_k>1`. Không sweep `c` đồng thời trong pilot để tránh thay đổi nhiều yếu tố cùng lúc.

**Diễn giải:** Nếu nhiều giá trị `tau` đều hoạt động ổn, phương pháp ít nhạy với nhiệt độ trong phạm vi thử nghiệm; nếu chỉ một giá trị cho kết quả tốt, cần ghi nhận sensitivity và tránh mô tả quá mức.

## 8. Training-seed robustness (P1)

**Pilot:** Mọi E0–E8 đều so sánh tại seed 42 trước. Sau khi có pilot, chọn tập arms tối thiểu để chạy thêm seeds:

1. [ ] Uniform SFT (E0).
2. [ ] ALG (E1).
3. [ ] Đối chứng predictability tốt nhất trong E2/E3 (chọn theo tiêu chí định trước, không tự ý bỏ arm bất lợi).
4. [ ] Random Assignment (E4), nếu sự khác biệt với ALG nhỏ hoặc nếu cần kiểm tra riêng việc alignment.

**Seeds đề xuất:** `42, 123, 456` (đây là lựa chọn mới cho kế hoạch). Tách biệt **training seeds** với **evaluation sampling seeds**; nhiều sampling seeds của một checkpoint không phải nhiều lần huấn luyện độc lập.

**Phân tích:** Báo cáo `mean ± std` giữa các training runs; dùng matched evaluation settings. Có thể tính paired bootstrap CI trên bộ câu hỏi để mô tả evaluation uncertainty. Không claim statistical significance chỉ dựa trên một checkpoint hoặc nhiều decode samples.

## 9. Template bảng kết quả

### 9.1. Main ablation table

| Method | AIME24 p@1 | AIME25 p@1 | AMC12 p@1 | MATH500 p@1 | Avg p@1 | Avg p@3 |
|---|---:|---:|---:|---:|---:|---:|
| E0 Uniform SFT | TBD | TBD | TBD | TBD | TBD | TBD |
| E1 ALG | TBD | TBD | TBD | TBD | TBD | TBD |
| E2 Predictability-Easy | TBD | TBD | TBD | TBD | TBD | TBD |
| E3 Predictability-Hard | TBD | TBD | TBD | TBD | TBD | TBD |
| E4 Random Assignment | TBD | TBD | TBD | TBD | TBD | TBD |
| E5 ALG-NoAnswerUpweight | TBD | TBD | TBD | TBD | TBD | TBD |

### 9.2. Hyperparameter sensitivity

| Parameter | Value | Avg p@1 | Avg p@3 | Weight P90 | Notes |
|---|---:|---:|---:|---:|---|
| `lambda` | 0 | TBD | TBD | TBD | E0 |
| `lambda` | 0.5 | TBD | TBD | TBD | E1 |
| `lambda` | 1 | TBD | TBD | TBD | E6 |
| `tau` | 1 | TBD | TBD | TBD | E7 |
| `tau` | 2 | TBD | TBD | TBD | E1 |
| `tau` | 4 | TBD | TBD | TBD | E8 |

## 10. Execution schedule (phụ thuộc GPU availability)

| Giai đoạn | Công việc | Điều kiện hoàn thành |
|---|---|---|
| **0. Audit** | Verify E0/E1, data, score cache, loss mass, eval script | Baselines matched và pipeline nhất quán |
| **1. RQ3 scoring** | Compute `d_k`, shuffle `u_k`, phân tích correlation | Có score/weight artifacts, validation pass |
| **2. RQ3 training** | Train E2, E3, E4 | Có evaluation table E0–E4 |
| **3. Leakage** | Label answer-only, train E5 | Có E0/E1/E5 và audit labels |
| **4. Hyperparameters** | Train E6, E7, E8 | Có sensitivity table |
| **5. Robustness** | Re-run training seeds cho nhóm so sánh chính | Có mean ± std và uncertainty reporting |
| **6. Paper update** | Viết Analysis/Limitations, kiểm tra reproducibility | Claim khớp dữ liệu; không mâu thuẫn các bảng |

### 10.1. Artifacts cần lưu

```text
experiments/
  scores/
    answer_gain.jsonl
    predictability_logp.jsonl
    shuffled_answer_gain_seed42.jsonl
  weights/
    E0_uniform.jsonl
    E1_alg.jsonl
    E2_easy.jsonl
    E3_hard.jsonl
    E4_random_assignment.jsonl
    E5_no_answer_upweight.jsonl
    E6_lambda1.jsonl
    E7_tau1.jsonl
    E8_tau4.jsonl
  configs/
    E0.yaml ... E8.yaml
  results/
    per_benchmark.csv
    seed_summary.csv
    score_analysis.csv
  figures/
    score_correlation.pdf
    sensitivity.pdf
```

Lưu cùng artifacts: source checkpoint revision, dataset fingerprint, tokenizer, weight construction version, training/evaluation seeds, code commit và log kiểm tra invariant.

## 11. Những thí nghiệm **chưa ưu tiên**

- Absolute answer likelihood `b_k` thay cho gain `u_k`.
- Position-only reweighting và step-length-only reweighting.
- DFT-style **step probability** qua matched allocation (bổ sung nếu reviewer đặc biệt hỏi về DFT).
- Alternative probes, step segmentations, sweep `c`.
- SGL + ALG, different trace sources, multi-student full ablations.

Các thí nghiệm này có thể làm sau E2–E8 tùy kết quả và compute. Đặc biệt, nếu mô hình chỉ upweight những steps cuối hoặc dài, position-/length-only controls sẽ đáng ưu tiên hơn.

## 12. Các vấn đề phải xác minh trong bản thảo trước khi nộp

- **Qwen3-8B inconsistency:** Table 1 ghi ALG **53.07 / 62.41** (Avg p@1 / p@3), trong khi đoạn Main Results ghi **48.48 / 57.61** và kết luận không thắng P-ALIGN. Phải xác định bản kết quả chuẩn trước khi viết conclusions.
- **Training vs evaluation variance:** Báo cáo ba evaluation sampling seeds không thay thế ba independent training seeds.
- **Answer-only label definition:** Đừng đồng nhất mọi step chứa gold-answer string với step chỉ lặp đáp án.
- **Claims:** RQ3 chỉ chứng minh một **predictive supervision signal có ích** dưới matched design; answer-likelihood gain không được trình bày như thước đo causal necessity/logical correctness của reasoning step.

---

## Final checklist

- [ ] E0/E1 matched, evaluation protocol verified.
- [ ] E2/E3/E4 hoàn tất (RQ3), kèm score correlation analysis.
- [ ] E5 hoàn tất (answer leakage), có audit answer-only labels.
- [ ] E6 hoàn tất (`lambda=1`).
- [ ] E7/E8 hoàn tất (`tau=1,4`).
- [ ] Seed robustness của các đối chứng quan trọng hoàn tất.
- [ ] Kết quả chính và bảng sensitivity có mean/variance theo phương án đã nêu.
- [ ] Không còn mâu thuẫn số liệu Qwen3-8B trong bản thảo.

**Các phần bản thảo làm cơ sở:** Section 4.1–4.3 (scoring và allocation), Section 5 (protocol), Section 7.2 và Limitations (điểm yếu của answer gain), Appendix F (DFT), Appendix H (upweighted-step characteristics).

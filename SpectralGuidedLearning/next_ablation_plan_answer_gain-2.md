# Kế hoạch ablation tiếp theo — Budget Normalization (ALG)

> **Trạng thái:** Bản cập nhật sau khi hoàn thành E0–E8.  
> **Track:** `r1-qwen-1.5b-palign` · **training seed:** `42`.  
> **Chỉ chạy mới:** **E9 Global Norm** và **E10 No Norm** (**2 training runs**).  
> **Reference:** **E7 — ALG, Per-trace Norm**, `lambda=0.5`, `tau=1`, `c=2` (đã chạy).  
> **Không nằm trong kế hoạch này:** E11 (Predictability), E12 (NoAnswerUpweight ở `tau=1`), thêm seeds hoặc sweep `tau`.

## 1. Mục tiêu và câu hỏi nghiên cứu

**Ablation: Effect of token-mass / budget normalization**

Kiểm tra liệu cơ chế **bảo toàn tổng supervision weight cho từng trace** có hữu ích hơn (a) chuẩn hóa tổng weight ở cấp **toàn bộ dataset**, hoặc (b) **không chuẩn hóa token mass** hay không.

- **E7 — Per-trace Norm:** Mỗi trace giữ nguyên tổng token-weight mass như uniform SFT.
- **E9 — Global Norm:** Chỉ tổng token-weight mass của **toàn dataset** được giữ nguyên; mỗi trace có thể được tăng/giảm ngân sách riêng.
- **E10 — No Norm:** Không ràng buộc tổng token-weight mass ở cả cấp trace lẫn cấp dataset.

**Nguyên tắc so sánh:** Cùng answer-gain scores, cùng within-trace z-score, cùng `lambda=0.5`, `tau=1`, `c=2`, cùng training/evaluation pipeline. **Chỉ thay đổi bước token-mass normalization.**

`tau=1` được chọn vì E7 là ALG configuration cho kết quả tốt nhất **trong các giá trị đã thử ở seed 42**; không gọi đây là optimum được chọn bằng validation độc lập.

## 2. Run matrix (chốt)

| ID | Method | Token-mass normalization | `lambda` | `tau` | `c` | Seed | Trạng thái |
|---|---|---|---:|---:|---:|---:|---|
| **E7** | ALG | **Per-trace** | 0.5 | 1 | 2 | 42 | **Đã chạy — baseline** |
| **E9** | ALG | **Global (dataset-wide)** | 0.5 | 1 | 2 | 42 | **Cần chạy** |
| **E10** | ALG | **None** | 0.5 | 1 | 2 | 42 | **Cần chạy** |

**Tổng:** **2 runs mới (E9, E10)**. Không chạy lại E7 nếu đã có checkpoint, logs và kết quả đánh giá đúng protocol.

### Kết quả E7 đã có

| ID | Avg pass@1 | Avg pass@3 | Weight P10 | Median | P90 | Steps `w>1` | Top-10% mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E7** | **42.78%** | **54.70%** | 0.595 | 0.761 | 1.233 | 18.42% | 39.45% |

> Giá trị E7 dựa trên ablation summary hiện tại. Chỉ có một training seed; không suy diễn ý nghĩa thống kê từ chênh lệch đơn lẻ.

## 3. Công thức và định nghĩa cần implement

Ký hiệu trace `i`, step `k`, số token thuộc step `l_ik` (không bao gồm EOS). Dùng **cùng score cache với E7**:

\[
u_{ik}=\log q_0(a^*\mid x_i,s_{i,1:k},\pi)-\log q_0(a^*\mid x_i,s_{i,1:k-1},\pi)
\]

\[
z_{ik}=\frac{u_{ik}-\mu_{u,i}}{\sigma_{u,i}+\epsilon},\qquad
r_{ik}=\exp\!\left(\frac{\operatorname{clip}(z_{ik},-2,2)}{1}\right)
\]

Với cả ba variants, giữ công thức shrinkage:

\[
w_{ik}=(1-\lambda)+\lambda\widehat r_{ik},\qquad\lambda=0.5.
\]

### E7 — Per-trace Norm (baseline đã có)

\[
\widehat r^{\mathrm{trace}}_{ik}
=r_{ik}\frac{\sum_j l_{ij}}{\sum_j l_{ij}r_{ij}}.
\]

Với mọi trace `i`:

\[
\sum_k l_{ik}w_{ik}=\sum_k l_{ik}.
\]

### E9 — Global Norm (mới)

\[
\widehat r^{\mathrm{global}}_{ik}
=r_{ik}\frac{\sum_{i,j}l_{ij}}{\sum_{i,j}l_{ij}r_{ij}}.
\]

- Tính **một hệ số global** từ toàn bộ *training traces* sau khi đã có `r_ik`, sau đó cố định hệ số này trong training.
- **Không** thay bằng normalization theo mini-batch: mini-batch normalization là một thuật toán khác.
- Tổng token-weight mass trên dataset được bảo toàn, nhưng từng trace không nhất thiết có budget bằng uniform SFT.

### E10 — No Norm (mới)

\[
\widehat r^{\mathrm{none}}_{ik}=r_{ik},\qquad
w_{ik}=0.5+0.5r_{ik}.
\]

- **Chỉ bỏ token-mass normalization**; vẫn giữ **within-trace z-score**, `clip`, `tau` và `lambda`.
- Không thực hiện bất kỳ thao tác tự động nào để ép trung bình weights về 1 trong loss reducer.
- Vì bỏ normalization cũng có thể đổi **loss/gradient scale**, kết quả E10 phải được diễn giải kèm diagnostics; không quy mọi chênh lệch accuracy cho ưu điểm của per-trace budget preservation.

**EOS:** giữ weight `1` trong tất cả runs, giống E7.

## 4. Thiết lập cần giữ cố định

| Thành phần | Cấu hình |
|---|---|
| Student | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Training data | 966 P-ALIGN stitched traces; cùng tokenizer và sentence-level segmentation |
| Initialization | Cùng frozen initial checkpoint và answer-gain score cache như E7 |
| Fine-tuning | Full fine-tuning, 3 epochs |
| Learning rate / schedule | `5e-5`, cosine decay đến `1e-5`, 10% warmup |
| Effective batch size | 32 |
| Training seed | 42 |
| Score standardization | Within-trace z-score ở **cả ba arms** |
| `lambda`, `tau`, `c` | `0.5`, `1`, `2` |
| Objective | Token-weighted SFT; **giữ nguyên loss denominator** và EOS weight `1` |
| Evaluation | AIME24, AIME25, AMC12, MATH500; giữ decoding/grading và generation budget 4096 |
| Metrics chính | Average pass@1, average pass@3; thêm breakdown theo benchmark |

Không đổi dataset, number of epochs, batch construction/packing, gradient accumulation, optimizer, checkpoint initialization, sampling parameters hoặc grading.

## 5. Kiểm tra bắt buộc trước khi train

- [ ] E7/E9/E10 dùng **giống hệt** `u_ik`, `z_ik`, `r_ik`, step/token boundaries.
- [ ] Cả ba arms đều có `w_ik > 0` với mọi response step; không cut/mask step.
- [ ] Định nghĩa per-trace budget ratio:

  \[
  B_i=\frac{\sum_k l_{ik}w_{ik}}{\sum_k l_{ik}}.
  \]

- [ ] **E7:** kiểm tra `B_i ≈ 1` với mọi trace.
- [ ] **E9:** kiểm tra `sum_i,k(l_ik*w_ik) / sum_i,k(l_ik) ≈ 1`, nhưng `B_i` được phép thay đổi.
- [ ] **E10:** ghi nhận dataset mass ratio và phân phối `B_i`; **không** normalize ngầm.
- [ ] Kiểm tra NaN/Inf, alignment từ steps sang tokens, EOS weight `1`, cùng loss reduction giữa các runs.
- [ ] Lưu lại config, weight stats, file score/weight và log để audit.

## 6. Metrics và artifacts cần xuất

### Quality metrics

- pass@1, pass@3 theo từng benchmark.
- Macro-average pass@1 và pass@3.

### Weight/budget diagnostics

- Dataset-wide token mass ratio: `sum(l*w)/sum(l)`.
- Phân phối **per-trace** `B_i`: mean, std, P10, median, P90.
- Phân phối step weights: P10, median, P90; fraction of steps `w>1`.
- Top-10% token-weight mass (dùng cùng định nghĩa của summary E7).
- Training loss / gradient norm nếu đang có logger; đặc biệt hữu ích khi giải thích E10.

### Files đề xuất lưu

- `results/normalization_ablation.csv`: bảng accuracy.
- `results/normalization_diagnostics.csv`: thống kê weight và budget.
- Run configs và full train/eval logs cho E9, E10.

## 7. Thứ tự thực hiện

1. **Đối chiếu E7:** xác nhận checkpoint, score cache, step lengths, loss reducer, evaluation settings và số liệu tham chiếu.
2. **Sinh weight E9:** một global scaling factor từ toàn bộ training dataset; validate tổng mass.
3. **Sinh weight E10:** `w=0.5+0.5r` (không budget normalization); validate phân phối weight.
4. **Train E9 và E10:** seed 42; mọi tham số khác giống E7.
5. **Evaluate:** cùng decoding settings, benchmark grading và aggregation như E7.
6. **Tổng hợp:** bảng accuracy + budget/weight diagnostics + kết luận giới hạn trong setting được thử.

## 8. Mẫu bảng báo cáo kết quả

### 8.1. Normalization ablation (bảng chính)

| Method | Normalization | `lambda` | `tau` | Avg p@1 | Avg p@3 | Dataset mass ratio | Std of trace `B_i` |
|---|---|---:|---:|---:|---:|---:|---:|
| **E7 ALG** | **Per-trace** | 0.5 | 1 | **42.78%** | **54.70%** | 1.00 *(lý thuyết, cần kiểm tra file)* | 0 *(lý thuyết, cần kiểm tra file)* |
| **E9 ALG** | Global | 0.5 | 1 | — | — | 1.00 *(lý thuyết, cần kiểm tra file)* | — |
| **E10 ALG** | None | 0.5 | 1 | — | — | — | — |

### 8.2. Theo benchmark

| Method | AIME24 p@1 | AIME25 p@1 | AMC12 p@1 | MATH500 p@1 | Avg p@1 | Avg p@3 |
|---|---:|---:|---:|---:|---:|---:|
| E7 Per-trace | — | — | — | — | 42.78% | 54.70% |
| E9 Global | — | — | — | — | — | — |
| E10 No Norm | — | — | — | — | — | — |

> Các ô `—` cần điền từ kết quả thực tế; không suy ra per-benchmark numbers từ macro-average.

## 9. Cách diễn giải kết quả

- **E7 vượt E9:** Ủng hộ việc bảo toàn supervision budget **ở cấp trace** thay vì chỉ bảo toàn trên toàn dataset trong setup đang xét.
- **E7 ≈ E9 và cả hai vượt E10:** Gợi ý global mass control có ích; chưa đủ để nói per-trace conservation tốt hơn global.
- **E10 bằng hoặc vượt E7:** Không có bằng chứng rằng budget normalization là thiết yếu trong setup này; cần xem thêm gradient/loss scale.
- **E9 vượt E7:** Per-trace conservation có thể đang giới hạn khả năng phân bổ budget giữa các traces.

**Giới hạn kết luận:** Chỉ dùng 1 seed và 1 cấu hình `tau=1`. E7 được chọn từ kết quả benchmark đã quan sát, không phải qua held-out validation. E10 thay đổi cả khả năng giữ budget lẫn scale của objective; không dùng E10 riêng lẻ để khẳng định cơ chế nhân quả.

## 10. Checklist nghiệm thu cuối cùng

- [ ] E9 Global Norm — hoàn thành training + evaluation.
- [ ] E10 No Norm — hoàn thành training + evaluation.
- [ ] E9/E10 dùng chung score cache và cùng settings với E7.
- [ ] Kiểm tra per-trace/global mass ratios từ **weight files thực tế**.
- [ ] Báo cáo pass@1/pass@3 cùng diagnostics cho E7/E9/E10.
- [ ] Lưu config, weight files, training/eval logs và per-benchmark results.
- [ ] **Không** chạy lại Predictability, E12, sweep `tau` hoặc training-seed sweep theo plan này.

---

**Chốt kế hoạch: chỉ 2 runs mới — E9 (Global Norm) và E10 (No Norm).** Đối chứng duy nhất trong nhóm normalization là E7 (Per-trace Norm, `lambda=0.5`, `tau=1`, `c=2`). E5 đã kiểm tra answer leakage tại `tau=2` nên không lặp lại trong kế hoạch này.

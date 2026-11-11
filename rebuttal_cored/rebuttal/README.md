# Chạy thực nghiệm rebuttal trên 4 B200

Mặc định dùng GPU `0,1,2,3`, batch `8` mỗi GPU và gradient accumulation 1:
effective batch size 32.

## Chạy lần lượt từng bước, chỉ setting A và C

Chạy từ folder `rebuttal_cored` (nếu đang ở root repo thì `cd rebuttal_cored`).
Trên server không có Internet, bỏ hai lệnh `setup` và
`download`: môi trường do quản lý server cài từ các file `.txt`, model/data
được tải theo `rebuttal/download.txt`.

Model/data dùng chung vẫn ở `/mnt/local/@PROJECT@/rebuttal/assets/`, bên ngoài
folder code `rebuttal_cored/`. Các script tự dùng đúng đường dẫn này; đặt
`REBUTTAL_ASSETS` nếu quản lý server đã tải vào vị trí khác. Dữ liệu prepared,
checkpoint và kết quả chạy nằm trong `rebuttal_cored/rebuttal/` như trước.

Data P-ALIGN cũ trong `palign/data/` không kèm trong code. Các dòng URL P-ALIGN
trong `rebuttal/download.txt` tải lại đúng bản cũ vào
`/mnt/local/@PROJECT@/rebuttal_cored/palign/data/` nếu cần chạy riêng P-ALIGN;
luồng rebuttal dưới đây tự chuẩn bị dữ liệu và không dùng các file cũ này.

Các file môi trường và đường dẫn mặc định:

| File thư viện | Venv | Công việc |
|---|---|---|
| `rebuttal.txt` | `/mnt/local/uvenvs/rebuttal` | Điều phối, prepare, báo cáo, chọn dữ liệu smoke |
| `sft.txt` | `/mnt/local/uvenvs/sft` | Train SFT(Long-CoT) |
| `palign.txt` | `/mnt/local/uvenvs/palign` | Tiền xử lý P-ALIGN và train P-ALIGN |
| `sgl.txt` | `/mnt/local/uvenvs/sgl` | Tạo spectral mask và train SGL |
| `SegmentSelectiveSFT.txt` | `/mnt/local/uvenvs/SegmentSelectiveSFT` | Attribution và train SSFT |
| `cored.txt` | `/mnt/local/uvenvs/cored` | Prefix, key step, expert, council, TIES và student CoReD |
| `rebuttal_eval.txt` | `/mnt/local/uvenvs/rebuttal_eval` | Probe evaluator, sinh đáp án và chấm eval của mọi phương pháp |

Script gọi trực tiếp Python trong đúng venv ở từng bước; không cần activate thủ
công khi chuyển phương pháp.

Các file `.txt` chỉ khai báo package và phiên bản từ PyPI mặc định, không dùng
URL wheel, `--find-links`, index riêng hay hậu tố `+cu...`. Các bộ sau đã được
đối chiếu metadata và nội dung wheel Linux x86_64/Python 3.11 trên PyPI:

| File môi trường | Torch | vLLM | Transformers | CUDA của wheel mặc định |
|---|---|---|---|---|
| `rebuttal.txt`, `sft.txt`, `palign.txt`, `rebuttal_eval.txt` | 2.11.0 | 0.26.0 | 5.5.4 | 13.0 |
| `cored.txt` | 2.11.0 | 0.26.0 | 5.16.1 | 13.0 |
| `sgl.txt` | 2.9.0 | Không dùng | 5.4.0 | 12.8 |
| `SegmentSelectiveSFT.txt` | 2.9.0 | Không dùng | 4.57.1 | 12.8 |

Torch 2.11 đi cùng torchvision 0.26/torchaudio 2.11; Torch 2.9 đi cùng
torchvision 0.24 và xformers 0.0.33.post1 nếu dùng SSFT. Giữ các bộ phiên bản
này cùng nhau; các env tách riêng nên có thể dùng bản CUDA khác nhau.
Với env đã cài sai CUDA, cần cài lại đúng bộ từ file `.txt`; sửa file trong repo
không tự thay thư viện đang có trên server. Sau khi cài, chạy `preflight` trước smoke/train.
Nếu công cụ trung gian dùng mirror hoặc thay wheel khác PyPI mặc định, cần kiểm
tra bản cài thực tế; chỉ số phiên bản package không xác nhận được biến thể CUDA.

Đã kiểm tra giải dependency chỉ từ PyPI cho Python 3.11/Linux và chạy train nhỏ
trên CPU qua script baseline/P-ALIGN và trainer SGL với các phiên bản trên.
Kiểm tra này không thay thế `preflight` và smoke trên B200 sau khi env được cài lại.

```bash
# Chỉ dùng trên máy có Internet; bỏ qua trên server chạy thí nghiệm
bash rebuttal/project_commands.sh setup
bash rebuttal/project_commands.sh download
```

Trên server đã có môi trường và model/data, chạy chuỗi sau. Bất kỳ bước nào
lỗi, kể cả smoke, đều dừng trước các bước còn lại. Train đủ 10 run A+C rồi mới eval.

```bash
bash rebuttal/project_commands.sh prepare && \
GPUS=0,1,2,3 bash rebuttal/project_commands.sh preflight && \
GPUS=0,1,2,3 bash rebuttal/project_commands.sh smoke --settings C && \
GPUS=0,1,2,3 bash rebuttal/project_commands.sh train --settings A C && \
GPUS=0,1,2,3 bash rebuttal/project_commands.sh eval --settings A C && \
bash rebuttal/project_commands.sh report --allow-partial
```

Với `train`/`eval`, đổi `--settings A C` thành `--settings A` hoặc `--settings C`
nếu chỉ chạy một setting; bỏ tùy chọn này để chạy cả ba. Báo cáo dùng `--allow-partial` vì bỏ B
nên chưa đủ ma trận 15 run. Training log vẫn được lưu tự động như trước.

Smoke mặc định chạy 5 phương pháp của setting C với backbone 1.5B, dùng các venv
của run chính; không lặp lại setting A/7B. Smoke chọn 8 mẫu s1K thật ngắn có ít
nhất hai đoạn. Mỗi baseline chạy 4 optimizer
updates, batch 1/GPU (global 4), cutoff 2048; SSFT dùng IG4 và phải tính attribution
thành công. CoReD giữ 3 expert, chạy 4 updates ở Phase 1 và 4 ở distillation,
bao gồm chọn prefix, key step, council và TIES. Bộ lọc đáp án của P-ALIGN và
chấm prefix của CoReD vẫn hoạt động; nếu không còn mẫu hợp lệ thì smoke thất bại.

Sau train, smoke kiểm tra trọng số checkpoint hữu hạn, nạp adapter qua vLLM
và sinh/chấm 4 câu thật trên mỗi benchmark, 1 đáp án/câu, tối đa 256 token,
context 4096. Không có ngưỡng accuracy để pass. Kết quả nằm riêng trong
`rebuttal/smoke-4gpu/summary.json`; chỉ `status: passed` sau đủ các bước mới là pass.
Đặt `REBUTTAL_SMOKE_RUNS` để dùng thư mục smoke khác khi thay code/cấu hình.

# Chạy thực nghiệm rebuttal trên 4 B200

Mặc định dùng GPU `0,1,2,3`, batch `8` mỗi GPU và gradient accumulation 1:
effective batch size 32.

## Chạy lần lượt từng bước, chỉ setting A và C

Chạy từ root repo. Trên server không có Internet, bỏ hai lệnh `setup` và
`download`: môi trường do quản lý server cài từ các file `.txt`, model/data
được tải theo `rebuttal/download.txt`.

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

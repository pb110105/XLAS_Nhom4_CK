# XLAS_CK_Nhom4

Chạy toàn bộ pipeline (Bài 4):

```powershell
python main.py
```

`main.py` sẽ xóa toàn bộ `data/output`, tạo lại ảnh xám và chạy lần lượt:

1. Bài 1: Histogram H1, H2 và thu hẹp [30, 120].
2. Bài 2: Convolution, median filter và tạo I1–I6.
3. Bài 3: Local Binary Pattern (LBP).

Kết quả của mỗi ảnh được chia theo từng bài:

```text
data/output/img01/
├── b1/
├── b2/
└── b3/
```

Ba bài có chung giao diện `run(input_dir, output_dir)` và vẫn có thể chạy riêng:

```powershell
python -m src.bai1
python -m src.bai2
python -m src.bai3
```

# HƯỚNG DẪN TÍNH TAY CHI TIẾT BÀI 3: CIRCULAR LBP (LOCAL BINARY PATTERN)

Bài viết này trình bày các bước toán học chuẩn xác để tính toán đặc trưng LBP mở rộng (Circular LBP) cho một điểm ảnh, bao gồm cả kỹ thuật nội suy song tuyến tính (Bilinear Interpolation) khi tọa độ lân cận không nằm trên lưới số nguyên, và cách xử lý chuỗi bit lớn theo đúng yêu cầu đề bài.

---

## BƯỚC 1: XÁC ĐỊNH TỌA ĐỘ CÁC ĐIỂM LÂN CẬN (NEIGHBORS)

Giả sử ta đang xét điểm tâm $c$ có tọa độ $(x_c, y_c)$ và giá trị mức xám là $I_c$. Ta cần tìm tọa độ của $P$ điểm lân cận nằm trên đường tròn bán kính $R$.

Với mỗi điểm $p$ (từ $p = 0$ đến $P - 1$), góc quay $\theta_p$ và tọa độ $(x_p, y_p)$ được tính bằng công thức:

$$\theta_p = \frac{2\pi p}{P}$$
$$x_p = x_c + R \cos(\theta_p)$$
$$y_p = y_c - R \sin(\theta_p)$$

*(Lưu ý: Trục Y của ảnh kỹ thuật số hướng từ trên xuống dưới, nên ta dùng dấu trừ để góc quay ngược chiều kim đồng hồ theo chuẩn toán học).*

---

## BƯỚC 2: TÍNH MỨC XÁM TẠI ĐIỂM $(x_p, y_p)$ BẰNG NỘI SUY SONG TUYẾN TÍNH

Nếu $x_p$ và $y_p$ là các số nguyên, ta chỉ cần lấy giá trị ảnh tại vị trí đó: $I_p = I(x_p, y_p)$.
Tuy nhiên, nếu tọa độ là **số thập phân**, ta phải tính giá trị ảo $I_p$ dựa trên 4 điểm nguyên gần nhất bao quanh nó.

1. **Tìm 4 điểm lân cận:**
   - $x_1 = \lfloor x_p \rfloor$ (Làm tròn xuống)
   - $x_2 = \lceil x_p \rceil$ (Làm tròn lên)
   - $y_1 = \lfloor y_p \rfloor$ (Làm tròn xuống)
   - $y_2 = \lceil y_p \rceil$ (Làm tròn lên)

2. **Tính trọng số khoảng cách:**
   - $\Delta x = x_p - x_1$
   - $\Delta y = y_p - y_1$

3. **Áp dụng công thức nội suy (Bilinear Interpolation):**
   $$I_p = (1-\Delta x)(1-\Delta y) \cdot I(x_1, y_1)$$
   $$\quad + \Delta x (1-\Delta y) \cdot I(x_2, y_1)$$
   $$\quad + (1-\Delta x) \Delta y \cdot I(x_1, y_2)$$
   $$\quad + \Delta x \Delta y \cdot I(x_2, y_2)$$

---

## BƯỚC 3: LẤY NGƯỠNG VÀ GÁN BIT (THRESHOLDING)

So sánh giá trị lân cận $I_p$ vừa tính được với giá trị tâm $I_c$:

$$
s(I_p - I_c) = \begin{cases} 
1 & \text{nếu } I_p \ge I_c \\
0 & \text{nếu } I_p < I_c 
\end{cases}
$$

Giá trị $s$ này chính là giá trị bit thứ $p$ của chuỗi LBP.

---

## BƯỚC 4: TÍNH GIÁ TRỊ LBP TỔNG QUÁT

Sau khi có $P$ bit, ta ghép chúng lại thành một con số thập phân. Bit thứ $p$ sẽ tương ứng với trọng số $2^p$:

$$LBP_{P,R} = \sum_{p=0}^{P-1} s(I_p - I_c) \cdot 2^p$$

---

## BƯỚC 5: XỬ LÝ THEO YÊU CẦU ĐỀ BÀI (TÁCH CHUỖI VÀ TÌM MAX)

Theo đề bài: *"Trường hợp P=16 (hoặc 24) thì tách chuỗi nhị phân thành 2 phần (hoặc 3 phần)... chỉ lấy phần có giá trị lớn nhất gán cho pixel đang xét"*.

**Trường hợp P = 16 (Tạo ra số 16-bit, lớn nhất là 65535):**
- Tách làm 2 phần 8-bit:
  - `Phần 1` (Low byte): Lấy 8 bit cuối (từ $p=0$ đến $p=7$). Tương đương phép toán: $V_1 = LBP_{16,R} \text{ AND } 255$
  - `Phần 2` (High byte): Lấy 8 bit đầu (từ $p=8$ đến $p=15$). Tương đương phép toán: $V_2 = LBP_{16,R} \gg 8$
- **Kết quả cuối cùng gán cho tâm:** $I_{new} = \max(V_1, V_2)$

**Trường hợp P = 24 (Tạo ra số 24-bit):**
- Tách làm 3 phần 8-bit:
  - `Phần 1`: $V_1 = LBP_{24,R} \text{ AND } 255$
  - `Phần 2`: $V_2 = (LBP_{24,R} \gg 8) \text{ AND } 255$
  - `Phần 3`: $V_3 = LBP_{24,R} \gg 16$
- **Kết quả cuối cùng gán cho tâm:** $I_{new} = \max(V_1, V_2, V_3)$

---

## VÍ DỤ TÍNH TAY CỤ THỂ (P=8, R=1)

Xét ma trận điểm ảnh 3x3 sau, tâm đang xét nằm ở giữa với $I_c = 55$:

| 12 | 50 | 60 |
|----|----|----|
| 14 | 55 | 70 |
| 16 | 20 | 10 |

Với $P=8, R=1$, ta quét 8 điểm xung quanh:

| $p$ | Góc $\theta$ | Vị trí tương đối | Giá trị $I_p$ | So sánh ($I_p \ge 55$) | Bit $s$ | Giá trị $s \times 2^p$ |
|-----|-------------|-------------------|---------------|-------------------------|---------|-------------------------|
| 0 | 0° | Bên phải | 70 | $70 \ge 55$ (Đúng) | 1 | $1 \times 2^0 = 1$ |
| 1 | 45° | Góc trên phải | 60 | $60 \ge 55$ (Đúng) | 1 | $1 \times 2^1 = 2$ |
| 2 | 90° | Bên trên | 50 | $50 < 55$ (Sai) | 0 | 0 |
| 3 | 135°| Góc trên trái | 12 | $12 < 55$ (Sai) | 0 | 0 |
| 4 | 180°| Bên trái | 14 | $14 < 55$ (Sai) | 0 | 0 |
| 5 | 225°| Góc dưới trái | 16 | $16 < 55$ (Sai) | 0 | 0 |
| 6 | 270°| Bên dưới | 20 | $20 < 55$ (Sai) | 0 | 0 |
| 7 | 315°| Góc dưới phải | 10 | $10 < 55$ (Sai) | 0 | 0 |

*(Lưu ý: Với R=1, các điểm lân cận trùng ngay lưới nguyên nên không cần tính công thức nội suy phức tạp, ta lấy thẳng giá trị trên ma trận).*

**Tổng kết:**
Giá trị LBP tại điểm tâm sẽ là:
$LBP_{8,1} = 1 + 2 + 0 + 0 + 0 + 0 + 0 + 0 = 3$

Vậy pixel tại vị trí có giá trị 55 sẽ được cập nhật thành **3** trong ảnh kết quả LBP.
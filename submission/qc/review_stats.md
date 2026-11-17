# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 537; đối tượng sau duyệt: 523; ngưỡng accepted: IoU >= 0.9
- Đã **loại 25 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 384 | 384/537 = 71.5% | 537 box máy |
| edited (sửa) | 58 | 58/537 = 10.8% | 537 box máy |
| deleted (xoá) | 95 | 95/537 = 17.7% | 537 box máy |
| added (vẽ thêm) | 81 | 81/523 = 15.5% | 523 đối tượng sau duyệt |

Trong số deleted + added, **10 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 71.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 174 | 11 | 3 | 33 |
| bicycle | 5 | 0 | 2 | 3 |
| motorbike | 145 | 33 | 60 | 17 |
| car | 47 | 3 | 4 | 21 |
| bus | 3 | 0 | 0 | 3 |
| truck | 1 | 0 | 5 | 1 |
| three_wheeler | 9 | 11 | 21 | 3 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 38 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 34 | 35 | 4 | 49.3% (KTC 95%: 38–61%) | 89.5% (KTC 95%: 76–96%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - wrong_class: 3/3
  - missing: 3/3
  - loose: 1/3
  - lọt: img_025.jpg (loose), img_029.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).

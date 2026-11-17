# Báo cáo Day 08 — 2A202604035 - Nguyễn Huy Thiên Hoà

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person | YOLO26m | 0.50 | Đạt recall 92% (>= 90%) trên gold, precision 55% (12 vật gold). Ngưỡng 0.25 (r02) sinh nhiều box nhiễu, nâng lên 0.50 giúp lọc FP hiệu quả. |
| motorbike | YOLO26m | 0.10 | Đạt recall 90% (>= 90%) trên gold (10 vật gold). Xe máy ở xa hoặc bị che khuất dễ bị tụt conf dưới 0.25, cần ngưỡng 0.10 để bắt đủ xe. |
| three_wheeler | YOLOE-26s, prompt: tuk-tuk, cyclo, cargo tricycle | 0.15 | Đạt recall 100% (7 vật gold), precision 78%. YOLO26 hoàn toàn không biết lớp này (gọi nhầm car/truck ở r02/r03), YOLOE (r04) nhận diện chuẩn xác. |
| car | YOLO26m | 0.30 (mặc định) | Không có ngưỡng nào <= 0.50 đạt recall 90% (tối đa 88% với 8 vật gold), do đó giữ ngưỡng mặc định 0.30 theo khuyến nghị của `threshold_sweep`. |
| bicycle / bus / truck | YOLO26m | 0.30 (mặc định) | Số vật trong tập gold quá ít (< 5 vật: bus=0, bicycle=0, truck=1). Mẫu không đủ ý nghĩa thống kê để sweep riêng, giữ mặc định 0.30. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn:** **SAM 2.1** (`sam2.1_b.pt`). Lý do: SAM 2.1 cân bằng hoàn hảo giữa chất lượng phân đoạn đường biên phức tạp (mui xe tuk-tuk, căm xe, tay lái) và thời gian xử lý (~11s/ảnh trên CPU/GPU phổ thông, trọng số ~160 MB). So với SAM 1 (`mobile_sam.pt` ~40 MB), SAM 2.1 cho mask liền mạch hơn, không bị vỡ vụn thành nhiều polygon rời rạc khi vật bị che một phần. SAM 3 đòi hỏi quyền truy cập gated repo HuggingFace và tài nguyên nặng hơn nên SAM 2.1 là lựa chọn tối ưu và thực tế nhất.
- Lỗi nào **không sửa được bằng đổi ngưỡng**?
  - Hiện tượng YOLO26 gọi tuk-tuk/xe ba gác là `car` hoặc `truck`: Đây là lỗi thiếu hụt danh mục (closed-set taxonomy mismatch) của COCO detector. Đổi ngưỡng conf dù lên 0.9 hay xuống 0.05 cũng không thể biến nhãn `car` thành `three_wheeler`.
  - Cách sửa: Sử dụng model open-vocabulary (YOLOE với text prompt "tuk-tuk", "cyclo", "cargo tricycle") để detect riêng lớp `three_wheeler`, sau đó gộp nhãn bằng cơ chế ưu tiên NMS / IoU merge (`merge_prelabels.py`), đè lên các box `car`/`truck` trùng vị trí của YOLO26.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: Trên `img_013.jpg`, YOLO26 nhận diện chiếc tuk-tuk chiếm nửa khung hình là `car` với độ tin cậy rất cao (conf > 0.85).
  - Điểm thấp mà đúng: Trên `img_011.jpg` và `img_032.jpg`, các xe máy ở xa trong hẻm có conf chỉ 0.12–0.18 nhưng thực tế là xe máy hoàn toàn hợp lệ theo guideline.
- Giới hạn: Quét ngưỡng thực hiện trên chính tập gold (4 ảnh in-sample), do đó số đo precision/recall của máy trên gold mang tính **lạc quan** (optimistic bias). Ở môi trường thực tế với tập test độc lập, chỉ số có thể giảm nhẹ.

## 2. AI error profile (từ `qc/review_stats.md`)

Bảng 4 trạng thái (sau khi đã loại bỏ 25 đối tượng thuộc vùng bẫy nhân tạo):

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 384 | 384/537 = 71.5% | 537 box máy |
| edited (sửa kích thước/vị trí) | 58 | 58/537 = 10.8% | 537 box máy |
| deleted (xoá box thừa/sai) | 95 | 95/537 = 17.7% | 537 box máy |
| added (vẽ thêm vật sót) | 81 | 81/523 = 15.5% | 523 đối tượng sau duyệt |

- Ghi chú: Trong tổng số deleted + added, có **10 trường hợp là đổi lớp** (máy nhận đúng vị trí nhưng gán sai nhãn, ví dụ tuk-tuk bị gọi là car/truck). Số đối tượng máy thực sự bỏ sót là 71 vật (81 - 10).
- Lớp nào máy tệ nhất?
  - `motorbike`: Bị xoá nhiều nhất (deleted = 60/537) và phải sửa nhiều (edited = 33/537), tập trung ở các ảnh có cụm xe dày đặc (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`). Máy hay vẽ các box chồng chéo nhau hoặc nhận nhầm biển hiệu/vật cản ven đường thành xe máy.
  - `three_wheeler`: Bị deleted 21 và edited 11 do xung đột giữa box `car` của YOLO26 và box `three_wheeler` của YOLOE, box của YOLOE đôi khi bao cả hàng hoá nhô ra quá mức.
- So với gold:
  - Precision của máy: **49.3%** (34/69 box máy trên gold; KTC 95%: 38–61%).
  - Recall của máy: **89.5%** (34/38 đối tượng gold; KTC 95%: 76–96%).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Kết quả bắt bẫy: **10/12 bẫy = 83.3%** (KTC 95%: 55–95%), **đạt chuẩn rubric** (ngưỡng yêu cầu ≥ 80%).
  - Bẫy thừa (`extra`): Bắt được 3/3 (100%) — các box thừa trên nền không bị bỏ sót.
  - Bẫy sai nhãn (`wrong_class`): Bắt được 3/3 (100%) — phát hiện và đổi lại nhãn chính xác.
  - Bẫy thiếu (`missing`): Bắt được 3/3 (100%) — vẽ thêm đầy đủ các đối tượng bị cố tình xoá.
  - Bẫy box lỏng (`loose`): Bắt được 1/3 (33.3%). Lọt 2 bẫy box lỏng tại `img_025.jpg` và `img_029.jpg`.
- Vì sao lọt bẫy box lỏng?
  - Nguyên nhân: Đây là biểu hiện điển hình của **thiên kiến tự động hoá (automation bias)**. Khi nhìn lướt trên CVAT ở mức zoom vừa phải, mắt người có xu hướng thấy box đã bao quanh đối tượng là chấp nhận ngay (bấm Next), bỏ qua bước zoom sát 200% để kiểm tra khoảng cách 2–4 px giữa mép box và thân xe/chân người.
- Phân tích mối quan hệ giữa Accepted và Bắt bẫy:
  - Tỉ lệ Accepted đạt 71.5% đi kèm tỉ lệ bắt bẫy 83.3% phản ánh quá trình duyệt thực chất và cẩn trọng. Người duyệt không bấm Accept mù quáng (nếu chấp nhận mù quáng, bắt bẫy sẽ rơi xuống < 50%). Việc bắt được toàn bộ bẫy missing, extra và wrong_class chứng minh quy trình quét ảnh tìm vật sót và kiểm tra nhãn được thực hiện nghiêm túc.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`):
  - Gán tay hoàn toàn (gold): **5.0 phút/ảnh** (20 phút / 4 ảnh).
  - Duyệt có pre-label (batch): **1.92 phút/ảnh** (46 phút / 24 ảnh).
  - Tốc độ tăng gấp **2.6 lần** (~61.6% thời gian được tiết kiệm).
  - Thiên lệch thời gian: Người duyệt gán gold trước nên đã nắm rõ ngữ cảnh và taxonomy, giúp thao tác review batch nhanh hơn; tuy nhiên mức chênh lệch 5.0 vs 1.92 phút/ảnh là thực tế và rất có ý nghĩa trong các dự án gán nhãn quy mô hàng nghìn ảnh.
- Kết luận: **RẤT ĐÁNG DÙNG**. Mặc dù precision thô của máy chỉ ~50% (cần xoá bớt box rác), nhưng recall đạt gần 90% giúp người gán nhãn tiết kiệm phần lớn công sức căn chỉnh toạ độ box từ đầu cho các lớp phổ biến (`person`, `car`, `motorbike`).
- **Ba quy tắc giảm thiên kiến của bạn:**
  1. **Quy tắc 1 (Ẩn nhãn máy quét trước - Step 0):** Luôn tắt hiển thị layer annotations (phím tắt `H` trên CVAT) trong 10 giây đầu tiên để mắt quét tự nhiên toàn bộ bức ảnh, tìm các đối tượng bỏ sót trước khi bật lại nhãn máy. Đo lường: Đảm bảo số lượng box `added` phản ánh đúng vật nhìn thấy bằng mắt.
  2. **Quy tắc 2 (Zoom bắt buộc 200% cho box tiền cảnh):** Không chấp nhận bất kỳ box nào ở cự ly gần/trung bình nếu chưa zoom tối thiểu 200% để kiểm tra 4 mép biên (khắc phục triệt để lỗi lọt bẫy loose box).
  3. **Quy tắc 3 (Nghỉ ngơi định kỳ theo chu kỳ 10 ảnh):** Cứ sau 10 ảnh hoặc 20 phút làm việc liên tục, dừng lại 2 phút để thư giãn mắt và não bộ, tránh trạng thái mệt mỏi dẫn đến thói quen bấm "Accept" liên tục.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không gồm người lái (đã tách người lái thành công ra khỏi xe) | Ban đầu 1 click dương bao luôn cả người lái, cần click âm thứ 3 để loại trừ người lái. Mask mui xe và bánh xe rất chuẩn. | [tuktuk_img013.jpg](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/masks/tuktuk_img013.jpg) |
| `img_003.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Ôm trọn thân xe tuk-tuk, tách biệt khỏi cabin phía trong | Mask bị phân thành 3 polygon do các cột chống mui xe cắt ngang thân xe tạo thành các vùng không liên tục. | [tuktuk_img003.jpg](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/masks/tuktuk_img003.jpg) |
| `img_017.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Bao phủ thân xe và kính xe; không dính xe máy bên cạnh | Mask bám sát đường cong vè xe và đèn pha; mép dưới chạm sát mặt đường không bị lem bóng đổ. | [tuktuk_img017.jpg](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/masks/tuktuk_img017.jpg) |
| `img_013.jpg` (so sánh) | SAM 1 (`mobile_sam.pt`) | 3 (2 pos, 1 neg) | Tách được người lái nhưng viền mui xe bị răng cưa | Diện tích mask chỉ đạt 245k px (so với 317k px của SAM 2.1), viền thô và bỏ sót góc nhọn của cản trước. | [tuktuk_img013_mobilesam.jpg](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/masks/tuktuk_img013_mobilesam.jpg) |

- Đối chiếu guideline mục 4.2:
  - Quy tắc giúp ích: Quy tắc "bóng đổ không thuộc vật" và "hai vật chạm nhau không chồng lấn" giúp định hướng đặt click âm chính xác tại vùng tiếp giáp giữa người lái và thân xe.
  - Quy tắc còn thiếu trong v1.0: Chưa quy định cụ thể việc xử lý kính xe nhìn xuyên thấu (đã được bổ sung vào v1.1).
- So sánh Interactor SAM trong CVAT vs Script offline: Script offline cho phép tinh chỉnh toạ độ điểm chính xác bằng số pixel và kiểm soát phiên bản model (`sam2.1_b.pt`), trong khi CVAT Interactor tương tác trực quan nhanh hơn nhưng phụ thuộc vào serverless Nuclio container và phiên bản model tích hợp sẵn trong CVAT.

## 6. Calibration (Part 6)

- Số quyết định: **4 quyết định** được lưu trong [submission/qc/calibration_decisions.csv](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/qc/calibration_decisions.csv).
  - Quyết định quan trọng nhất: `CAL-01` (chốt định nghĩa `three_wheeler` cho mọi phương tiện ba bánh chở khách/hàng dù có buồng lái hoặc kết cấu giống ô tô/xe tải nhỏ, giải quyết dứt điểm việc máy phân loại nhầm thành car/truck) và `CAL-03` (phân định xe điện hai bánh theo bàn đạp cơ học).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: Đã hoàn thiện toàn bộ trong [submission/guidelines/guideline_v1.1_calibrated.md](file:///D:/Lab/K5-L23-Day08-AI-Assisted-Labeling/submission/guidelines/guideline_v1.1_calibrated.md) với đầy đủ bảng câu hỏi mở, ca khó kèm ảnh minh chứng (`img_013.jpg`, `img_008.jpg`, `img_010.jpg`, `img_025.jpg`).
- Phân tích nguyên nhân gốc:
  - 50% do **giới hạn của mô hình AI** (YOLO26 đóng taxonomy, không có khái niệm xe ba bánh Đông Nam Á).
  - 30% do **guideline v1.0 còn mở** (chưa hướng dẫn xe điện, hàng cồng kềnh nhô ra).
  - 20% do **người duyệt bị thiên kiến tự động hoá** (bỏ qua kiểm tra độ ôm sát mép box dẫn đến lọt bẫy loose).

## 7. QC và quyết định (Part 6)

- Mẫu kiểm tra QC: **6 ảnh** (gồm 1 ca khó `img_013.jpg` do có nhãn `can_review` + 5 ảnh ngẫu nhiên lấy theo seed 42: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_017.jpg`, `img_033.jpg`), tổng cộng **100 đối tượng**.
  - Người kiểm: Học viên (Hạn chế phương pháp luận: người kiểm trùng với người duyệt, có thể tồn tại điểm mù nhận thức chủ quan).
- Tỉ lệ lỗi đo được kèm mẫu số:
  - Lỗi theo **đối tượng**: **0/100 = 0.0%** (đạt ngưỡng cho phép).
  - Lỗi theo **ảnh**: **0/6 = 0.0%** (ngưỡng quy định Release ≤ 2%).
- Top lỗi: Không ghi nhận lỗi vi phạm nghiêm trọng nào trong 6 ảnh mẫu sau khi đã được hiệu chỉnh ở vòng review batch.
- **Giới hạn mẫu nhỏ:**
  - Theo tính toán của `qc_report_auto.md`, với 0 ảnh lỗi trong 6 ảnh mẫu, khoảng tin cậy Wilson 95% có cận trên lên tới **39%**. Do kích thước mẫu nhỏ (6 ảnh), việc đạt 0% lỗi trong mẫu không chứng minh được 100% rằng toàn bộ dataset có tỉ lệ lỗi < 2% về mặt lý thuyết xác suất thống kê. Đây là giới hạn cố hữu của việc lấy mẫu kiểm tra nhanh trong điều kiện thực hành.
- Khả năng của script kiểm tra tự động:
  - Script kiểm tra được: Cấu trúc file, tính hợp lệ của toạ độ hộp, sự tồn tại của nhãn trong taxonomy, thống kê accepted/edited/deleted.
  - Script KHÔNG kiểm tra được: Độ ôm sát thực tế theo pixel của box vào vật thật, việc phân loại sai ngữ cảnh thực tế (ví dụ xe mô hình, người trong tranh), và những vật mà cả người lẫn máy cùng bỏ sót nếu không có nhãn ground truth đối chiếu.

**Quyết định: RELEASE**, ngày 2026-11-17.
- Lý do quyết định: Đi theo cây quyết định Hold → Rework → Release:
  1. Không có vấn đề dữ liệu hỏng, các câu hỏi mở của guideline đã được giải quyết triệt để trong v1.1 (không vi phạm điều kiện HOLD).
  2. Tỉ lệ ảnh lỗi trong mẫu đạt 0.0% (<= 2%), không có lỗi sai sót mang tính hệ thống, tỉ lệ bắt bẫy đạt 83.3% (>= 80%) (không vi phạm điều kiện REWORK).
  3. Đạt tất cả tiêu chí của nhánh **RELEASE**.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Sử dụng trợ lý Antigravity để hỗ trợ thiết lập tự động môi trường, chạy script lệnh, tải mô hình, xuất nhập dữ liệu CVAT API và tính toán thống kê. Toàn bộ quá trình vẽ nhãn gold và review batch được thực hiện trực tiếp bởi học viên trên giao diện CVAT.
- **Gold có bị nhiễm không:** **HOÀN TOÀN KHÔNG BỊ NHIỄM**. 4 ảnh gold được gán thủ công từ đầu (Part 1) trong 20 phút trước khi bất kỳ mô hình AI nào được chạy hay xem trước kết quả.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:** Hệ điều hành Windows 11, GPU NVIDIA GeForce RTX 3050 Laptop (4GB VRAM), CVAT Docker v2.81.0. Quá trình cài đặt diễn ra suôn sẻ nhờ kiểm tra trước môi trường với `check_env.py`.
- **Dùng cache `data/cache/` thay vì tự chạy:** **KHÔNG**. Toàn bộ các lượt chạy YOLO26, YOLOE và SAM đều được thực thi trực tiếp bằng trọng số mô hình trên máy cục bộ.
- **Lỗi / bug script đã gặp:** Cần lưu ý đường dẫn Windows PowerShell khi truyền chuỗi tham số có dấu phẩy hoặc dấu ngoặc kép (`--thr`, `--split`); chạy trực tiếp qua môi trường Python giải quyết triệt để vấn đề phân tách chuỗi.
- **Một điều bạn làm khác lần sau:** Tôi sẽ chủ động zoom lớn 200% ngay từ những ảnh đầu tiên trong checklist review để không bị lọt bất kỳ bẫy loose box nào.

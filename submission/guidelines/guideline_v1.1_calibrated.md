# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202604035 - Nguyễn Huy Thiên Hoà

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Có bàn đạp cơ trợ lực đạp chân: gán `bicycle`. Vận hành thuần động cơ điện không bàn đạp (xe máy điện, scooter điện): gán `motorbike`. | `img_010.jpg`, `img_016.jpg` | `CAL-03` |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Nếu xe đỗ sát nhau nhưng thân xe cao ≥ 20 px và nhìn thấy rõ đầu xe hoặc yên xe: gán từng xe. Chỉ dùng `ignore_region` cho cụm xe ở xa cao < 20 px hoặc bị che khuất > 80%. | `img_005.jpg`, `img_018.jpg` | `CAL-02` |
| 3 | Hàng hoá nhô ra khỏi xe ba gác, xe máy: box xe có gồm hàng không? | Box xe ôm cả giỏ hàng, thùng hàng gắn cố định trên giá xe. Nếu hàng nhô ra khỏi xe quá 20% chiều dài thân xe (như chở cành cây, thanh sắt dài) thì không gán phần nhô đó vào box xe để tránh méo bounding box. | `img_008.jpg`, `img_032.jpg` | `CAL-02` |
| 4 | Người ngồi trong xe ba bánh có mui che: gán `person` khi nào? | Nếu nhìn thấy từ 20% thân người trở lên qua cửa sổ/khung hở của tuk-tuk: gán 1 box `person`. Nếu chỉ thấy một phần đầu lấp ló (< 20% diện tích người): bỏ qua, không đoán. | `img_013.jpg` | `CAL-01` |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe hay không? | Kính xe thuộc diện tích thân xe, do đó mask của xe ôm toàn bộ kính chắn gió (kể cả nhìn xuyên qua kính thấy tài xế/nội thất). Người bên trong vẽ mask `person` riêng. | `img_013.jpg` | `CAL-01` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg`, `img_004.jpg` | Xe tuk-tuk / xe ba gác có kết cấu buồng lái và mui che | YOLO26 gọi nhầm thành `car` hoặc `truck` | Luôn chuyển về `three_wheeler`. Gán cả xe + khung mui vào `three_wheeler`; người lái vẽ box `person` riêng. | Mục 2: Bổ sung định nghĩa `three_wheeler` cho mọi phương tiện 3 bánh dù có mui |
| `img_025.jpg`, `img_029.jpg` | Box máy vẽ thừa khoảng trống ở cạnh dưới hoặc cạnh bên của xe máy / người | Máy tạo box lỏng (IoU < 0.9) mà mắt thường dễ bỏ qua | Phóng to 200% để căn chỉnh 4 cạnh chạm biên ngoài cùng nhìn thấy của vật (sai số ≤ 2px). | Mục 4.1.1: Quy chuẩn kiểm tra mép box ôm sát |
| `img_018.jpg`, `img_024.jpg` | Bãi đỗ xe máy đông đúc ở góc xa | Máy sót nhiều xe nhỏ hoặc nhận diện nhầm chi tiết nền | Sử dụng `ignore_region` bao phủ toàn bộ vùng các xe nhỏ < 20 px thay vì cố vẽ từng vật bị biến dạng. | Mục 4.1.6: Quy tắc cụm vật nhỏ dùng `ignore_region` |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | — |
| v1.1 | Chốt phân loại xe đạp điện/xe máy điện; làm rõ ranh giới box xe khi chở hàng hoá cồng kềnh; chốt quy tắc tuk-tuk mui che và mask kính xe; siết chặt quy trình bắt box lỏng | Giải quyết các câu hỏi mở mục 6, khắc phục lỗi phân loại sai tuk-tuk của YOLO26 (`CAL-01`), chốt quy tắc hàng cồng kềnh (`CAL-02`), phân định xe điện (`CAL-03`), và ngăn ngừa lọt bẫy loose (`CAL-04`) |

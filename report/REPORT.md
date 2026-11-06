# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** 3.13.16/2.11.0+cu130/8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.


1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Vì sao bạn nghĩ đây là một bài toán classification, phân loại các vật thể trong một bức ảnh. Record bounding box hay mask, nên không chỉ ra vật thể nào ở vị trí nào nên không chỉ rõ vị trí của 1 vật thể cab trên ảnh
3. Danh sách tên lớp (class list) đi kèm taxonomy/checkpoint trên dữ liệu của ImageNet-1K, model chỉ dựa trên các dữ liệu có sẵn đó để tạo class name
4. Vì nếu chỉ ghi 1 trường thì người gán nhãn có thể hiểu sai và không phát hiện kịp thời.
5.Vấn đề là bài toán này chỉ cho phép một nhãn mỗi ảnh, trong khi ảnh thật thường có nhiều thứ. Nếu guideline không nói rõ, mỗi người gán nhãn sẽ chọn theo cảm tính và dữ liệu sẽ không nhất quán.
6. Vì score chỉ thể hiện độ tin cậy của model chứ không chắc hoàn toàn là sử thật, sau khi model dự đoán thì score chỉ nghiêng về cab chứ không hoàn toàn chính xác, vẫn có khả năng là model dự đoán sai.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường:
Góc trên-trái ở (x = 385.33, y = 69.24), tức cách mép trái ảnh khoảng 385 pixel (hơn nửa chiều ngang) và cách mép trên khoảng 69 pixel
Góc dưới-phải ở (x = 498.92, y = 348.92), tức cách mép trái khoảng 499 pixel và cách mép trên khoảng 349 pixel
Hộp rộng khoảng 113.6 pixel (498.92 − 385.33) và cao khoảng 279.7 pixel (348.92 − 69.24)
3. Chọn hai ngưỡng 
Ngưỡng 0.20: 17 prediction → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
Ngưỡng 0.60: 6 prediction → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']
4. Khi hạ ngưỡng từ 0.60 xuống 0.20,số box tăng từ 6 lên 17 người **reviewer** phải kiểm tra từng box mới thêm, xóa box bị sai và sửa nhãn cho hợp lý.
5. Vẽ hộp sát quanh toàn bộ phần nhìn thấy được của vật thể: cạnh box chạm điểm ngoài cùng của vật cả bốn phía, không chừa nền thừa và không cắt vào vật.
6. Trên ảnh `kitchen` bếp nấu ở góc trái bị cắt ở mép trái ảnh.viết thành quy tắc trong guideline, không cần hỏi Lab Coach là vật bị cắt mép hoặc che khuất vẫn gán nhãn nếu nhận ra được loại vật; chỉ vẽ hộp quanh phần nhìn thấy, không đoán phần bị che. vì vẫn nhận ra rõ là bếp, và tình huống này có thể lặp lại ở nhiều ảnh, nên cần viết guideline thay vì hỏi từng lần.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[53.0,344.0],[52.0,345.0],[50.0,345.0],[49.0,346.0]
2. So với box, polygon hình dạng thật của vật thể, Diện tích và ranh giới chính xác
3. `instance_id` dùng tham chiếu, chấm điểm và sửa đúng từng mask. Nó **không** phải class ID vì chỉ loại vật thể, nhiều vật dùng chung một giá trị. instance_id phân biệt từng cá thể và **không** phải tracking ID vì Tracking ID giữ một vật thể là cùng một vật qua nhiều khung hình video, instance_id chỉ có nghĩa trong một ảnh tĩnh
4. Đặt các đỉnh polygon đúng trên đường biên nhìn thấy của vật thể, sao cho mask bao trọn phần vật thể nhìn thấy và không lấn sang nền hay vật khác
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Hai vật chạm/chồng nhau: ba cái bát ở góc trái dưới. Dùng guideline.Đĩa trắng (0.74), đĩa bánh nướng có vỏ bánh (0.53) và bát nhân rau (0.70) chồng lên nhau, và bbox của chúng chồng nhau rõ. Quy tắc ở câu 4 đủ để giải quyết. Đĩa trắng nằm trước nên giữ biên đầy đủ, đĩa bánh ở sau thì dừng ở mép đĩa trắng. Chỉ cần xác định cái nào nằm trước, không cần ai quyết định thêm.
Biên mờ: mặt bàn và tấm bột/vỏ bánh phủ bột mì. Dùng  guideline.Màu bột và mặt bàn gỗ gần nhau nên đường ranh khó thấy. Cách xử lý là vẽ theo mép nhìn thấy rõ nhất, và nếu không phân biệt được thì để thuộc bàn. Chỗ này chỉ là chuyện kỹ thuật vẽ biên.
Bị che: bàn (dining table) bị đồ vật trên bàn che. Dùng guideline.Hũ mứt, các đĩa và bát che một phần mặt bàn. Theo quy tắc “chỉ vẽ phần nhìn thấy”, mask bàn chừa lỗ ở những chỗ đó. Ảnh overlay cho thấy mask đang làm đúng như vậy.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn duy nhất đại diện cho toàn bộ ảnh. |Cảnh traffic hỗn hợp nhiều xe (buýt, taxi) nhưng chỉ được gán 1 nhãn duy nhất (cab, score: 0.51).  |Chọn 1 nhãn đúng nhất theo guideline quy định.  | Kiểm tra nhãn được chọn có đúng guideline và khách quan không. |
| Phát hiện vật thể | Bounding box hình chữ nhật [x_min, y_min, x_max, y_max] kèm nhãn. | Ở ảnh kitchen (threshold 0.35), mô hình bỏ sót nhiều xoong chảo treo trên giá và thìa bên phải (chỉ nhận diện được bowl, person...). | Vẽ box ôm sát từng vật thể thuộc taxonomy, tránh vẽ thừa/thiếu. | Kiểm tra độ ôm sát của box, phát hiện lỗi bỏ sót hoặc nhầm nhãn. |
| Instance segmentation | Đa giác (polygon) ôm khít viền đối tượng kèm nhãn. | Ở ảnh kitchen, nhãn cái bàn (dining table) bị đè/gộp chung lên cả các đĩa bánh và lọ mứt phía trên. | Chấm các điểm đa giác chính xác theo đường biên thực tế của vật thể. | Kiểm tra xem polygon có bị lấn biên, vẽ ẩu hoặc gộp các vật thể lại không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu: Chỉ sử dụng 3 ảnh COCO công khai của lab, không tải lên dữ liệu cá nhân hay thông tin bảo mật.
2. Khi gặp dữ liệu sai phạm: Dừng ngay lập tức và báo cáo trực tiếp cho Lab Coach / Mentor hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

# Slide materials — ICBHI 2017

Nội dung nguồn để dựng slide bằng **Gamma**. Rút từ `final/index.ipynb`.

```
slide/
├── slides.md                    # nội dung, 29 card, phân tách bằng ---
├── README.md                    # file này
└── images/
    ├── 01_se-vs-sp.png          # biểu đồ Sp–Se của 24 baseline
    ├── 02_ablation.png          # biểu đồ kết quả 5 dòng ablation
    ├── 03_survey.png            # 27 baseline xếp theo Score tăng dần
    ├── 04_dataset.png           # thống kê bộ dữ liệu, lưới 2x3
    ├── 05_pipeline.png          # sơ đồ pipeline baseline + điểm gắn cải tiến
    ├── 06_beats_official.png    # Hình 1 bài báo BEATs (ICML 2023)
    └── 07_beats_tokenizer.png   # Hình 2 bài báo BEATs (ICML 2023)
```

## Nhập vào Gamma

1. Gamma → **Create new** → **Import** → **Paste in text**
2. Dán toàn bộ nội dung `slides.md`
3. Ở bước tuỳ chọn, chọn chế độ giữ nguyên cấu trúc văn bản
   *(Gamma có tuỳ chọn cho phép nó tự viết lại nội dung — **tắt đi**, nếu không nó sẽ diễn giải
   lại các con số và làm sai số liệu.)*
4. Mỗi `---` thành một card riêng

## Chèn ảnh thủ công

Gamma không đọc được đường dẫn ảnh cục bộ, nên bảy ảnh phải tải lên bằng tay. Trong `slides.md`
chỗ cần chèn được đánh dấu bằng dòng `🖼 **Chèn ảnh:** ...` — xoá dòng đó sau khi chèn xong.

| Card | Ảnh |
|---|---|
| *Thống kê bộ dữ liệu* | `images/04_dataset.png` |
| *Toàn cảnh: Score tăng dần* | `images/03_survey.png` |
| *BEATs được tiền huấn luyện thế nào* | `images/06_beats_official.png` |
| *Acoustic tokenizer sinh nhãn rời rạc thế nào* | `images/07_beats_tokenizer.png` |
| *Pipeline của baseline* | `images/05_pipeline.png` |
| *Phát hiện 2: sensitivity mới là nút thắt* | `images/01_se-vs-sp.png` |
| *Kết quả* | `images/02_ablation.png` |

## Nguồn ảnh

`06_beats_official.png` (Hình 1) và `07_beats_tokenizer.png` (Hình 2) **trích từ bài báo gốc**
của BEATs (Chen et al., ICML 2023, arXiv:2212.09058), render lại từ PDF ở độ phân giải cao. Hai
card tương ứng trong `slides.md` đều có dòng ghi nguồn — **giữ nguyên khi dựng slide.**
Năm ảnh còn lại do notebook của đồ án sinh ra.

## Ghi chú khi trình bày

**Mạch chuyện:** đặt giả thuyết → kiểm chứng → giả thuyết bị bác bỏ → phân tích vì sao.
Kết quả âm, nhưng ba phát hiện phụ mới là phần có giá trị. Đừng dừng ở card *"Giả thuyết bị bác
bỏ"* — ba card phân tích ngay sau đó mới là trọng tâm.

**Ba card cần nhấn:**

- *Độ đo không phải accuracy* — mô hình vô dụng vẫn đạt accuracy 52,80 %. Đặt nền cho mọi thứ sau đó.
- *Đọc kỹ hơn: hướng B thực sự đẩy được Se* — chuyển từ "thất bại" sang "cơ chế đúng, hoàn cảnh sai".
- *Vì sao (1)* — trọng số in ra là `1.006, 0.998...`, tức gần như bằng 1. Bằng chứng trực tiếp
  nhất cho thấy tiền đề ban đầu sai.

**Câu hỏi nhiều khả năng bị hỏi:**

| Câu hỏi | Trả lời ngắn |
|---|---|
| Sao không dùng model đã train sẵn của PAFA? | Họ không phát hành. Và kể cả có, dòng 0 phải đi qua **đúng pipeline của ta** thì hiệu số mới có nghĩa. |
| 1 seed thì kết luận được gì? | Không kết luận được cải tiến có hại; chỉ kết luận được **không có bằng chứng** nó có lợi. Đã nêu ở phần hạn chế. |
| Sao Score thấp hơn bài báo (62,71 so với 63,49)? | 1 seed, và điểm vận hành khác hẳn — Sp thấp hơn 5,7 nhưng Se cao hơn 4,1. |
| Vậy đồ án có kết quả gì? | Ba phát hiện về **bản thân benchmark**, đều kiểm chứng được và đều không thể rút ra từ việc đọc bài báo. |

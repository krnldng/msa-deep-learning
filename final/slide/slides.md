# Phân loại âm thanh hô hấp ICBHI 2017

Benchmark các mô hình cơ sở và thử nghiệm cải tiến

Đồ án cuối kỳ — Deep Learning

---

## Bài toán

Phân loại **từng chu kỳ thở** thành 4 lớp từ âm thanh ống nghe

| | |
|---|---|
| Dữ liệu | ICBHI 2017 — 920 bản ghi, 126 bệnh nhân, ~5,5 giờ |
| Đơn vị phân loại | 6.898 chu kỳ thở |
| 4 lớp | normal · crackle · wheeze · both |
| Phân chia | Official 60/40 — 4.142 train / 2.756 test |

Mất cân bằng nặng: normal 3.642 · crackle 1.864 · wheeze 886 · **both chỉ 506**

---

## Thống kê bộ dữ liệu

🖼 **Chèn ảnh:** `images/04_dataset.png`

Sáu bảng số, mỗi bảng ứng với một quyết định thiết kế hoặc một kết luận về sau

---

## Ba điều bảng số đã báo trước

**Train đã cân bằng, test thì không** — 49,8 % normal ở train so với 57,3 % ở test.
Đây là gốc rễ khiến hai cải tiến của hướng B thất bại *(quay lại ở cuối bài)*

**Thiết bị mang theo phân bố lớp** — Meditron 71 % normal, AKGC417L chỉ 44 %.
Đưa thiết bị vào làm đặc trưng là mời mô hình đi đường tắt

**96,6 % chu kỳ ngắn hơn 5 giây** — trung bình chỉ 2,70 s.
Nên phải lặp lại tín hiệu cho đủ cửa sổ, không đệm số 0 *(đệm 0 tạo khoảng lặng không có thật)*

---

## Độ đo của ICBHI: không phải accuracy

**Sp** (Specificity) = tỉ lệ chu kỳ *normal* phân loại đúng

**Se** (Sensitivity) = tỉ lệ chu kỳ *bất thường* phân loại đúng, gộp cả 3 lớp

**ICBHI Score = (Sp + Se) / 2**

Thử với mô hình "luôn đoán normal":

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| Accuracy | 52,80 % | nghe được |
| Sp / Se | 100 / **0** | không phát hiện ca bệnh nào |
| ICBHI Score | **50,00** | phơi bày ngay thất bại |

---

## Cái bẫy: kết quả công bố trải từ 50 % đến 99 %

Phần lớn chênh lệch đến từ **giao thức đánh giá**, không phải chất lượng mô hình

| Giao thức | Score | So sánh được? |
|---|---|---|
| Official 60/40, tách theo bệnh nhân | 50–66 % | Đúng benchmark |
| Random 80/20, tách theo bệnh nhân | 65–75 % | Dòng nghiên cứu khác |
| Chia ngẫu nhiên ở mức chu kỳ | 90–99 % | Rò rỉ bệnh nhân |

Bài nào công bố ">95 % trên ICBHI" là đang học thuộc bệnh nhân, không học bệnh lý

---

## Khảo sát các mô hình hiện có

**27 phương pháp công bố từ 2020 đến 2026**, tất cả trên cùng giao thức Official 60/40

Cách tổng hợp: lấy từ 4 bài báo cùng tái lập kết quả của nhau — BTS, Ensemble-KD, QLung, PAFA — nên các dòng nhất quán với nhau, không phải gom góp từ những abstract rời rạc

Chỉ nhận vào bảng nếu bài báo:

- Dùng đúng file phân chia chính thức
- Báo cáo đủ cả Sp, Se và Score
- Trên bài toán 4 lớp ở mức chu kỳ thở

---

## Toàn cảnh: Score tăng dần

🖼 **Chèn ảnh:** `images/03_survey.png`

Ba thế hệ backbone tách thành ba cụm rõ rệt:

- **CNN / ResNet (2020–23):** 49,55 → 58,29 — chạm trần dưới 60
- **AST (2023–24):** 59,55 → 62,37 — nhảy một bậc nhờ transformer tiền huấn luyện
- **BEATs / CLAP / SSM (2024–26):** 62,56 → 66,49 — bậc thứ hai

Ranh giới không nằm ở phương pháp mà ở **backbone**: mọi mô hình trên 62 điểm đều dùng nền tiền huấn luyện quy mô lớn

---

## Những mốc chính

| Phương pháp | Năm | Backbone | Score |
|---|---|---|---|
| RespireNet | 2021 | ResNet34 | 56,20 |
| AST + CE | 2023 | AST | 59,55 |
| AST + Patch-Mix CL | 2023 | AST | 62,37 |
| **BEATs + CE** | 2025 | **BEATs** | **63,49** |
| BTS | 2024 | CLAP | 63,54 |
| PAFA | 2025 | BEATs | 64,84 |
| Meta-Ensemble (30× compute) | 2026 | — | 66,49 |

Trần thực tế cho mô hình đơn: **64,5–65,5**. Con số 66,49 cần ensemble tốn gấp 30 lần.

Một mô hình đơn đạt **63–65 là kết quả cạnh tranh được**.

---

## Phát hiện 1: backbone quan trọng hơn phương pháp

**BEATs + cross-entropy thuần đạt 63,49**

- Cao hơn AST + Patch-Mix (62,37), vốn là SOTA được trích dẫn nhiều nhất
- Ngang BTS (63,54) mà không cần cơ chế contrastive nào

Bốn năm cải tiến phương pháp trên nền AST thực chất bị nghẽn ở **backbone**

**Hệ quả:** mốc phải vượt là **63,49**, không phải 62,37. Vượt 62,37 chẳng chứng minh được gì nếu chỉ cần đổi backbone là đã vượt.

---

## BEATs được tiền huấn luyện thế nào

🖼 **Chèn ảnh:** `images/06_beats_official.png`

Hai thành phần **chưng cất lẫn nhau qua nhiều vòng lặp** (iter1 → iter2 → iter3):

- **Acoustic Tokenizer** sinh nhãn rời rạc cho âm thanh không nhãn
- **Audio SSL Model** học đoán token bị che — masked audio modeling
- Mỗi vòng, mô hình SSL dạy lại tokenizer, rồi tokenizer sinh nhãn tốt hơn cho vòng sau

*Nguồn: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023 — Hình 1 (arXiv:2212.09058)*

---

## Acoustic tokenizer sinh nhãn rời rạc thế nào

🖼 **Chèn ảnh:** `images/07_beats_tokenizer.png`

Hai loại tokenizer, dùng ở các vòng lặp khác nhau:

**(a) Random-Projection** — dùng ở **vòng 1**. Chiếu spectrogram qua một lớp ngẫu nhiên, tra láng giềng gần nhất trong codebook **đóng băng** → token. Không học gì cả, chỉ để khởi động.

**(b) Self-Distilled** — dùng ở **vòng 2 và 3**. Codebook **học được**; tokenizer bị ép khớp đầu ra của *Teacher* (chính là mô hình SSL của vòng trước) qua cosine similarity, gradient truyền ngược qua bước rời rạc bằng **straight-through**.

Chính (b) là mũi tên *Knowledge Distillation* trong sơ đồ trước — mô hình SSL dạy lại tokenizer, tokenizer sinh nhãn tốt hơn cho vòng sau.

*Nguồn: Chen et al., ICML 2023 — Hình 2 (arXiv:2212.09058)*

---

## Vì sao BEATs khác AST

| | AST (2021) | BEATs (2023) |
|---|---|---|
| Tiền huấn luyện | **Có giám sát** trên 527 nhãn AudioSet | **Tự giám sát**, đoán token bị che |
| Khởi tạo | Từ ViT của ImageNet — ảnh | Từ chính dữ liệu âm thanh |
| Tham số | ~86 M | ~90 M |
| AudioSet mAP | 45,9 | **48,6** |

**Giả thuyết** giải thích vì sao BEATs hợp với bài này hơn: tiền huấn luyện tự giám sát học cấu trúc âm học mịn hơn, thay vì bị định hình bởi 527 nhãn sự kiện — mà crackle và wheeze thì chẳng giống lớp nào trong AudioSet

---

## Phát hiện 2: sensitivity mới là nút thắt

🖼 **Chèn ảnh:** `images/01_se-vs-sp.png`

Biểu đồ Sp–Se của toàn bộ 24 phương pháp, kèm các đường đẳng Score

- Mọi phương pháp đều đạt Sp 63–88 nhưng Se chỉ 18–49
- **Chưa phương pháp công bố nào vượt Se = 50**
- Trong khi độ đo lại cân bằng Sp và Se
- Tiến bộ từ 2020 đi sang phải (Se), không đi lên (Sp)

---

## Hướng cải tiến đã chọn: A + B

| | Ý tưởng | Bằng chứng trước đó |
|---|---|---|
| **A** | Đưa metadata (thiết bị, vị trí nghe, tuổi, giới) vào BEATs qua FiLM | BTS đạt +0,98 trên CLAP |
| **B** | Hàm loss cân bằng hai nhóm, khớp đúng công thức Score | QLung đạt +2,46 trên AST |

**Lập luận của hướng B:**

ICBHI Score = ½ × recall(normal) + ½ × recall(bất thường gộp)

Vậy hàm loss khớp độ đo là **CE cân bằng theo nhóm** — không phải focal loss, cũng không phải CE cân bằng 4 lớp

Mô hình đề xuất: **MC-BEATs**

---

## Thiết lập thí nghiệm

Bảng khảo sát loại bỏ — mỗi dòng đúng **một** thay đổi có kiểm soát

| Dòng | Cấu hình |
|---|---|
| 0 | BEATs + CE — mốc đối chứng, tự train lại |
| 1 | + group-balanced loss |
| 2 | + hiệu chỉnh prior (hậu kỳ, không train lại) |
| 3 | + metadata nối trực tiếp |
| 4 | **MC-BEATs** — metadata qua FiLM |

Siêu tham số lấy đúng theo PAFA · chạy trên Colab Pro · checkpoint tự khôi phục khi đứt phiên

---

## Pipeline của baseline

🖼 **Chèn ảnh:** `images/05_pipeline.png`

Tám bước, và bốn dòng cải tiến gắn vào ba điểm khác nhau trên cùng một đường ống

---

## Ba quyết định trong pipeline đáng chú ý

**Cache waveform 16 kHz, không cache spectrogram** — BEATs tự tính fbank 128 mel bên trong. Cache phổ sẽ phình từ 600 MB lên 1,4–2,8 GB mà không được gì

**Chu kỳ ngắn thì lặp lại, không đệm số 0** — 96,6 % chu kỳ ngắn hơn 5 giây. Đệm 0 tạo ra khoảng lặng nhân tạo không có trong âm thanh thật

**Không dùng float16** — BEATs sinh `loss = NaN` dưới fp16. Khi đó `argmax` của NaN luôn trả về lớp 0, mô hình "luôn đoán normal" và cho Score **đúng 50,00**: trông như đã train nhưng vô nghĩa

---

## Kiểm tra giao thức phát hiện 3 lỗi dữ liệu

Notebook dùng `assert` để xác minh số chu kỳ và trùng bệnh nhân, thay vì tin tưởng

| Lỗi trong bộ dữ liệu | Hậu quả nếu bỏ qua |
|---|---|
| Zip chứa file .txt không phải chú thích | Crash khi phân tích tên file |
| Split chính thức không tách hoàn toàn theo bệnh nhân (156, 218) | Tưởng nhầm là lỗi pipeline |
| Bản ghi `226_1b1_Pl` ghi sai tên thiết bị trong file split | Mất **11 chu kỳ**, train còn 4.131 |

Không có bước kiểm tra này, ta đã train trên 4.131 chu kỳ rồi so sánh với các con số tính trên 4.142

---

## Kết quả

🖼 **Chèn ảnh:** `images/02_ablation.png`

| Dòng | n_seed | Sp | Se | Score | Δ |
|---|---|---|---|---|---|
| 0 · BEATs + CE | **2** | 69,63 | 53,74 | **61,69 ± 1,45** | — |
| 1 · group-balanced | 1 | 71,94 | 53,36 | 62,65 | +0,96 |
| 2 · hiệu chỉnh prior | 1 | 68,71 | 54,80 | 61,76 | +0,07 |
| 3 · metadata concat | 1 | 68,52 | 54,29 | 61,41 | −0,28 |
| 4 · **MC-BEATs** | 1 | 72,77 | 49,28 | 61,02 | −0,66 |

Thanh sai số của dòng 0 **phủ trùm điểm số của cả bốn dòng còn lại**

---

## Nhiễu lớn hơn tín hiệu

Không phải "cải tiến thất bại", cũng không phải "cải tiến thành công".
**Thí nghiệm chưa đủ sức trả lời câu hỏi.**

Hai seed của **cùng một cấu hình** ở dòng 0:

| | Sp | Se | Score |
|---|---|---|---|
| seed 1 | 73,08 | 52,34 | **62,71** |
| seed 2 | 66,18 | 55,14 | **60,66** |
| chênh lệch | 6,90 | 2,80 | **2,05** |

| | Giá trị |
|---|---|
| Chênh lệch giữa hai seed, **cùng cấu hình** | **2,05 điểm** |
| Hiệu ứng lớn nhất giữa các **phương pháp** | **0,96 điểm** |

Nhiễu gấp hơn hai lần tín hiệu

---

## Thêm một seed, bảng đảo chiều

| Dòng | Với dòng 0 **1 seed** | Với dòng 0 **2 seed** |
|---|---|---|
| 1 · group-balanced | −0,06 | **+0,96** |
| 2 · hiệu chỉnh prior | −0,95 | **+0,07** |
| 3 · metadata concat | −1,30 | −0,28 |
| 4 · MC-BEATs | −1,69 | −0,66 |

**Không có gì trong mô hình thay đổi** — chỉ có mẫu thống kê thay đổi.

Khi nhiễu lớn hơn hiệu ứng, ta có thể dựng ra lời giải thích hợp lý cho **bất kỳ** chiều nào của kết quả. Phải đủ seed **trước khi** diễn giải, không phải sau.

> Một seed không phải là một kết quả

---

## Vì sao (1): hàm loss gần như không làm gì

Trọng số thực tế in ra: `[1.006, 0.998, 0.998, 0.998]` — gần như **bằng 1**

| | normal | bất thường | tỉ lệ normal |
|---|---|---|---|
| Toàn bộ dữ liệu | 3.642 | 3.256 | 52,8 % |
| **Tập train** | ~2.063 | ~2.079 | **49,8 %** |

**Tiền đề của hướng B sai.** Tập train đã cân bằng gần như hoàn hảo giữa hai nhóm.

Mất cân bằng thật nằm **bên trong** nhóm bất thường (`both` chỉ có 506) — đúng thứ mà loss cân bằng-hai-nhóm cố ý không đụng tới.

Bài học: kiểm tra phân bố lớp trên **đúng tập train**, đừng suy từ thống kê toàn bộ

---

## Vì sao (2): train và test lệch phân bố lớp

| | Tỉ lệ normal |
|---|---|
| Train | **49,8 %** |
| Test | **57,3 %** |

Kết quả hiệu chỉnh prior: val chọn δ = −0,20 → **61,76** · oracle chọn δ = +0,55 → **63,82**

- δ tối ưu là **dương**: phải đoán `normal` nhiều hơn — ngược hẳn kỳ vọng ban đầu
- Tập val cắt từ train nên mang phân bố của train, chỉ sai hướng
- Oracle cho thấy **+1,17 vẫn đang nằm trên bàn** nếu hiệu chỉnh đúng cách

---

## Vì sao (3): hiệu năng phụ thuộc thiết bị tới 12,7 điểm

| Thiết bị | n (test) | Sp | Se | Score |
|---|---|---|---|---|
| Meditron | 459 | 89,81 | 37,24 | **63,53** |
| AKGC417L | 1.836 | 70,32 | 52,90 | 61,61 |
| Litt3200 | 461 | 63,38 | 38,24 | **50,81** |

- Khoảng cách **lớn hơn mọi hiệu ứng** đang được so sánh trong bảng ablation
- Litt3200 gần như vô dụng — chỉ nhỉnh hơn mô hình "luôn đoán normal" (50,00)
- Giải thích vì sao metadata gây hại: thiết bị mang quá nhiều thông tin về **nhóm bệnh nhân**, mời mô hình đi đường tắt

---

## Hạn chế

- **Số seed quá ít** — dòng 0 có 2, các dòng khác chỉ 1. Đây không còn là hạn chế bên lề mà **chính là kết quả**
- **σ = 1,45 ước lượng từ đúng 2 mẫu** — bản thân con số này cũng rất không chắc chắn. Cần ≥ 5 seed
- **Chọn epoch tốt nhất dựa trên tập test** — quy ước của cả lĩnh vực, và có thể chính là nguồn gây phương sai lớn: chọn cực trị trên test khuếch đại nhiễu
- **File phân chia chính thức rò rỉ nhẹ** — 2 trên 126 bệnh nhân
- **Baseline là trích dẫn**, chỉ dòng 0 được train lại

---

## Kết luận

Ba kết luận **không phụ thuộc vấn đề seed**, vì đọc thẳng từ dữ liệu chứ không từ so sánh giữa các lần chạy:

1. Tập train ICBHI **đã cân bằng hai nhóm** — nên hàm loss khớp hoàn hảo với độ đo lại hoá ra vô nghĩa
2. Train và test **lệch phân bố lớp** — mọi hiệu chỉnh dựa trên val cắt từ train đều đi sai hướng
3. Hiệu năng **phụ thuộc thiết bị 12,7 điểm** — nhấn chìm mọi khác biệt giữa các phương pháp

Và kết luận thứ tư, về phương pháp luận:

**Phương sai giữa các seed lớn hơn khoảng cách giữa các phương pháp đang cạnh tranh.**
Điều đó đặt dấu hỏi cho mọi bài báo báo cáo mức cải thiện dưới 1 điểm — mà nhìn bảng §3, phần lớn khoảng cách giữa các phương pháp SOTA đều dưới ngưỡng đó.

---

## Hướng đi tiếp

Xếp theo tỉ lệ kỳ vọng thu được trên công sức, dựa trên chính số liệu đã có

1. **Chạy đủ 5 seed cho dòng 0 và dòng 4** — việc này chặn tất cả những việc còn lại
2. **Xem lại cách chọn mô hình** — dùng trung bình N epoch cuối thay vì cực đại trên test, ổn định hơn nhiều
3. **Hiệu chỉnh theo độ lệch prior train sang test** — oracle chỉ ra +1,17, không cần train lại
4. **Tấn công mất cân bằng bên trong nhóm bất thường** — `both` chỉ có 506 chu kỳ
5. **Thích ứng theo thiết bị** — khe hở 12,7 điểm là lớn nhất mà số liệu chỉ ra

---

## Cảm ơn

Notebook đầy đủ: `final/index.ipynb`

Khảo sát baseline: `final/docs/01_baselines.md`

Phân tích hướng cải tiến: `final/docs/02_directions.md`

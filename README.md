# Nhận dạng chữ viết tay tiếng Việt

## Thí nghiệm riêng 70/15/15

Lệnh và hướng dẫn ở `test_701515`. Các file `test`, `test1`, `test2`, `test3`, trainer triển khai và model cũ giữ nguyên.

```powershell
python -m src.experiments.run_701515 --background
```

Lệnh tạo snapshot và chạy **tuần tự** chữ đơn, chữ ghép, seq. Nếu lượt thử đã được khởi động, chỉ xem `reports/split_701515/experiment_state.json` và `experiment.log`; không chạy lại để tạo trùng. Nếu bị dừng:

```powershell
python -m src.experiments.run_701515 --resume --background
```

Ngân sách mặc định: chữ đơn 300 epoch, ghép 100, seq 160, có dừng sớm theo validation. Dữ liệu thử và model có tên riêng:

| Model thử | Train | Validation | Test |
|---|---:|---:|---:|
| `models/baseline_701515` | 2.071 | 444 | 444 |
| `models/seq/word_701515` | 77.521 | 16.612 | 16.611 |
| `models/ghep_701515` | 3.947 | 846 | 846 |

Đây là 70/15/15 theo quota nguyên gần nhất, không thể luôn có đúng số thập phân vì mẫu phải nguyên. Tổng số mẫu mỗi model bằng lượt gốc. Seq chia toàn bộ người viết vào một tập (129/35/37 người); không còn split chính thức VNOnDB. Chữ đơn chỉ một người, chia có stratify và giữ các nét trùng hoàn toàn trong train; chưa có kiểm chứng người viết mới.

Chữ ghép chia raw theo người/style trước, rồi sinh lại dữ liệu phái sinh từ glyph thuộc split chữ đơn mới tương ứng. Không chuyển những mẫu composed cũ sang tập khác vì các mẫu đó có thể dùng chung nét nguồn. Source `composed` được báo cáo riêng; cùng một người viết nhưng không trùng glyph nguồn. Báo cáo nguồn, từng mẫu và giao test cũ/mới nằm dưới `reports/split_701515/`.

**Khởi tạo ngẫu nhiên, không dùng model cũ:** sau khi đổi split, model cũ có thể đã học mẫu test mới. Mean/variance, SVM, từ điển và tham số kết hợp chỉ fit trên train/validation mới tương ứng. Kết quả chạy 1 epoch dùng kiểm tra kỹ thuật không phải chất lượng model cuối; lượt đầy đủ dùng resume để tiếp tục đúng khởi tạo ngẫu nhiên đó.

Trainer của thí nghiệm là bản đóng băng trong `reports/split_701515/workspace/src`, cấu hình riêng để dùng snapshot mới và có retry ghi file Windows. Không sửa code của các trainer triển khai. Manifest `original_files.json` lưu SHA256 của model/cấu hình/trainer cũ; kiểm tra:

```powershell
python -m src.experiments.prepare_701515 --verify-originals
```

Sau khi cả ba model hoàn tất, chương trình so sánh trên **cùng mẫu thuộc giao test cũ và test mới**, tránh đánh giá model cũ trên dữ liệu cũ từng train. Chữ ghép chỉ so giao raw, không dùng composed mà glyph nguồn có thể bị model cũ thấy. Giao test này nhỏ hơn tổng test, nên đọc kèm số mẫu. Không dùng kết quả test để chọn checkpoint hoặc alpha.

Đã có tiến trình theo dõi riêng cho hai model `ghep_701515` và `word_701515`. Nó chờ train, fit SVM và đánh giá xong, sau đó nạp lại các model và gọi đường nhận dạng demo. Chỉ khi các kiểm tra thành công mới tạo `reports/split_701515/READY_TO_TEST.txt` và hiện thông báo Windows **“ĐÃ HOÀN THÀNH”**. Nếu tiến trình train dừng bất thường, watcher thử tiếp tục tối đa 3 lần từ checkpoint và báo lỗi nếu chưa hoàn thành. Nó không tự gửi tin nhắn vào cuộc trò chuyện ChatGPT.

```powershell
Get-Content reports\split_701515\completion_watch.json
# Khi hoàn thành:
Get-Content reports\split_701515\READY_TO_TEST.txt
```

Nếu cần khởi động lại watcher sau khi đã đóng máy, chạy `python -m src.experiments.notify_701515 --background`. Không thay đổi trạng thái model thành hoàn tất chỉ vì đã có một checkpoint; thông báo hoàn tất yêu cầu cả hai pipeline đánh giá xong.

**Giới hạn so sánh:** model cũ dùng chuyển trọng số, lượt 70/15/15 học từ đầu; các tập test cũng khác. Không kết luận mức tăng/giảm chỉ do tỷ lệ. Nếu cần nghiên cứu riêng tác động của tỷ lệ thì phải có control học từ đầu với split cũ, cùng seed/ngân sách và nhiều lần lặp.

Quay lại như cũ không cần sửa trainer: chỉ chạy các lệnh demo gốc:

```powershell
python -m src.serve --run-name baseline_stable
python -m src.seq.serve_seq --run-name word_stable
python -m src.ghep.serve_ghep --run-name ghep_handwriting_v4
```

Các model thử không được tự thay thành model chính. Dữ liệu/model/báo cáo sinh ra tiếp tục được `.gitignore` loại khỏi Git.

Model `seq` nhận chuỗi nét bút của **một từ**. Model `baseline` nhận **một ký tự**. Ảnh chụp canvas không chứa thứ tự/hướng nét để phát lại trực tiếp cho các model này.

## Chữ ghép: ghep_stable (Python 3.12.10)

### Bản sửa kiểu viết thực tế: ghep_handwriting_v4

Lỗi `th` bị nhận thành `anh/yê` cho thấy điểm cao trên tập VNOnDB có nhãn trùng không phủ được những lớp chỉ có dữ liệu máy sinh. Dữ liệu `ghep_stable` có 7 mẫu synthetic cho `th`, trong khi `anh` có thêm 193 mẫu human trong train. Vì thế, các metric tổng hợp trước đây không đủ để khẳng định model đọc tốt chữ ghép do bạn viết bằng chuột.

`ghep_handwriting_v4` dùng thêm 2.368 mẫu **phái sinh từ các chữ đơn trong tập train**, giữ riêng khỏi human. Có 16 biến thể mỗi lớp, giữ chiều cao/baseline và nét mũ/móc của glyph gốc. Không lấy glyph validation/test làm nguyên liệu train. Mẫu ghép từ chữ đơn chỉ mở rộng kiểu viết rời; không thay thế dữ liệu chữ ghép liền tay của nhiều người.

V4 bổ sung 444 mẫu validation phái sinh từ glyph thuộc val chữ đơn, không trùng glyph nguồn train. Hai nhóm vẫn cùng một người viết: đây là kiểm chứng khác mẫu chữ, chưa phải độc lập người viết. Source `composed` được báo cáo riêng, không gộp vào `human`. V3 là lượt thăm dò đã dừng; lượt v4 đang chạy chuyển tiếp từ trọng số cuối v3.

Mỗi epoch lấy cùng số mẫu cho từng lớp (`--samples-per-class 16`) để giảm ưu thế của các lớp human thường gặp. Nhánh hình học có `tanh` giới hạn đặc trưng chuẩn hóa quá lớn trên kiểu chữ lạ. SVM hiệu chỉnh bớt prior lớp đông mẫu khi có dữ liệu phái sinh. Các bản model cũ tiếp tục dùng đúng cách tiền xử lý/giải mã cũ.

Trong phiên hỗ trợ, lượt train nền mới đã được khởi động. Xem tiến độ trước khi chạy lệnh tạo mới:

```powershell
python -m src.ghep.status --run-name ghep_handwriting_v4
Get-Content models\ghep_handwriting_v4\train.log -Tail 20 -Wait
```

Lệnh tạo run này trên máy khác (không chạy trùng khi nó đang train):

```powershell
python -m src.ghep.run_full --run-name ghep_handwriting_v4 --epochs 100 --init-from ghep_stable --composed-per-class 16 --glyph-validation-per-class 3 --samples-per-class 16 --robust-geometry --lr 0.0001 --patience 20 --lr-patience 6
```

Dừng sớm và giảm learning rate vẫn theo validation. Một ảnh `th` được **dựng lại thủ công từ screenshot**, không phải nét gốc của người dùng, được giữ ở `reports/ghep_th_screenshot_regression.json` để kiểm tra; không đưa vào train hoặc dùng chọn checkpoint. Chưa thể coi kết quả trên mẫu dựng lại là xác nhận ảnh gốc đã được nhận đúng.

Sau train, dừng demo cũ bằng Ctrl+C, rồi mở bản mới:

```powershell
python -m src.ghep.serve_ghep --run-name ghep_handwriting_v4
```

Web có mục **“Model đoán sai? Lưu nét đang viết với nhãn đúng”**. Nhập mã người viết nhất quán, chọn nhãn thực sự đã viết, rồi bấm **Lưu mẫu train**. Nét gốc được lưu thành InkML ở `dataset/raw/inkml_ghep/browser_corrections/`, chỉ có split `train`. Đáp án bài tập không tự động thành nhãn, không có thao tác tự gửi mẫu. Model đang mở không học ngay; run mới sẽ nạp các mẫu này. Khi tạo bộ kiểm chứng thật, cần thu người viết khác khỏi tập train.

Tiếp tục run bị gián đoạn:

```powershell
python -m src.ghep.run_full --run-name ghep_handwriting_v4 --resume --epochs 100
```

Chỉ chạy sau khi tiến trình cũ đã dừng. Để học mẫu sửa mới thu được, tạo **tên run mới** và khởi tạo từ checkpoint cũ, vì resume giữ nguyên snapshot dữ liệu.

### Pipeline ghep_stable trước đây (tham khảo)

Các lệnh bên dưới mô tả run cũ. File `test3` hiện trỏ sang v4 ở phần trên. Để tạo run cũ trên bộ dữ liệu ban đầu, lệnh từng dùng là:

```powershell
python -m src.ghep.run_full --run-name ghep_stable --epochs 400 --init-from ghep
```

Pipeline tạo dữ liệu riêng, train CNN + hình học, fit SVM trên train, chọn hệ số kết hợp trên validation, sau đó mới đánh giá test. Lượt train dài **không tự khởi động** khi sửa code; bạn chạy lệnh trên. Model và dữ liệu `ghep` cũ được giữ nguyên. `ghep_verify` chỉ là lượt kiểm tra kỹ thuật, không phải lượt train đủ 400 epoch.

Sau khi hoàn tất, dừng demo cũ bằng Ctrl+C nếu đang chiếm cổng 8002, rồi chạy:

```powershell
python -m src.ghep.serve_ghep --run-name ghep_stable
```

Mở `http://127.0.0.1:8002`. Web, Tkinter và đánh giá đều dùng cùng đường nhận dạng, bảng nhãn và tiền xử lý của run. Đáp án bài mẫu không được gửi cho model. Các tỷ lệ hiển thị là điểm model, chưa phải xác suất viết đúng đã hiệu chỉnh.

Theo dõi / tiếp tục nếu bị dừng:

```powershell
python -m src.ghep.status --run-name ghep_stable
python -m src.ghep.run_full --run-name ghep_stable --resume --epochs 400
```

Chỉ chạy `--resume` khi lượt cũ đã dừng. Checkpoint khôi phục model, optimizer, learning rate, bộ đếm giảm learning rate/dừng sớm và lịch sử trong cùng tệp; epoch bị dừng giữa chừng sẽ được chạy lại. Tạo `models/ghep_stable/STOP` để dừng có kiểm soát; resume xóa tệp này. Thêm `--background` vào lệnh train để chạy nền và đọc `models/ghep_stable/train.log`.

Những thay đổi chính:

- CNN ảnh 64×192 giữ thông tin vị trí, GroupNormalization và dropout 0.35. Nhánh hình học nhìn hướng nét theo ô, mật độ mực, điểm đầu/cuối, nét ngắn/chấm ở bên trái để hỗ trợ `uê/iê`. Không nối mực qua chỗ nhấc bút và không dùng thời gian giả lập làm đặc trưng.
- Chuyển trọng số tương thích từ `ghep`, khởi tạo trọng số nối nhánh hình học bằng 0 để bắt đầu với kết quả của mạng nguồn trước khi học thêm.
- SVM RBF dùng đặc trưng hình học. Fit và hiệu chỉnh CV chỉ trên train, nhóm theo người/style; mẫu augmentation của cùng nhóm không tách sang fold kiểm chứng.
- Thử alpha 0/0.15/0.3/0.5 trên validation. Chỉ bật SVM khi cải thiện accuracy trung bình giữa nguồn synthetic/human, không giảm accuracy từng nguồn hoặc số mẫu `uê/iê` đúng. Nếu không cải thiện thì alpha=0. Test không dùng chọn alpha hoặc checkpoint.
- AdamW, weight decay 0.0001, clipping gradient, bù lớp ít mẫu và augmentation không lật chữ. Learning rate mặc định 0.0001, giảm sau 12 epoch không cải thiện, dừng sớm sau 50; 400 là giới hạn tối đa. Đo thêm train sạch để quan sát overfitting, không so trực tiếp loss train có augmentation/trọng số với loss validation sạch.

Dữ liệu:

- Đọc InkML trong `dataset/raw/inkml_ghep` với nhãn `Tg_Truth`, `Dataset_Split`, `Writer_ID` hoặc `Style_ID`. Hỗ trợ namespace InkML. Split train/validation/test phải khác người/style; chương trình kiểm tra trùng người và trùng nét giữa các tập.
- Mặc định thêm VNOnDB ở mức từ, **chỉ khi nhãn trùng chính xác** với một lớp chữ ghép. Không tự cắt từ hoặc bỏ dấu để đoán nhãn. Giữ nguyên split chính thức của VNOnDB.
- Với dữ liệu hiện có: train 1.782 mẫu (1.036 synthetic + 746 human), validation 415 (148 + 267), test 630 (296 + 334). Phần human không phủ đủ 148 lớp.
- Riêng `uê/iê` hiện chỉ có 7 mẫu synthetic mỗi lớp trong train, 1 mỗi lớp trong validation, 2 mỗi lớp trong test. **Chưa có mẫu human trùng chính xác cho cặp này**; vẫn cần thu nét thật từ nhiều người/thiết bị để kiểm chứng việc hết nhầm.
- Nếu không có VNOnDB, thêm `--no-vnondb` khi tạo run. Khi thêm lớp khác bảng nhãn model cũ, dùng tên run mới và `--init-from=` để train mới.

Mỗi run lưu `data/` snapshot (pickle, label map, manifest/hash và data report), `label_map.json`, `preprocessing.json`, `train_config.json`, `ghep_best.keras`, `ghep_last.keras`, `shape_svm.joblib`, `hybrid_selection.json`, `history.csv`, `curves.png`, `state.json` và `progress.json`. Tiến độ là file phụ, có thể bỏ qua một lần ghi nếu bị khóa; checkpoint bắt buộc lưu thành công và có thử lại khi Windows khóa tạm thời.

Cache tự kiểm tra fingerprint nguồn và các file bắt buộc. Resume dùng snapshot của run để không âm thầm đổi dữ liệu/nhãn khi nguồn đã được bổ sung. Muốn học dữ liệu mới, tạo **run mới**.

Báo cáo ở `reports/ghep_stable/`: tách `deep`, `svm`, `hybrid`, có accuracy theo nguồn, style và family, các cặp nhầm; `predictions_*` ghi mọi mẫu, `errors_*` ghi mẫu sai và `ue_ie_*` ghi riêng cặp cần theo dõi. Báo cáo lưu hash model và danh sách mẫu. Không suy ra chất lượng chữ thật từ điểm synthetic.

```powershell
python -m src.ghep.evaluate_ghep --run-name ghep_stable --split test
python -m src.ghep.test_ghep --run-name ghep_stable --draw
python -m unittest discover -s tests -v
node tests/test_ghep_ui.cjs
```

## Chữ đơn: baseline_stable (Python 3.12.10)

File `test` chứa các lệnh cho chữ đơn. Chạy trong **Terminal PyCharm**, thư mục dự án, với `.venv` đã kích hoạt:

```powershell
python -m src.char.run_full --run-name baseline_stable --epochs 300 --init-from baseline
```

Đây là một lệnh chạy trực tiếp, hiện kết quả từng epoch. Pipeline tự lấy mẫu lại nét thô, train mạng sâu, fit SVM, chọn hệ số kết hợp trên validation rồi đánh giá validation/test. Đợi hoàn tất trước khi chạy dòng demo trong `test`:

```powershell
python -m src.serve --run-name baseline_stable
```

Mở `http://127.0.0.1:8000`. Có thể thử model sau khi checkpoint `best.keras` đầu tiên xuất hiện; giai đoạn này SVM chưa được fit. Demo tự nạp lại model tốt hơn và SVM tương ứng khi chúng có sẵn.

- **Mạng sâu:** CNN + hai BiGRU + nhánh đặc trưng hình dạng. Giữ 128 điểm/ký tự và tối thiểu 4 điểm/nét; không xem đoạn nhấc bút như nét mực. Nhánh hình dạng bổ sung chiều cao, vị trí, điểm đầu/cuối và phân bố điểm phía trên để học các cặp như `a/d`.
- **Học máy:** SVM RBF dùng histogram hướng và mật độ nét theo ô, kèm đặc trưng kích thước/thân chữ. SVM chỉ dự đoán chữ gốc; dấu phụ và dấu thanh vẫn từ mạng sâu. Hiệu chỉnh điểm SVM bằng CV trong tập train.
- **Kết hợp:** thử alpha `0, 0.15, 0.3, 0.5` trên validation. Chỉ bật SVM khi tăng số ký tự đúng toàn bộ và không làm giảm số mẫu `a/d` đúng. Nếu không cải thiện, alpha bằng 0 và demo dùng mạng sâu.
- **Chống quá khớp:** dropout 0.35, AdamW weight decay 0.0001, augmentation không lật chữ, trọng số bù lớp hiếm và gradient clipping. Mỗi epoch đo accuracy trên train sạch và validation. Loss train có bù lớp và augmentation, nên cần nhìn thêm accuracy sạch khi so sánh với validation.
- **Train dài có kiểm soát:** tối đa 300 epoch, giảm learning rate sau 10 epoch không cải thiện, dừng sớm sau 40. Chọn checkpoint bằng full-character accuracy trên validation, dùng validation loss để phân xử khi accuracy bằng nhau.

Theo dõi bằng Terminal khác:

```powershell
python -m src.char.status --run-name baseline_stable
```

Nếu bị dừng, dùng lệnh tiếp tục thay cho lệnh tạo run mới:

```powershell
python -m src.char.run_full --run-name baseline_stable --resume --epochs 300
```

`last.keras` chứa cả optimizer và trạng thái epoch; `best.keras` giữ model validation tốt nhất. Muốn dừng chủ động, tạo `models/baseline_stable/STOP`; `--resume` xóa tệp này. Epoch đang chạy dở sẽ được chạy lại. Thêm `--background` để chạy nền và đọc `models/baseline_stable/train.log`.

Nếu gặp `PermissionError: [WinError 5]` khi thay `progress.json`, Windows đang từ chối quyền thay tệp tại thời điểm cập nhật. Pipeline thử lại khi gặp lỗi truy cập tạm thời; nếu file tiến độ vẫn bị giữ, nó bỏ qua lần cập nhật đó và train tiếp. Các checkpoint vẫn phải ghi thành công, không bỏ qua lỗi lưu model. Khi một lượt đã dừng do lỗi này, dùng lệnh `--resume` ở trên để tiếp tục từ epoch hoàn thành gần nhất, giữ optimizer và learning rate.

Model mới ở `models/baseline_stable/`, gồm nét đã lấy mẫu lại (`data.npz`), metadata, label maps, cấu hình, lịch sử, biểu đồ, checkpoint và `shape_svm.joblib`/`hybrid_selection.json`. Báo cáo ở `reports/baseline_stable/`, tách `deep`, `svm` (SVM cho base + dấu từ deep), `hybrid`; có `a_d_*` liệt kê từng mẫu `a/d` và `confusions_*` chỉ ra các cặp nhầm. Model `baseline` cũ được giữ nguyên. `baseline_smoke` và `baseline_verify` là các lượt kiểm tra kỹ thuật, không phải model đã train đủ để triển khai.

**Giới hạn dữ liệu:** hiện 2.959 mẫu chữ đơn đều từ một người viết, gồm khoảng 15 mẫu `a` và 15 mẫu `d`. Tập train/validation/test giữ split cũ để so sánh; cùng người xuất hiện ở cả ba tập. Số đo không thể đại diện cho chữ của các bé mới. Cần thu thêm các kiểu viết `a/d` trên nhiều người và tách theo người khi có đủ dữ liệu. Không gán nhãn các nét bị cắt ngắn của `d` thành `a`, vì điều đó làm sai dữ liệu.

Sau khi bổ sung dữ liệu vào `dataset/raw/inkml`, cần chạy lại `python -m src.preprocess` để cập nhật danh sách mẫu, rồi tạo run với **tên mới**. Không preprocess lại dữ liệu đang dùng cho lượt train hiện tại.

Kiểm tra:

```powershell
python -m unittest discover -s tests -v
node tests/test_char_ui.cjs
python -m src.char.evaluate --run-name baseline_stable --split test
```

## Train bản ổn định

Chạy trong thư mục dự án, dùng Python của môi trường `.venv`:

```powershell
.\.venv\Scripts\python.exe -m src.seq.run_full --run-name word_stable --data-name word_v2 --init-from thu --lr 0.0005 --epochs 160 --background
```

Pipeline tự tiền xử lý nếu chưa có `word_v2`, train rồi đánh giá test bằng model tốt nhất. Nó cũng so sánh greedy / beam / gợi ý ngôn ngữ trên 1.000 mẫu validation. Không dùng test để chọn checkpoint hoặc điều chỉnh mô hình ngôn ngữ.

- Train: 66.991 mẫu; validation: 18.639; test: 25.114. Các người viết và trang không trùng giữa ba tập.
- Tối đa 160 epoch; dừng sớm sau 20 epoch không cải thiện CER đủ mức `0.0001`; giảm learning rate sau 5 epoch không cải thiện.
- AdamW với weight decay `0.0001`, dropout `0.3`, augmentation hình học và clipping gradient. Các cơ chế này hỗ trợ kiểm soát overfitting; phải đọc đường train/validation để đánh giá thực tế.
- CER train đo trên 1.000 mẫu cố định, tắt augmentation; CER validation đo toàn bộ tập validation.
- Lưu model, optimizer, learning rate, epoch và danh sách mẫu để khôi phục. `word_best.keras` giữ checkpoint seq có validation CER thấp nhất. Các run seq cũ còn tên `best.keras` vẫn đọc được.
- `--init-from thu` chuyển trọng số model cũ sang kiến trúc đã sửa padding. Model cũ vẫn nằm nguyên trong `models/seq/thu`.

Theo dõi tiến độ, không cần nạp TensorFlow:

```powershell
.\.venv\Scripts\python.exe -m src.seq.status_seq --run-name word_stable
Get-Content models\seq\word_stable\train.log -Tail 20 -Wait
```

Dừng có kiểm soát bằng cách tạo tệp `STOP`. Công việc đang chạy sẽ dừng; checkpoint của epoch hoàn thành gần nhất dùng để tiếp tục:

```powershell
New-Item models\seq\word_stable\STOP -ItemType File -Force
.\.venv\Scripts\python.exe -m src.seq.run_full --run-name word_stable --resume --epochs 160 --background
```

Đợi tiến trình cũ dừng trước khi chạy lại. Khi resume, tệp `STOP` được xóa và cấu hình train đã lưu được sử dụng. Nếu chưa hoàn thành epoch đầu tiên thì chưa có checkpoint để resume.

Môi trường Windows hiện tại dùng TensorFlow 2.21 trên CPU, dù máy có GPU NVIDIA. `train.log` ghi thiết bị thực sự được sử dụng. `--threads 6` giới hạn số luồng tính toán chính.

## Thử model

Model chính của seq là `models/seq/word_stable/word_best.keras`. Sau khi hoàn thành ít nhất một epoch và có tệp này:

```powershell
.\.venv\Scripts\python.exe -m src.seq.serve_seq --run-name word_stable
```

Mở `http://127.0.0.1:8001`. Khi train tạo checkpoint tốt hơn, demo tự nạp checkpoint mới ở lần nhận dạng tiếp theo.

Kết quả đọc từ nét bút được giữ riêng với gợi ý tiếng Việt. Bài chính tả chỉ dùng so sánh sau nhận dạng. Gợi ý không được tính vào điểm của bé. Mức hỗ trợ từng ký tự được tính từ căn chỉnh CTC, không phải xác suất đã hiệu chỉnh hoặc điểm đẹp chữ.

Phần học sâu là CNN + BiGRU + CTC. Phần học máy thống kê là mô hình ký tự n-gram và từ vựng chỉ học từ nhãn **train**, dùng để đề xuất từ gần giống. Không có quy tắc ép `Phưng` thành `phương`; hiệu quả của phần gợi ý được báo cáo riêng vì nó có thể sửa sai thành một từ khác.

## Những thay đổi về độ ổn định

- Nét mũ/móc ngắn giữ ít nhất 4 điểm thay vì 2; lấy mẫu dày hơn (`step=0.04`, tối đa 400 điểm/từ).
- Mask sau chuẩn hóa và sau mỗi CNN; padding cố định trước convolution stride 2. Các frame thật có cùng kết quả khi chạy riêng hoặc ghép batch.
- Beam search chuyển đúng blank ở chỉ số 0; các ứng viên được tính điểm trên cùng xác suất chuỗi CTC.
- Thêm nét dấu hoặc hoàn tác sẽ hủy kết quả cũ đang chờ; nhấn Space hai lần không thêm trùng từ.
- Tiền xử lý được lưu cùng run; dữ liệu `word_v2` tách riêng khỏi `word` cũ. Không ghi đè bộ dữ liệu đã tồn tại.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node tests\test_seq_ui.cjs
.\.venv\Scripts\python.exe -m src.seq.evaluate_seq --run-name word_stable --split validation --max-samples 1000 --decoder all
.\.venv\Scripts\python.exe -m src.seq.evaluate_seq --run-name word_stable --split validation --words phương --decoder all
```

Báo cáo ở `reports/seq/<run-name>/`. `vowels_*` đo riêng độ chính xác hình dạng `a/ă/â`, `o/ô/ơ`, `e/ê`; `confusions_*` liệt kê nhầm, thiếu, thừa ký tự. Các metric lưu hash model và tập mẫu để so sánh cùng điều kiện. Validation mặc định sử dụng danh sách mẫu đã lưu của run; `--full-split` dùng toàn bộ tập chính thức.

## Dữ liệu để dạy bé

Dữ liệu VNOnDB hiện tại chưa được kiểm chứng trên nhóm bé và thiết bị sẽ triển khai. Cần thu thêm nét bút của nhiều bé, trên chuột/bút/màn hình sẽ dùng thực tế, với nhãn được kiểm tra. Giữ người viết ngoài tập train để kiểm chứng. Tăng epoch không tự khắc phục sự khác biệt này.

Nhận đúng nội dung là một phần của ứng dụng dạy viết. Đánh giá hướng/thứ tự nét, chiều cao, vị trí dấu và phản hồi từng nét cần bộ mẫu chuẩn cùng nhãn đánh giá tương ứng.
# Train-model-Handwriting

# Historical-Stock-Price-Prediction-Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nolanminhdinh/Historical-Stock-Price-Prediction-Model/blob/main/SOTA.ipynb)

## Tổng quan dự án

Dự án này tập trung xây dựng và triển khai một mô hình **Deep Learning tiên tiến (State-of-the-Art)** để dự đoán xu hướng giá cổ phiếu của các doanh nghiệp lớn tại Việt Nam (ví dụ trong dự án **HPG**, **MBB**).

Khác với các phương pháp thống kê truyền thống, dự án sử dụng kiến trúc mạng lai (Hybrid Architecture) phức tạp nhằm tối ưu hóa độ chính xác:
* **CNN (Convolutional Neural Networks):** Trích xuất các đặc trưng ngắn hạn và xu hướng biến động từ dữ liệu thô.
* **Bi-LSTM (Bidirectional LSTM):** Học sự phụ thuộc dài hạn của chuỗi thời gian theo cả hai chiều (quá khứ và tương lai).
* **Attention Mechanism (Cơ chế Chú ý):** Tự động đánh trọng số cho các mốc thời gian quan trọng, giúp mô hình tập trung vào các biến động giá có ý nghĩa nhất.

--

## 🧠 Kiến trúc Mô hình: Mô phỏng Tư duy Nhà đầu tư (Model Architecture)

Điểm độc đáo của dự án là việc thiết kế kiến trúc mạng lai (Hybrid Architecture) nhằm **mô phỏng lại quá trình ra quyết định của một nhà đầu tư chuyên nghiệp**. 

Mỗi lớp trong mô hình đóng vai trò như một bước trong tư duy phân tích:

### 1. Quan sát & Nhận diện (Feature Extraction - CNN)
> *"Nhà đầu tư nhìn vào biểu đồ nến để nắm bắt các mẫu hình giá ngắn hạn."*

* **Lớp CNN (Convolutional Neural Network):** Đóng vai trò như "đôi mắt", trích xuất các đặc trưng quan trọng từ dữ liệu thô (giá đóng cửa, khối lượng).
* **Tác dụng:** Loại bỏ nhiễu (noise) của thị trường hàng ngày và nhận diện các mẫu hình biến động cục bộ (local patterns) như xu hướng tăng/giảm đột ngột.

### 2. Phân tích Xu hướng Chuỗi (Trend Analysis - Bi-LSTM)
> *"Nhà đầu tư xâu chuỗi dữ liệu quá khứ và hiện tại để hiểu bối cảnh thị trường."*

* **Lớp Bi-LSTM (Bidirectional LSTM):** Đóng vai trò như "bộ nhớ", học sự phụ thuộc của chuỗi thời gian theo cả hai chiều: từ Quá khứ -> Hiện tại và từ Tương lai (trong ngữ cảnh training) -> Quá khứ.
* **Tác dụng:** Giúp mô hình không chỉ nhìn thấy giá ngày hôm nay mà còn hiểu được đà tăng trưởng (momentum) tích lũy từ chuỗi ngày trước đó.

### 3. Tập trung vào Điểm đột biến (Attention Mechanism)
> *"Nhà đầu tư bỏ qua những ngày thị trường đi ngang (sideway) và dồn sự chú ý vào các phiên có biến động mạnh để ra quyết định."*

* **Cơ chế Attention:** Đóng vai trò như "trực giác", tự động gán trọng số cao hơn cho các mốc thời gian có ảnh hưởng lớn đến giá tương lai (ví dụ: các phiên có khối lượng giao dịch đột biến).
* **Tác dụng:** Giúp mô hình tập trung vào "tín hiệu" (signals) thay vì bị phân tâm bởi các dữ liệu ít quan trọng, từ đó tối ưu hóa độ chính xác dự báo.
---

## Điểm nổi bật (Key Features)

* **Quy trình dữ liệu tự động (Automated Pipeline):** Tự động thu thập dữ liệu lịch sử và realtime thông qua thư viện `vnstock`.
* **Xử lý dữ liệu nâng cao:** Chuẩn hóa dữ liệu sử dụng kỹ thuật Robust Scaling (dựa trên phân vị IQR) thay vì MinMax thông thường điều này giúp loại bỏ ảnh hưởng của các phiên giao dịch đột biến (Outliers), giúp mô hình học được xu hướng thực chất của thị trường và xử lý chuỗi thời gian bằng kỹ thuật Sliding Window.
* **Mô hình SOTA:** Kết hợp `Conv1D` + `Bi-LSTM` + `Attention Layer` để giảm thiểu sai số dự báo.
* **Đánh giá toàn diện:** Sử dụng các chỉ số RMSE, MAE, MAPE và R2-Score để kiểm chứng hiệu quả.

---

##  Công nghệ & Cấu trúc

**Programming language**: Python
**Libraries & Frameworks**: TensorFlow (Keras), Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Vnstock.
**Tools & Platforms**: Google Colab, Jupyter Notebook, Git/GitHub.

Cấu trúc thư mục dự án:

```text
├── 📂 data/               # Chứa dữ liệu thô và dữ liệu đã xử lý (HPG, MBB)
├── 📂 models/             # Chứa file Scaler (.pkl) và cấu hình Model
├── 📂 notebooks/          # Mã nguồn chính (Jupyter Notebooks)
│   ├── Collection_Data.ipynb  # Code thu thập dữ liệu tự động
│   └── SOTA.ipynb             # Code huấn luyện và đánh giá mô hình
├── 📂 images/             # Ảnh biểu đồ kết quả (dùng cho báo cáo)
├── requirements.txt       # Danh sách các thư viện cần cài đặt
└── README.md              # Thông tin mô tả, tài liệu hướng dẫn

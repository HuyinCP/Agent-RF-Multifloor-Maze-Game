# Multi-Floor Maze: Reinforcement Learning trong mê cung nhiều tầng

Agent phải đi từ điểm bắt đầu đến đích trong một mê cung được sinh ngẫu nhiên. Đường đi có thể qua nhiều tầng, cửa cần chìa khóa, bẫy và kẻ địch. Mỗi lượt, agent nhìn thấy vùng lân cận và chọn một hành động; mục tiêu là **đến đích trước khi hết máu hoặc hết số bước**.

Đây là bài toán học tăng cường với các episode hữu hạn. Mỗi episode tạo một map mới; agent tương tác với môi trường qua API Gymnasium `reset()` và `step(action)`. Curriculum gồm 9 cấp độ, từ map một tầng không có địch đến map ba tầng có khóa, bẫy và nhiều loại địch.

## Môi trường và agent

Môi trường chính là [`MazeEnv`](multi_floor_maze/mfm/env.py). Map dạng lưới được tạo bằng BSP, nối các phòng bằng hành lang, đặt cầu thang giữa các tầng và kiểm tra đường đi. Agent có máu, đạn, thể lực và số chìa khóa đang giữ. Các ô có thể chứa tường, bẫy, cửa, chìa khóa, gói hồi máu/đạn, cầu thang hoặc đích. Kẻ địch gồm patrol, chaser và sniper.

Ở mỗi bước, môi trường nhận hành động, cập nhật vị trí và tương tác, di chuyển kẻ địch/đạn, rồi trả về `(observation, reward, terminated, truncated, info)`. Episode kết thúc khi agent tới đích hoặc chết (`terminated`), hoặc khi đạt giới hạn số bước (`truncated`). Cấu hình 9 level nằm trong `CURRICULUM` ở cùng file; có thể truyền `config` riêng để tạo map tùy chỉnh.

Ví dụ khởi tạo một episode (chạy từ thư mục `multi_floor_maze`):

```python
from mfm.env import CURRICULUM, MazeEnv

env = MazeEnv(config=CURRICULUM[1])
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(3)  # đi sang phải
```

### Observation space

Actor nhận một `Dict` gồm hai phần:

| Thành phần | Kích thước mặc định | Nội dung |
|---|---:|---|
| `visual` | `16 × 9 × 9` | Vùng 9 × 9 quanh agent, mã hóa theo kênh: tường, bẫy, cửa, cầu thang, đích, vật phẩm, agent, ba loại địch, đạn, tia laser và ô đã thăm. |
| `vector` | `15` | Trạng thái đã chuẩn hóa: máu, đạn, thể lực, chìa khóa, tiếng ồn, tầng hiện tại, hướng tới mục tiêu trung gian, hướng tầng đích, hướng di chuyển gần nhất, thời gian còn lại, mức khám phá, trạng thái đứng trên cầu thang và khoảng cách tới địch gần nhất. |

`visual` chỉ chứa vùng cục bộ; `vector` cung cấp thêm tín hiệu dẫn hướng. Bán kính quan sát mặc định là 4 ô và có thể đổi bằng `view_radius`.

### Action space

Actor chọn **một trong 10 hành động rời rạc** (`Discrete(10)`):

| ID | Hành động |
|---:|---|
| `0–3` | Đi lên, xuống, trái, phải. |
| `4–7` | Bắn lên, xuống, trái, phải; tốn 1 viên đạn. |
| `8` | Lướt tối đa 2 ô theo hướng gần nhất; tốn thể lực. |
| `9` | Dùng cầu thang khi đang đứng trên ô cầu thang. |

Di chuyển vào cửa sẽ tự mở nếu agent có chìa khóa. Bắn và lướt tạo tiếng ồn, có thể khiến địch chú ý.

### Reward

Reward mặc định khuyến khích hoàn thành nhiệm vụ và khám phá: tới đích `+25`, nhặt chìa `+2.5`, mở cửa `+3.5`, thăm ô mới `+0.05`, tiến gần mục tiêu trung gian tối đa `+0.10` mỗi bước. Môi trường phạt mỗi bước `−0.02`, va tường/hành động không hợp lệ `−0.12`, trúng đòn `−2`, dẫm bẫy `−2.5` và chết `−18`. Ngoài ra còn có reward cho hạ địch, dùng cầu thang lần đầu và nhặt vật phẩm. Có thể điều chỉnh các giá trị bằng `config["rewards"]`.

## Thuật toán và cách train

Dự án dùng **PPO** và **A2C** từ Stable-Baselines3 với `MultiInputPolicy`: CNN xử lý `visual`, MLP xử lý `vector`, sau đó ghép đặc trưng để actor chọn hành động và critic ước lượng giá trị. Training dùng 4 môi trường `DummyVecEnv`, chuẩn hóa reward bằng `VecNormalize`, và lần lượt train qua 9 level trong `CURRICULUM`. Sau mỗi level, script đánh giá tỉ lệ thắng; nếu dưới ngưỡng, train thêm một đợt rồi chuyển sang level tiếp theo.

Các script chính:

| Script | Mục đích |
|---|---|
| [`train.py`](multi_floor_maze/train.py) | Train PPO cơ bản trên CPU. |
| [`train_a2c.py`](multi_floor_maze/train_a2c.py) | Train A2C cơ bản trên CPU. |
| [`train_experiments.py`](multi_floor_maze/train_experiments.py) | So sánh 5 cấu hình: PPO, A2C, PPO với `γ=0.95`, mạng nhỏ hơn, và PPO với entropy coefficient `0.02`. Mặc định đặt device là `cuda`. |
| [`evaluate_experiments.py`](multi_floor_maze/evaluate_experiments.py) | Đánh giá các model và random baseline trên nhiều map; xuất JSON vào `outputs/evals/`. |

## Cài đặt và chạy

Chạy các lệnh từ thư mục `multi_floor_maze` (khuyến nghị Python 3.10+):

```bash
cd multi_floor_maze
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python train.py                                  # PPO cơ bản
python train_a2c.py                              # A2C cơ bản
python train_experiments.py --only exp2_a2c      # một thí nghiệm
python evaluate_experiments.py --n-maps 100       # đánh giá model đã train
```

Checkpoint và kết quả được lưu trong `multi_floor_maze/outputs/` (thư mục này không được đưa lên Git). Training đầy đủ qua 9 level tốn nhiều thời gian; `train_experiments.py` cần PyTorch có CUDA nếu giữ nguyên cấu hình device.

Để xem agent chơi trên map tùy chỉnh, cài thêm `Pillow` cho chức năng GIF rồi chạy:

```bash
pip install pillow
streamlit run app_custom_map.py
```

Ứng dụng Streamlit cho phép chọn kích thước, số tầng, bẫy, địch, khóa và model đã train. Ngoài ra có [`play_custom_map.py`](multi_floor_maze/play_custom_map.py) cho CLI, [`generate_gifs.py`](multi_floor_maze/generate_gifs.py) để xuất demo, và [`render_maps.py`](multi_floor_maze/render_maps.py) để vẽ map. Các notebook trong `multi_floor_maze/` phục vụ nghiên cứu sinh map và curriculum.

# Multi-Floor Maze — bài toán Reinforcement Learning

Agent cần đi từ điểm bắt đầu đến đích trong mê cung nhiều tầng được sinh ngẫu nhiên. Trên đường có cầu thang, bẫy, cửa cần chìa khóa, vật phẩm và kẻ địch. Agent chỉ quan sát vùng gần mình, nên phải vừa tìm đường vừa quản lý máu, đạn và thể lực. Mục tiêu của mỗi episode là **đến đích trước khi chết hoặc hết số bước**.

## Môi trường game

[`MazeEnv`](multi_floor_maze/mfm/env.py) dùng API Gymnasium. `reset(seed=...)` tạo map và trả về quan sát ban đầu; `step(action)` thực hiện một lượt rồi trả về `(observation, reward, terminated, truncated, info)`. `terminated=True` khi thắng hoặc chết; `truncated=True` khi hết số bước. Map được tạo bằng BSP với các phòng nối bằng hành lang. Cấu hình game nằm trong `CURRICULUM` gồm 9 level, từ map 7×7 một tầng đến map 13×13 ba tầng.

Ví dụ chạy từ thư mục `multi_floor_maze`:

```python
from mfm.env import CURRICULUM, MazeEnv

env = MazeEnv(config=CURRICULUM[1])
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(3)  # đi sang phải
```

### Observation space (đầu vào của actor)

Actor nhận `Dict` gồm hai phần. Đây là **quan sát**, không phải toàn bộ trạng thái của map:

| Thành phần | Kích thước mặc định | Ý nghĩa |
|---|---:|---|
| `visual` | `16 × 9 × 9` | Vùng 9×9 quanh agent. Các kênh đánh dấu tường, bẫy, cửa, cầu thang, đích, vật phẩm, agent, địch, đạn, laser và ô đã thăm. |
| `vector` | `15` | Máu, đạn, thể lực, chìa khóa, tiếng ồn, tầng hiện tại, hướng tới mục tiêu, hướng đi gần nhất, thời gian còn lại, mức khám phá, cầu thang và độ gần của địch. |

### Action space (đầu ra của actor)

Actor chọn một số nguyên trong `Discrete(10)`:

| Action | Tác dụng |
|---:|---|
| `0, 1, 2, 3` | Đi lên, xuống, trái, phải. |
| `4, 5, 6, 7` | Bắn lên, xuống, trái, phải; tốn 1 đạn. |
| `8` | Lướt tối đa 2 ô theo hướng gần nhất; tốn thể lực. |
| `9` | Dùng cầu thang khi đang đứng trên ô cầu thang. |

Đi vào cửa sẽ mở cửa nếu agent có chìa khóa. Bắn và lướt tạo tiếng ồn, có thể thu hút địch.

### Reward và kết thúc episode

Reward mặc định: tới đích `+25`, nhặt chìa `+2.5`, mở cửa `+3.5`, khám phá ô mới `+0.05`, tiến gần mục tiêu tối đa `+0.10` mỗi bước. Môi trường phạt mỗi bước `−0.02`, va tường `−0.12`, trúng đòn `−2`, dẫm bẫy `−2.5` và chết `−18`. Ngoài ra còn có reward cho hạ địch, nhặt vật phẩm và sang tầng mới. Các giá trị có thể đổi bằng `config["rewards"]`.

## Thuật toán và training

Dự án dùng **PPO** và **A2C** của Stable-Baselines3. `MultiInputPolicy` dùng CNN đọc `visual` và MLP đọc `vector`, ghép hai nhóm đặc trưng để actor chọn action và critic ước lượng giá trị. Training chạy 4 game environment bằng `DummyVecEnv`, chuẩn hóa reward với `VecNormalize`, rồi học lần lượt trên 9 level. Sau mỗi level, script đánh giá tỉ lệ thắng; nếu dưới ngưỡng, nó train thêm một đợt trước khi chuyển level.

| Script | Công dụng |
|---|---|
| [`train.py`](multi_floor_maze/train.py) | Train PPO trên CPU. |
| [`train_a2c.py`](multi_floor_maze/train_a2c.py) | Train A2C trên CPU. |
| [`train_experiments.py`](multi_floor_maze/train_experiments.py) | Chạy 5 cấu hình PPO/A2C để so sánh; mặc định dùng CUDA. |
| [`evaluate_experiments.py`](multi_floor_maze/evaluate_experiments.py) | Đánh giá model và random baseline trên các map mới; xuất JSON. |

## Cài đặt và chạy

Khuyến nghị Python 3.10+. Từ thư mục gốc của repo:

```powershell
cd multi_floor_maze
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
python train.py                 # train PPO; checkpoint lưu trong outputs/
python evaluate_experiments.py --n-maps 100
```

Trên macOS/Linux, dùng `source .venv/bin/activate` thay cho lệnh kích hoạt Windows. Có thể chạy `python train_a2c.py` hoặc `python train_experiments.py --only exp2_a2c` để train biến thể khác. Script experiments đặt `DEVICE="cuda"`; cần GPU/CUDA hoặc đổi sang `cpu` trong script.

Để xem agent chơi bằng Streamlit, cần có checkpoint đã train và cài thêm Pillow (chưa có trong `requirements.txt`):

```powershell
pip install pillow
streamlit run app_custom_map.py
```

Model, kết quả đánh giá và GIF được lưu trong `multi_floor_maze/outputs/` (không đưa lên Git). Có thể dùng [`play_custom_map.py`](multi_floor_maze/play_custom_map.py) để chạy map tùy chỉnh từ CLI.

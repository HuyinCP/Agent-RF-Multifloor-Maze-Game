# Cấu trúc Zeppelin — Các thí nghiệm & Đánh giá

## Tổng quan

- **Train:** Chạy `python train_experiments.py` (hoặc `--only exp1_ppo` …) → lưu `outputs/exp1_ppo_final.zip` … `outputs/exp5_ppo_ent02_final.zip`.
- **Eval:** Chạy `python evaluate_experiments.py` → lưu `outputs/evals/exp1_ppo.json` … `outputs/evals/random.json`.
- **Zeppelin:** Mỗi thí nghiệm một section; cuối có section so sánh tất cả.

---

## 5 thí nghiệm

| ID | Tên | Mô tả |
|----|-----|-------|
| exp1_ppo | Thí nghiệm 1: PPO (mặc định) | PPO gamma=0.99, ent_coef=0.01, net 128→64 |
| exp2_a2c | Thí nghiệm 2: A2C | Cùng curriculum, A2C (nhanh hơn PPO mỗi step) |
| exp3_ppo_gamma095 | Thí nghiệm 3: PPO gamma=0.95 | Discount ngắn hơn (refining discounted return) |
| exp4_ppo_smallnet | Thí nghiệm 4: PPO mạng nhỏ | features_dim=64, net [64,32] (improving inner model) |
| exp5_ppo_ent02 | Thí nghiệm 5: PPO ent_coef=0.02 | Khám phá nhiều hơn (change policy / hyperparams) |

+ **Random:** baseline không train, dùng `outputs/evals/random.json`.

---

## Cấu trúc từng section trong Zeppelin

### Cho từng thí nghiệm (1..5)

1. **Markdown:** Tiêu đề "Thí nghiệm K: …" và mô tả ngắn (mô hình, siêu tham số).
2. **Code (tùy chọn):** Load model từ `outputs/expK_*.zip` nếu cần chạy demo.
3. **Code đọc kết quả eval:**
   - Đọc `outputs/evals/expK_*.json` (hoặc đường dẫn tương đối notebook).
   - Biến: `results` = list theo level, mỗi phần tử có `mean_reward`, `std_reward`, `success_rate`, `returns`, `terminations`, `regret`, `mean_steps`.
4. **Biểu đồ bắt buộc (Evaluations/Comparisons):**
   - **Expected return:** Bar chart theo level (trục ngang: level, trục dọc: mean_reward), có thể thêm error bar = std_reward.
   - **Return line graph:** Trục ngang = index episode (1..100 mỗi level), trục dọc = return; có thể vẽ 9 đường (mỗi level) hoặc gộp tất cả returns theo level rồi vẽ line theo thứ tự.
   - **Histogram / distribution:** Histogram của tất cả `returns` (gộp 9 level) hoặc từng level; tiêu đề "Phân phối reward".
   - **Regret line graph:** Trục ngang = level (hoặc episode index), trục dọc = regret (trong JSON đã có `regret` theo level); hoặc vẽ regret = max_theoretical - return từng episode nếu có.
   - **No of terminations:** Stacked bar theo level: 3 phần win / timeout / death (từ `terminations` trong mỗi level).

### Section cuối: So sánh tất cả thí nghiệm

1. **Markdown:** "So sánh tất cả mô hình".
2. **Code:** Load tất cả `outputs/evals/exp1_ppo.json` … `random.json`.
3. **Bảng tổng hợp:**
   - Cột: Model (exp1_ppo, exp2_a2c, … random).
   - Hàng: overall_mean_reward, overall_success_rate; hoặc từng level mean_reward / success_rate.
4. **Biểu đồ so sánh:**
   - Bar chart: Win rate theo level, mỗi nhóm = 1 level, mỗi bar = 1 model (exp1, exp2, … random).
   - Bar chart: Mean reward theo level, tương tự.
   - Có thể thêm: histogram so sánh distribution reward giữa các model (overlay hoặc cạnh nhau).

---

## Định dạng JSON eval (`outputs/evals/<exp_id>.json`)

```json
{
  "experiment_id": "exp1_ppo",
  "label": "exp1_ppo",
  "n_maps_per_level": 100,
  "seed_base": 20000,
  "results": [
    {
      "level": 1,
      "name": "Crawl",
      "n_maps": 100,
      "mean_reward": 26.13,
      "std_reward": 0.64,
      "success_rate": 1.0,
      "mean_steps": 45.2,
      "returns": [26.1, 25.9, ...],
      "terminations": { "win": 100, "timeout": 0, "death": 0 },
      "steps": [4, 5, ...],
      "regret": 0.0,
      "mean_max_theoretical": 28.5
    }
  ],
  "overall_mean_reward": 15.2,
  "overall_success_rate": 0.65
}
```

- **Expected return** = `mean_reward` theo level (bar chart).
- **Return line graph** = vẽ `returns` (flatten theo level hoặc từng level một đường).
- **Histogram** = histogram của mảng `returns` (có thể gộp toàn bộ level).
- **Regret line** = `regret` theo level (hoặc từng episode: max_theoretical - return nếu có).
- **No of terminations** = stacked bar từ `terminations.win`, `terminations.timeout`, `terminations.death` theo level.

---

## Thứ tự đề xuất trong notebook Zeppelin

1. Phần hiện có (tổng quan, demo, lý thuyết, sinh map, …).
2. **Thí nghiệm 1: PPO** — mô tả + load eval JSON + 5 loại biểu đồ.
3. **Thí nghiệm 2: A2C** — tương tự.
4. **Thí nghiệm 3: PPO gamma=0.95** — tương tự.
5. **Thí nghiệm 4: PPO mạng nhỏ** — tương tự.
6. **Thí nghiệm 5: PPO ent_coef=0.02** — tương tự.
7. **So sánh tất cả** — bảng + bar win rate/reward theo level theo model + (tùy chọn) histogram so sánh.

Sau khi train xong 5 mô hình và chạy `evaluate_experiments.py`, chỉ cần dán/điều chỉnh code đọc JSON và vẽ biểu đồ vào đúng từng section như trên là đủ đánh giá đầy đủ (Expected return, Return line, Histogram, Regret, No of terminations).

"""
Chơi trên map tự custom: người dùng set kích thước, bẫy, quái, khóa;
chọn bộ não (model) để agent chơi, có thể xuất GIF.

Usage:
  # Map 11x11, 2 tầng, 2 bẫy, 1 patrol floor 0, dùng exp2_a2c, xuất GIF
  python play_custom_map.py --width 11 --height 11 --n-floors 2 --traps 2 --enemies "patrol:0" --model exp2_a2c --output demos/custom.gif

  # Map khó: 13x13, 3 tầng, 3 bẫy, 2 khóa, sniper+chaser+patrol, dùng exp3 PPO
  python play_custom_map.py --width 13 --height 13 --n-floors 3 --traps 3 --locks 2 \\
    --enemies "sniper:0,chaser:1,patrol:2" --model exp3_ppo_gamma095 --seed 99

  # Chỉ chạy, in kết quả, không tạo GIF
  python play_custom_map.py --width 9 --height 9 --model exp2_a2c --no-gif
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mfm.env import MazeEnv

# Render & load model từ generate_gifs
from generate_gifs import (
    GIF_DURATION,
    load_model as _load_model_from_path,
    run_episode,
    save_gif,
)

OUTPUTS = ROOT / "outputs"
DEMOS_DIR = OUTPUTS / "demos"

# Tên model → (path, type). path có thể là thư mục exp hoặc file .zip
MODEL_ALIASES = {
    "exp1_ppo": ("exp1_ppo", "PPO"),
    "exp2_a2c": ("exp2_a2c", "A2C"),
    "exp3_ppo_gamma095": ("exp3_ppo_gamma096/exp3_ppo_gamma095_final.zip", "PPO"),
    "exp4_ppo_smallnet": ("exp4_ppo_smallnet", "PPO"),
    "exp5_ppo_ent02": ("exp5_ppo_ent02", "PPO"),
}


def resolve_model(model_arg: str):
    """Trả về (path: Path, model_type: str). model_arg = tên exp hoặc đường dẫn .zip."""
    p = Path(model_arg)
    if p.suffix == ".zip":
        path = (ROOT / model_arg).resolve() if not p.is_absolute() else p
        if not path.exists():
            path = OUTPUTS / p.name
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {model_arg}")
        model_type = "A2C" if "a2c" in path.name.lower() else "PPO"
        return path, model_type

    name = model_arg.strip()
    if name in MODEL_ALIASES:
        rel, model_type = MODEL_ALIASES[name]
        if rel.endswith(".zip"):
            path = OUTPUTS / rel
        else:
            path = OUTPUTS / rel / f"{rel}_final.zip"
            if not path.exists():
                path = OUTPUTS / f"{rel}_final.zip"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy model {name}: {path}")
        return path, model_type

    path = OUTPUTS / name / f"{name}_final.zip"
    if not path.exists():
        path = OUTPUTS / f"{name}_final.zip"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {name}")
    model_type = "A2C" if "a2c" in name else "PPO"
    return path, model_type


def parse_enemies(s: str, n_floors: int):
    """Parse chuỗi dạng 'patrol:0,chaser:1,sniper:0' -> list dict {type, floor}."""
    if not s or not s.strip():
        return []
    out = []
    for part in s.strip().split(","):
        part = part.strip()
        if ":" in part:
            typ, floor = part.split(":", 1)
            floor = int(floor)
        else:
            typ, floor = part, 0
        typ = typ.strip().lower()
        if typ not in ("patrol", "chaser", "sniper"):
            continue
        if 0 <= floor < n_floors:
            out.append({"type": typ, "floor": floor})
    return out


def build_custom_config(args):
    """Tạo config env từ args (tương thích CURRICULUM trong mfm.env)."""
    nf = max(1, args.n_floors)
    w = max(5, args.width)
    h = max(5, args.height)
    enemies = parse_enemies(args.enemies or "", nf)
    locks = []
    for i in range(args.locks):
        f = min(i, nf - 1)
        locks.append({"k": f, "d": f})

    cfg = {
        "name": args.name or "Custom",
        "width": w,
        "height": h,
        "n_floors": nf,
        "max_steps": args.max_steps or (w * h * 2 + nf * 50),
        "agent_hp": args.agent_hp,
        "agent_ammo": args.agent_ammo,
        "agent_max_ammo": max(args.agent_ammo, 5) if args.agent_ammo else 0,
        "agent_stamina": args.agent_stamina,
        "n_traps": args.traps,
        "n_health": args.health,
        "n_ammo": args.ammo,
        "enemies": enemies,
        "locks": locks,
        "start_floor": min(args.start_floor, nf - 1),
        "goal_floor": min(args.goal_floor, nf - 1),
    }
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="Chơi map custom với agent (chọn model), có thể xuất GIF."
    )
    # Map
    ap.add_argument("--width", type=int, default=11, help="Chiều rộng map (số ô)")
    ap.add_argument("--height", type=int, default=11, help="Chiều cao map (số ô)")
    ap.add_argument("--n-floors", type=int, default=1, help="Số tầng")
    ap.add_argument("--max-steps", type=int, default=None, help="Số bước tối đa (mặc định theo kích thước)")
    ap.add_argument("--name", type=str, default="Custom", help="Tên map (hiển thị)")
    # Vật cản / đồ
    ap.add_argument("--traps", type=int, default=0, help="Số bẫy")
    ap.add_argument("--health", type=int, default=3, help="Số health pack")
    ap.add_argument("--ammo", type=int, default=2, help="Số ammo pack")
    ap.add_argument("--locks", type=int, default=0, help="Số cặp cửa/chìa")
    ap.add_argument("--enemies", type=str, default=None,
                    help='Quái: "patrol:0,chaser:1,sniper:0" (type:tầng, cách nhau bằng dấu phẩy)')
    # Agent
    ap.add_argument("--agent-hp", type=int, default=10, help="HP agent")
    ap.add_argument("--agent-ammo", type=int, default=5, help="Ammo xuất phát")
    ap.add_argument("--agent-stamina", type=int, default=5, help="Stamina")
    ap.add_argument("--start-floor", type=int, default=None, help="Tầng bắt đầu (mặc định: tầng cao nhất)")
    ap.add_argument("--goal-floor", type=int, default=0, help="Tầng đích")
    # Model & chạy
    ap.add_argument("--model", type=str, required=True,
                    help="Bộ não: exp2_a2c, exp3_ppo_gamma095, hoặc đường dẫn .zip")
    ap.add_argument("--seed", type=int, default=42, help="Seed sinh map")
    ap.add_argument("--output", type=str, default=None, help="Đường dẫn lưu GIF (không đặt = không ghi file)")
    ap.add_argument("--no-gif", action="store_true", help="Chỉ chạy, in kết quả, không tạo GIF")
    ap.add_argument("--duration", type=int, default=GIF_DURATION, help="Ms mỗi frame GIF")
    args = ap.parse_args()

    if args.start_floor is None:
        args.start_floor = max(0, args.n_floors - 1)

    config = build_custom_config(args)
    print("Config map:", {k: v for k, v in config.items()})

    model_path, model_type = resolve_model(args.model)
    print(f"Load model: {model_path} ({model_type})")
    model = _load_model_from_path(model_type, model_path)
    label = model_path.stem.replace("_final", "")

    frames, info, total_r = run_episode(model, config, label, seed=args.seed)
    win = info.get("win", False)
    steps = info.get("steps", len(frames) - 1)
    print(f"  Kết quả: {'WIN' if win else 'LOSE'} | steps={steps} | reward={total_r:+.2f} | frames={len(frames)}")

    if not args.no_gif:
        out_path = args.output
        if not out_path:
            tag = "WIN" if win else "LOSE"
            DEMOS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = DEMOS_DIR / f"custom_{label}_seed{args.seed}_{tag}.gif"
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        save_gif(frames, out_path, duration=args.duration)
        print(f"  GIF: {out_path}")


if __name__ == "__main__":
    main()

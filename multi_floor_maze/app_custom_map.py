"""
Ứng dụng web: custom map + chọn bộ não agent → xem agent chơi và tải GIF.

Chạy:
  streamlit run app_custom_map.py

Mở trình duyệt tại http://localhost:8501
"""
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from play_custom_map import (
    MODEL_ALIASES,
    OUTPUTS,
    build_custom_config,
    parse_enemies,
    resolve_model,
)
from generate_gifs import (
    GIF_DURATION,
    load_model,
    run_episode,
    save_gif,
)

DEMOS_DIR = OUTPUTS / "demos"


def list_available_models():
    """Liệt kê tên model có file tồn tại (để hiển thị trong dropdown)."""
    names = []
    for alias in MODEL_ALIASES:
        rel, _ = MODEL_ALIASES[alias]
        if rel.endswith(".zip"):
            p = OUTPUTS / rel
        else:
            p = OUTPUTS / rel / f"{rel}_final.zip"
            if not p.exists():
                p = OUTPUTS / f"{rel}_final.zip"
        if p.exists():
            names.append(alias)
    for d in OUTPUTS.iterdir():
        if d.is_dir() and d.name.startswith("exp"):
            z = d / f"{d.name}_final.zip"
            if z.exists() and d.name not in names:
                names.append(d.name)
    return sorted(set(names))


st.set_page_config(page_title="Custom Map — Agent Play", page_icon="🎮", layout="wide")
st.title("🎮 Custom Map — Agent chơi thử")
st.caption("Tùy chỉnh map (kích thước, bẫy, quái, khóa), chọn bộ não (model), xem agent chơi và tải GIF.")

with st.sidebar:
    st.header("🗺️ Thiết lập map")
    width = st.number_input("Chiều rộng (ô)", min_value=5, max_value=25, value=11)
    height = st.number_input("Chiều cao (ô)", min_value=5, max_value=25, value=11)
    n_floors = st.number_input("Số tầng", min_value=1, max_value=5, value=1)
    max_steps = st.number_input("Số bước tối đa (0 = tự tính)", min_value=0, value=0,
                                help="0 để tự tính theo kích thước")
    st.divider()
    st.subheader("Vật cản / đồ")
    traps = st.number_input("Số bẫy", min_value=0, value=0)
    health = st.number_input("Health pack", min_value=0, value=3)
    ammo = st.number_input("Ammo pack", min_value=0, value=2)
    locks = st.number_input("Số cặp cửa/chìa", min_value=0, max_value=5, value=0)
    enemies_text = st.text_input(
        "Quái (patrol:0, chaser:1, sniper:0)",
        value="",
        placeholder="vd: patrol:0, chaser:0",
        help="type:tầng, cách nhau bằng dấu phẩy"
    )
    st.divider()
    st.subheader("Agent")
    agent_hp = st.number_input("HP agent", min_value=1, value=10)
    agent_ammo = st.number_input("Ammo xuất phát", min_value=0, value=5)
    agent_stamina = st.number_input("Stamina", min_value=1, value=5)
    nf_max = max(0, n_floors - 1)
    start_floor = st.number_input("Tầng bắt đầu", min_value=0, max_value=nf_max, value=min(nf_max, 1))
    goal_floor = st.number_input("Tầng đích", min_value=0, max_value=nf_max, value=0)
    st.divider()
    st.subheader("Bộ não (model)")
    available = list_available_models()
    if not available:
        st.warning("Chưa có model nào trong outputs/. Chạy train trước.")
        model_choice = None
    else:
        model_choice = st.selectbox(
            "Chọn model",
            options=available,
            index=available.index("exp2_a2c") if "exp2_a2c" in available else 0,
            help="Model đã train (policy) dùng để điều khiển agent."
        )
    st.divider()
    seed = st.number_input("Seed sinh map", min_value=0, value=42)
    duration = st.number_input("GIF: ms mỗi frame", min_value=100, value=GIF_DURATION)
    run_btn = st.button("▶️ Chạy / Play", type="primary", use_container_width=True)

if run_btn:
    if not model_choice:
        st.error("Hãy chọn một model (cần train và có file trong outputs/).")
        st.stop()
    args = SimpleNamespace(
        width=width,
        height=height,
        n_floors=n_floors,
        max_steps=max_steps if max_steps else None,
        name="Custom",
        traps=traps,
        health=health,
        ammo=ammo,
        locks=locks,
        enemies=enemies_text or None,
        agent_hp=agent_hp,
        agent_ammo=agent_ammo,
        agent_stamina=agent_stamina,
        start_floor=start_floor,
        goal_floor=goal_floor,
    )
    config = build_custom_config(args)
    with st.spinner("Đang load model và chạy..."):
        try:
            model_path, model_type = resolve_model(model_choice)
            model = load_model(model_type, model_path)
            label = model_path.stem.replace("_final", "")
            frames, info, total_r = run_episode(model, config, label, seed=seed)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()
    win = info.get("win", False)
    steps = info.get("steps", len(frames) - 1)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kết quả", "🏆 WIN" if win else "💀 LOSE")
    with col2:
        st.metric("Steps", steps)
    with col3:
        st.metric("Reward", f"{total_r:+.2f}")
    gif_path = DEMOS_DIR / f"app_custom_{label}_seed{seed}_{'WIN' if win else 'LOSE'}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, gif_path, duration=duration)
    st.subheader("📹 GIF quá trình chơi")
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
    # Nhúng GIF dạng base64 để trình duyệt phát animation (st.image đôi khi chỉ ra frame đầu)
    gif_b64 = base64.b64encode(gif_bytes).decode()
    st.markdown(
        f'<img src="data:image/gif;base64,{gif_b64}" alt="playback" width="560" style="display: block;">',
        unsafe_allow_html=True,
    )
    st.download_button(
        "⬇️ Tải GIF",
        data=gif_bytes,
        file_name=gif_path.name,
        mime="image/gif",
        use_container_width=True,
    )
else:
    st.info("👈 Chỉnh map và chọn model ở sidebar, rồi bấm **Chạy / Play**.")

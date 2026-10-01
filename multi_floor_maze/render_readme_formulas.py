"""Compile README equations with LaTeX and export them as PNG images.

Requires a ``pdflatex`` executable on PATH and ``pymupdf`` (``pip install pymupdf``).
Run from any directory: ``python multi_floor_maze/render_readme_formulas.py``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import fitz
from PIL import Image, ImageDraw


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "formulas"
SCALE = 3.0
PADDING_X = 38
PADDING_Y = 24

FORMULAS = {
    "objective.png": r"""
J(\theta)=\mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\qquad \gamma=0.99
""",
    "approach_reward.png": r"""
r_t^{\mathrm{approach}}=0.10\,\operatorname{clip}(d_t-d_{t+1},-1,1)
""",
    "gae.png": r"""
\begin{aligned}
\delta_t &= r_t+\gamma V_\phi(o_{t+1})-V_\phi(o_t),\\
\hat A_t &= \sum_{\ell=0}^{T-t-1}(\gamma\lambda)^\ell\delta_{t+\ell},
\qquad \lambda=0.95
\end{aligned}
""",
    "ppo_ratio.png": r"""
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}
""",
    "ppo_clip.png": r"""
\begin{aligned}
L_{\mathrm{clip}}^{\mathrm{PPO}}(\theta)
&=\mathbb{E}_t\Bigl[\min\bigl(\rho_t(\theta)\hat A_t,\\
&\qquad\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\bigr)\Bigr],
\quad \epsilon=0.2
\end{aligned}
""",
    "a2c_actor.png": r"""
L_{\mathrm{actor}}^{\mathrm{A2C}}(\theta)
=-\mathbb{E}_t\!\left[\log\pi_\theta(a_t\mid o_t)\,\hat A_t\right]
""",
    "total_loss.png": r"""
\begin{aligned}
L_{\mathrm{total}}
&=L_{\mathrm{actor}}+c_v\mathbb{E}_t\!\left[(V_\phi(o_t)-\hat G_t)^2\right]\\
&\quad-c_e\mathbb{E}_t\!\left[\mathcal H(\pi_\theta(\cdot\mid o_t))\right],
\qquad c_v=0.5,\;c_e=0.01
\end{aligned}
""",
}

TEX_TEMPLATE = r"""\documentclass[12pt,border=5pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
\Large
\(\displaystyle
%s
\)
\end{document}
"""


def render(filename: str, equation: str) -> None:
    with TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        source = temp / "formula.tex"
        source.write_text(TEX_TEMPLATE % equation.strip(), encoding="utf-8")
        result = subprocess.run(
            [
                "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                "-file-line-error", "-output-directory", str(temp), str(source),
            ],
            cwd=temp,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(f"LaTeX failed for {filename}:\n{result.stdout[-3000:]}")

        with fitz.open(temp / "formula.pdf") as pdf:
            pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=True)
        formula = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)

    width = formula.width + PADDING_X * 2
    height = formula.height + PADDING_Y * 2
    card = Image.new("RGBA", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=15,
        fill="#f8fafc",
        outline="#cbd5e1",
        width=2,
    )
    card.alpha_composite(formula, (PADDING_X, PADDING_Y))
    card.convert("RGB").save(OUTPUT_DIR / filename, optimize=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, latex in FORMULAS.items():
        render(name, latex)
        print(OUTPUT_DIR / name)

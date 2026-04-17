from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "FIGURES"
EXPECTED_FIGURES = [
    "sp500_with_events.png",
    "car_by_event.png",
    "returns_by_group.png",
    "volatility_by_group.png",
    "formal_car_by_event.png",
    "formal_car_by_group.png",
    "abnormal_returns_by_group.png",
    "window_comparison.png",
    "event_sign_distribution.png",
    "return_vs_volatility.png",
    "car_distribution_by_group.png",
    "scatter_return_vs_volatility.png",
    "final_car_by_window_core.png",
    "final_car_by_window_expanded.png",
    "final_core_vs_expanded.png",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> None:
    if not FIGURES_DIR.exists():
        raise FileNotFoundError(f"ERRO: pasta de figuras nao encontrada: {FIGURES_DIR}")

    for figure_name in EXPECTED_FIGURES:
        figure_path = FIGURES_DIR / figure_name
        if not figure_path.exists():
            raise FileNotFoundError(f"ERRO: figura nao encontrada: {figure_path}")
        if figure_path.stat().st_size <= 0:
            raise ValueError(f"ERRO: figura vazia: {figure_path}")

    ok("figuras esperadas encontradas")
    ok("figuras com tamanho maior que zero")
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()

from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "FIGURES"
FIGURE_FILES = [
    FIGURES_DIR / "formal_car_by_event.png",
    FIGURES_DIR / "formal_car_by_group.png",
    FIGURES_DIR / "abnormal_returns_by_group.png",
]
MIN_WIDTH = 1000
MIN_HEIGHT = 600


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> None:
    for figure_path in FIGURE_FILES:
        if not figure_path.exists():
            raise FileNotFoundError(f"ERRO: figura nao encontrada: {figure_path}")
        if figure_path.stat().st_size == 0:
            raise ValueError(f"ERRO: figura vazia: {figure_path}")

        with Image.open(figure_path) as image:
            width, height = image.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            raise ValueError(
                "ERRO: resolucao da figura abaixo do minimo esperado. "
                f"Figura: {figure_path.name}, tamanho: {width}x{height}"
            )

    ok("figuras validas")
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()

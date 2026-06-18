import shutil
from pathlib import Path

from src.bai1 import run as run_bai1
from src.bai2 import run as run_bai2
from src.bai3 import run as run_bai3
from src.images_processing import run as create_grayscale_images


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data" / "input"
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"


def clear_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def run():
    clear_output()
    create_grayscale_images(str(INPUT_DIR), str(OUTPUT_DIR))
    run_bai1(str(GRAYSCALE_DIR), str(OUTPUT_DIR))
    run_bai2(str(GRAYSCALE_DIR), str(OUTPUT_DIR))
    run_bai3(str(GRAYSCALE_DIR), str(OUTPUT_DIR))


if __name__ == "__main__":
    run()

import csv
from pathlib import Path

import cv2
import numpy as np


GRAY_INPUT_DIR = Path("data/output/grayscale")
OUTPUT_DIR = Path("data/output")
LOW_GRAY = 30
HIGH_GRAY = 120


def calculate_histogram(gray_image):
    """Calculate a 256-level grayscale histogram."""
    return np.bincount(gray_image.ravel(), minlength=256).astype(np.int64)


def save_histogram_csv(gray_image, output_path):
    """Save histogram data as: gray_level,pixel_count."""
    histogram = calculate_histogram(gray_image)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for gray_level, pixel_count in enumerate(histogram):
            writer.writerow([gray_level, int(pixel_count)])


def build_equalization_table(gray_image):
    """
    Build a histogram equalization table with 7 columns:
    rk, nk, p(rk), sk, rk'=(L-1)sk, round(rk'), equalized nk.
    """
    histogram = calculate_histogram(gray_image)
    total_pixels = gray_image.size
    probabilities = histogram / total_pixels
    cumulative_probabilities = np.cumsum(probabilities)
    mapped_values = 255 * cumulative_probabilities
    rounded_mapped_values = np.rint(mapped_values).astype(np.uint8)
    equalized_image = rounded_mapped_values[gray_image]
    equalized_histogram = calculate_histogram(equalized_image)

    table_rows = []
    for gray_level in range(256):
        table_rows.append(
            [
                gray_level,
                int(histogram[gray_level]),
                probabilities[gray_level],
                cumulative_probabilities[gray_level],
                mapped_values[gray_level],
                int(rounded_mapped_values[gray_level]),
                int(equalized_histogram[gray_level]),
            ]
        )

    return equalized_image, table_rows


def save_equalization_table_csv(table_rows, output_path):
    """Save the histogram equalization table as a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rk", "nk", "p_r_rk", "sk", "rk_prime", "round_rk_prime", "nk_after_equalization"])
        for row in table_rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    f"{row[2]:.6f}",
                    f"{row[3]:.6f}",
                    f"{row[4]:.6f}",
                    row[5],
                    row[6],
                ]
            )


def draw_histogram(gray_image, output_path, title):
    """Draw a histogram image and save it as PNG."""
    histogram = calculate_histogram(gray_image)

    bar_gap = 1
    bar_width = 3
    plot_width = 256 * (bar_width + bar_gap)
    plot_height = 360
    left_margin = 150
    right_margin = 60
    top_margin = 70
    bottom_margin = 80
    canvas_width = left_margin + plot_width + right_margin
    canvas_height = top_margin + plot_height + bottom_margin

    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    plot_left = left_margin
    plot_top = top_margin
    plot_right = left_margin + plot_width
    plot_bottom = top_margin + plot_height

    text_color = (0, 0, 0)
    axis_color = (35, 35, 35)
    grid_color = (225, 225, 225)
    bar_color = (35, 95, 190)

    max_count = histogram.max()
    grid_line_count = 5
    for index in range(1, grid_line_count + 1):
        grid_y = plot_bottom - index * plot_height // grid_line_count
        grid_value = int(round(max_count * index / grid_line_count))
        cv2.line(canvas, (plot_left, grid_y), (plot_right, grid_y), grid_color, 1)

        grid_label = str(grid_value)
        label_size, _ = cv2.getTextSize(grid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas,
            grid_label,
            (plot_left - label_size[0] - 12, grid_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_8,
        )

    cv2.arrowedLine(canvas, (plot_left, plot_bottom), (plot_left, plot_top - 18), axis_color, 2, tipLength=0.025)
    cv2.arrowedLine(canvas, (plot_left, plot_bottom), (plot_right + 25, plot_bottom), axis_color, 2, tipLength=0.025)

    if max_count > 0:
        for gray_level, pixel_count in enumerate(histogram):
            bar_height = int((pixel_count / max_count) * (plot_height - 8))
            bar_left = plot_left + gray_level * (bar_width + bar_gap)
            bar_right = bar_left + bar_width - 1
            bar_top = plot_bottom - bar_height
            cv2.rectangle(canvas, (bar_left, bar_top), (bar_right, plot_bottom), bar_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.8
    title_thickness = 2
    text_scale = 0.6
    text_thickness = 2

    title_size, _ = cv2.getTextSize(title, font, title_scale, title_thickness)
    title_x = (canvas_width - title_size[0]) // 2
    cv2.putText(canvas, title, (title_x, 38), font, title_scale, text_color, title_thickness, cv2.LINE_8)

    cv2.putText(canvas, "0", (plot_left - 5, plot_bottom + 28), font, text_scale, text_color, text_thickness, cv2.LINE_8)
    cv2.putText(canvas, "255", (plot_right - 40, plot_bottom + 28), font, text_scale, text_color, text_thickness, cv2.LINE_8)
    cv2.putText(
        canvas,
        "Muc xam",
        (plot_left + plot_width // 2 - 45, plot_bottom + 62),
        font,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_8,
    )

    y_axis_label = "So pixel"
    y_label_size, _ = cv2.getTextSize(y_axis_label, font, text_scale, text_thickness)
    y_label_image = np.full((y_label_size[1] + 16, y_label_size[0] + 16, 3), 255, dtype=np.uint8)
    cv2.putText(
        y_label_image,
        y_axis_label,
        (8, y_label_size[1] + 8),
        font,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_8,
    )
    y_label_image = cv2.rotate(y_label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y_label_top = plot_top + (plot_height - y_label_image.shape[0]) // 2
    y_label_left = 18
    canvas[
        y_label_top:y_label_top + y_label_image.shape[0],
        y_label_left:y_label_left + y_label_image.shape[1],
    ] = y_label_image

    cv2.imwrite(str(output_path), canvas)


def shrink_histogram_range(equalized_image, lower_bound=LOW_GRAY, upper_bound=HIGH_GRAY):
    """
    Shrink the gray-level range of the equalized image to [lower_bound, upper_bound].
    Formula: s = ((smax - smin) / (rmax - rmin)) * (r - rmin) + smin.
    """
    input_min = int(equalized_image.min())
    input_max = int(equalized_image.max())
    if input_max == input_min:
        return np.full_like(equalized_image, lower_bound, dtype=np.uint8)

    image_float = equalized_image.astype(np.float32)
    narrowed_image = ((upper_bound - lower_bound) / (input_max - input_min)) * (image_float - input_min) + lower_bound
    return np.clip(narrowed_image, lower_bound, upper_bound).astype(np.uint8)


def get_output_folder(file_stem):
    """Convert image_01_gray to data/output/img01."""
    image_number = file_stem.replace("image_", "").replace("_gray", "")
    return OUTPUT_DIR / f"img{image_number}"


def get_image_prefix(file_stem):
    """Convert image_01_gray to image_01."""
    return file_stem.replace("_gray", "")


def process_one_image(gray_image_path):
    gray_image = cv2.imread(str(gray_image_path), cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print(f"Cannot read grayscale image: {gray_image_path}")
        return

    output_folder = get_output_folder(gray_image_path.stem)
    image_prefix = get_image_prefix(gray_image_path.stem)
    if not output_folder.exists():
        print(f"Skip {gray_image_path.name}: missing output folder {output_folder}")
        return

    equalized_image, equalization_table = build_equalization_table(gray_image)
    narrowed_image = shrink_histogram_range(equalized_image)

    cv2.imwrite(str(output_folder / f"{image_prefix}_H2_equalized.png"), equalized_image)
    cv2.imwrite(str(output_folder / f"{image_prefix}_H2_narrow_30_120.png"), narrowed_image)

    save_histogram_csv(gray_image, output_folder / f"{image_prefix}_H1_histogram.csv")
    save_equalization_table_csv(equalization_table, output_folder / f"{image_prefix}_H2_equalization_table.csv")
    save_histogram_csv(equalized_image, output_folder / f"{image_prefix}_H2_histogram.csv")
    save_histogram_csv(narrowed_image, output_folder / f"{image_prefix}_H2_narrow_30_120_histogram.csv")

    draw_histogram(gray_image, output_folder / f"{image_prefix}_H1_histogram.png", "H1 - Histogram anh xam")
    draw_histogram(equalized_image, output_folder / f"{image_prefix}_H2_histogram.png", "H2 - Histogram can bang")
    draw_histogram(
        narrowed_image,
        output_folder / f"{image_prefix}_H2_narrow_30_120_histogram.png",
        "Histogram H2 thu hep [30, 120]",
    )

    print(f"Saved results for {gray_image_path.name} to {output_folder}")


def process_all_images():
    if not GRAY_INPUT_DIR.exists():
        print(f"Cannot find grayscale folder: {GRAY_INPUT_DIR}")
        return

    gray_image_paths = sorted(GRAY_INPUT_DIR.glob("*_gray.png"))
    if not gray_image_paths:
        print(f"No *_gray.png files found in {GRAY_INPUT_DIR}")
        return

    for gray_image_path in gray_image_paths:
        process_one_image(gray_image_path)


if __name__ == "__main__":
    process_all_images()

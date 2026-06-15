from pathlib import Path

import cv2
import numpy as np


GRAY_INPUT_DIR = Path("data/output/grayscale")
OUTPUT_DIR = Path("data/output")
LOW_GRAY = 30
HIGH_GRAY = 120


def tinh_histogram(anh_xam):
    """Tinh histogram 256 muc xam cua anh."""
    hist = cv2.calcHist([anh_xam], [0], None, [256], [0, 256])
    return hist.flatten()


def ve_histogram(anh_xam, duong_dan_luu, tieu_de):
    """Ve histogram va luu thanh file anh PNG."""
    hist = tinh_histogram(anh_xam)

    chieu_rong = 512
    chieu_cao = 360
    le_trai = 45
    le_tren = 35
    le_duoi = 35
    vung_ve_rong = chieu_rong
    vung_ve_cao = chieu_cao - le_tren - le_duoi

    canvas = np.full((chieu_cao, chieu_rong + le_trai + 15, 3), 255, dtype=np.uint8)
    x0 = le_trai
    y0 = le_tren
    x1 = le_trai + vung_ve_rong
    y1 = le_tren + vung_ve_cao

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)

    gia_tri_lon_nhat = hist.max()
    if gia_tri_lon_nhat > 0:
        do_rong_cot = vung_ve_rong / 256
        for muc_xam, so_pixel in enumerate(hist):
            cot_cao = int((so_pixel / gia_tri_lon_nhat) * (vung_ve_cao - 8))
            cot_x0 = int(x0 + muc_xam * do_rong_cot)
            cot_x1 = int(x0 + (muc_xam + 1) * do_rong_cot)
            cot_y0 = y1 - cot_cao
            cv2.rectangle(canvas, (cot_x0, cot_y0), (max(cot_x1, cot_x0 + 1), y1), (40, 80, 170), -1)

    cv2.putText(canvas, tieu_de, (x0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(canvas, "0", (x0 - 2, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(canvas, "255", (x1 - 30, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.imwrite(str(duong_dan_luu), canvas)


def can_bang_histogram(anh_xam):
    """Can bang histogram cua anh xam."""
    return cv2.equalizeHist(anh_xam)


def thu_hep_histogram(anh_h2, can_duoi=LOW_GRAY, can_tren=HIGH_GRAY):
    """
    Hieu chinh thu hep mien muc xam cua anh H2 ve khoang [can_duoi, can_tren].
    Cong thuc: s = can_duoi + r * (can_tren - can_duoi) / 255.
    """
    anh_float = anh_h2.astype(np.float32)
    anh_thu_hep = can_duoi + anh_float * (can_tren - can_duoi) / 255
    return np.clip(anh_thu_hep, can_duoi, can_tren).astype(np.uint8)


def lay_thu_muc_output(ten_file):
    """
    Chuyen ten image_01_gray.png thanh thu muc data/output/img01.
    """
    so_thu_tu = ten_file.replace("image_", "").replace("_gray", "")
    return OUTPUT_DIR / f"img{so_thu_tu}"


def xu_ly_mot_anh(duong_dan_anh_xam):
    anh_xam = cv2.imread(str(duong_dan_anh_xam), cv2.IMREAD_GRAYSCALE)
    if anh_xam is None:
        print(f"Khong doc duoc anh xam: {duong_dan_anh_xam}")
        return

    ten_goc = duong_dan_anh_xam.stem
    thu_muc_luu = lay_thu_muc_output(ten_goc)

    if not thu_muc_luu.exists():
        print(f"Bo qua {duong_dan_anh_xam.name}: chua co thu muc {thu_muc_luu}")
        return

    anh_h2 = can_bang_histogram(anh_xam)
    anh_h2_thu_hep = thu_hep_histogram(anh_h2)

    cv2.imwrite(str(thu_muc_luu / "H2_equalized.png"), anh_h2)
    cv2.imwrite(str(thu_muc_luu / "H2_narrow_30_120.png"), anh_h2_thu_hep)

    ve_histogram(anh_xam, thu_muc_luu / "H1_histogram.png", "H1 - Histogram anh xam")
    ve_histogram(anh_h2, thu_muc_luu / "H2_histogram.png", "H2 - Histogram can bang")
    ve_histogram(
        anh_h2_thu_hep,
        thu_muc_luu / "H2_narrow_30_120_histogram.png",
        "Histogram H2 thu hep [30, 120]",
    )

    print(f"Da luu ket qua {duong_dan_anh_xam.name} vao {thu_muc_luu}")


def xu_ly_bai1():
    if not GRAY_INPUT_DIR.exists():
        print(f"Khong tim thay thu muc anh xam: {GRAY_INPUT_DIR}")
        return

    danh_sach_anh_xam = sorted(GRAY_INPUT_DIR.glob("*_gray.png"))
    if not danh_sach_anh_xam:
        print(f"Khong co anh *_gray.png trong {GRAY_INPUT_DIR}")
        return

    for duong_dan_anh_xam in danh_sach_anh_xam:
        xu_ly_mot_anh(duong_dan_anh_xam)


if __name__ == "__main__":
    xu_ly_bai1()

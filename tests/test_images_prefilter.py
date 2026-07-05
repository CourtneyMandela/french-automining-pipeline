import cv2
import numpy as np

from french_mining.images import (
    EDGE_DENSITY_TEXT_THRESHOLD,
    _edge_density,
    _face_area_ratio_from_boxes,
    is_talking_head_or_text,
)


def write_image(path, array):
    cv2.imwrite(str(path), array)


def make_checkerboard(size: int = 200, block: int = 4) -> np.ndarray:
    """Coarse (multi-pixel-block) checkerboard, so the pattern survives
    Canny's Gaussian smoothing / JPEG compression -- a single-pixel-period
    checkerboard gets filtered out as noise and doesn't test anything.
    """
    block_grid = np.indices((size // block + 1, size // block + 1)).sum(axis=0) % 2
    pattern = np.kron(block_grid, np.ones((block, block))) * 255
    return pattern[:size, :size].astype(np.uint8)


def test_face_area_ratio_from_boxes_returns_zero_when_no_faces():
    assert _face_area_ratio_from_boxes([], (480, 640)) == 0.0


def test_face_area_ratio_from_boxes_computes_largest_face_fraction():
    # Frame is 100x100 = 10000 px; a 40x40 face box = 1600 px -> 0.16 ratio.
    faces = [(10, 10, 40, 40)]
    assert _face_area_ratio_from_boxes(faces, (100, 100)) == 0.16


def test_face_area_ratio_from_boxes_uses_largest_when_multiple_faces():
    faces = [(0, 0, 10, 10), (0, 0, 50, 50)]
    ratio = _face_area_ratio_from_boxes(faces, (100, 100))
    assert ratio == 0.25  # 50*50 / 10000


def test_edge_density_is_low_for_a_solid_color_image(tmp_path):
    image = np.full((200, 200), 128, dtype=np.uint8)
    density = _edge_density(image)
    assert density < 0.05


def test_edge_density_is_high_for_a_checkerboard_pattern():
    image = make_checkerboard()
    density = _edge_density(image)
    assert density > EDGE_DENSITY_TEXT_THRESHOLD


def test_is_talking_head_or_text_returns_false_for_plain_photo_like_image(tmp_path):
    # A smooth gradient -- low edge density, no detectable face.
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "gradient.jpg"
    write_image(path, image)

    assert is_talking_head_or_text(path) is False


def test_is_talking_head_or_text_returns_true_for_text_heavy_checkerboard(tmp_path):
    checker = make_checkerboard()
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "checker.png"
    write_image(path, image)

    assert is_talking_head_or_text(path) is True


def test_is_talking_head_or_text_returns_true_for_unreadable_path(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    assert is_talking_head_or_text(missing) is True

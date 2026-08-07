import numpy as np

from tools.video_label_mapping import convert_mask


def test_vipseg_panoptic_decoding():
    # VOID, sky stuff, person thing instance, mountain stuff, ball thing instance.
    raw = np.array([[0, 29, 6103], [30, 7501, 14]], dtype=np.uint16)
    mapping = {29: 1, 61: 2, 30: 12, 75: 11}
    converted = convert_mask(raw, mapping, ignore_index=255, panoptic=True)
    expected = np.array([[255, 1, 2], [12, 11, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(converted, expected)


def test_vspw_semantic_decoding():
    raw = np.array([[0, 29, 61, 30]], dtype=np.uint8)
    mapping = {29: 1, 61: 2, 30: 12}
    expected = np.array([[255, 1, 2, 12]], dtype=np.uint8)
    np.testing.assert_array_equal(convert_mask(raw, mapping, panoptic=False), expected)

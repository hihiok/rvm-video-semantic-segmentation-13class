import numpy as np

from tools.video_label_mapping import convert_mask


def test_vipseg_panoptic_decoding():
    # VOID, sky stuff, person thing instance, mountain stuff, ball thing instance.
    raw = np.array([[0, 29, 6103], [30, 7501, 14]], dtype=np.uint16)
    mapping = {29: 1, 61: 2, 30: 12, 75: 11}
    converted = convert_mask(raw, mapping, ignore_index=255, panoptic=True)
    expected = np.array([[255, 1, 2], [12, 11, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(converted, expected)


def test_vspw_semantic_decoding_preserves_source_ignore():
    raw = np.array([[0, 29, 253, 61, 30, 255]], dtype=np.uint8)
    mapping = {29: 1, 61: 2, 30: 12}
    expected = np.array([[255, 1, 255, 2, 12, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(convert_mask(raw, mapping, panoptic=False), expected)
    assert raw[0, 2] == 253, "Conversion must not modify the official source mask"


def test_vspw_253_respects_custom_ignore_index():
    raw = np.array([[29, 253, 255]], dtype=np.uint8)
    expected = np.array([[1, 254, 254]], dtype=np.uint8)
    np.testing.assert_array_equal(
        convert_mask(raw, {29: 1}, ignore_index=254, panoptic=False), expected
    )


def test_other_unknown_vspw_category_ids_remain_errors():
    with np.testing.assert_raises(ValueError):
        convert_mask(np.array([[252]], dtype=np.uint8), {}, panoptic=False)


def test_vipseg_panoptic_253_is_decoded_as_an_instance_not_vspw_void():
    expected = np.array([[7]], dtype=np.uint8)
    np.testing.assert_array_equal(
        convert_mask(np.array([[253]], dtype=np.uint16), {2: 7}, panoptic=True), expected
    )

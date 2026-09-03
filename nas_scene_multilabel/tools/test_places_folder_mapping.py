#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_dataset import (  # noqa: E402
    BUNDLED_PLACES_IO,
    build_places_alias_map,
    canonicalize_places_category,
    parse_io,
    places_labels,
)


def main():
    io_map = parse_io(BUNDLED_PLACES_IO)
    aliases = build_places_alias_map(io_map)

    cases = {
        "field-cultivated": "field/cultivated",
        "office-cubicles": "office_cubicles",
        "basketball-court-indoor": "basketball_court/indoor",
        "desert-sand": "desert/sand",
        "lake-natural": "lake/natural",
        "airport-terminal": "airport_terminal",
        "office": "office",
        "beach": "beach",
    }
    for folder, expected in cases.items():
        got = canonicalize_places_category(folder, aliases)
        print(folder, "->", got)
        assert got == expected, (folder, got, expected)

    y = places_labels("office", io_map)
    assert y["indoor"] == 1 and y["outdoor"] == 0 and y["office"] == 1

    y = places_labels("beach", io_map)
    assert y["outdoor"] == 1 and y["indoor"] == 0 and y["landscape"] == 1

    y = places_labels("basketball_court/indoor", io_map)
    assert y["indoor"] == 1 and y["sports"] == 1

    print("PLACES_FOLDER_MAPPING_TEST=PASS")


if __name__ == "__main__":
    main()

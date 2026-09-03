"""Taxonomy and conservative source mappings for FSD 8-label scene tagging."""

LABELS = [
    "night",
    "indoor",
    "rain_snow",
    "office",
    "outdoor",
    "landscape",
    "sports",
    "objective_image",
]

DISPLAY_NAMES = {
    "night": "夜景",
    "indoor": "室内",
    "rain_snow": "雨/雪",
    "office": "办公场景",
    "outdoor": "户外",
    "landscape": "风景",
    "sports": "运动",
    "objective_image": "客观图",
}

# Places365 uses canonical slash/underscore names. The local downloaded dataset
# flattens these into hyphenated directory names; prepare_manifest.py resolves
# folder names against the official taxonomy instead of guessing IO semantics.
LANDSCAPE_POS = {
    "badlands", "bamboo_forest", "beach", "butte", "canyon", "cliff", "coast",
    "creek", "desert/sand", "desert/vegetation", "field/cultivated", "field/wild",
    "forest/broadleaf", "forest_path", "glacier", "harbor", "iceberg", "islet",
    "lagoon", "lake/natural", "marsh", "mountain", "mountain_path", "mountain_snowy",
    "ocean", "orchard", "pasture", "rainforest", "rice_paddy", "river", "skyline",
    "snowfield", "swamp", "tundra", "valley", "vineyard", "volcano", "waterfall",
    "wheat_field", "wind_farm", "downtown", "village",
}
LANDSCAPE_NEG = {
    "bathroom", "bedroom", "closet", "corridor", "dining_room", "dorm_room",
    "elevator_lobby", "hospital_room", "hotel_room", "kitchen", "laundromat",
    "locker_room", "office", "office_cubicles", "operating_room", "server_room",
    "storage_room", "television_room", "utility_room", "waiting_room",
}

SPORTS_POS = {
    "arena/hockey", "athletic_field/outdoor", "baseball_field", "basketball_court/indoor",
    "boxing_ring", "football_field", "golf_course", "gymnasium/indoor",
    "ice_skating_rink/indoor", "ice_skating_rink/outdoor", "martial_arts_gym",
    "racecourse", "raceway", "ski_resort", "ski_slope", "soccer_field",
    "stadium/baseball", "stadium/football", "stadium/soccer", "swimming_pool/indoor",
    "swimming_pool/outdoor", "volleyball_court/outdoor",
}
SPORTS_NEG = {
    "bathroom", "bedroom", "bookstore", "cafeteria", "classroom", "closet",
    "coffee_shop", "conference_room", "dining_room", "hotel_room", "kitchen",
    "library/indoor", "living_room", "museum/indoor", "office", "office_cubicles",
    "restaurant", "shopping_mall/indoor", "supermarket", "waiting_room",
    "beach", "canyon", "coast", "desert/sand", "forest/broadleaf", "mountain",
    "ocean", "river", "street", "valley", "waterfall",
}

OFFICE_POS = {
    "office", "office_cubicles", "conference_room", "conference_center",
    "computer_room", "home_office", "reception",
}
OFFICE_NEG = {
    "bathroom", "bedroom", "bookstore", "cafeteria", "classroom", "closet",
    "dining_room", "hotel_room", "kitchen", "laundromat", "living_room",
    "locker_room", "movie_theater/indoor", "museum/indoor", "restaurant",
    "shopping_mall/indoor", "supermarket", "television_room", "waiting_room",
}

# Places can provide conservative snow positives. Generic rain is supplied by
# 10_scenes because Places365 has no reliable rainy-weather taxonomy.
RAIN_SNOW_PLACES_POS = {
    "glacier", "ice_floe", "ice_shelf", "iceberg", "mountain_snowy", "snowfield",
    "ski_resort", "ski_slope", "ice_skating_rink/outdoor",
}
RAIN_SNOW_PLACES_NEG = {
    "desert/sand", "desert/vegetation", "beach", "ocean", "office", "bedroom",
    "kitchen", "living_room", "restaurant", "supermarket",
}

# 10_scenes mappings are deliberately explicit/conservative. Folder names are
# normalized to lowercase with spaces/hyphens collapsed to underscores.
TEN_SCENES_ALIASES = {
    "night": {
        "night", "night_scene", "night_scenes", "nightscape", "night_view",
    },
    "indoor": {
        "indoor", "indoors", "indoor_scene", "indoor_scenes",
    },
    "rain_snow": {
        "rain", "rainy", "rain_scene", "snow", "snowy", "snow_scene",
        "rain_snow", "rain_and_snow", "rain_or_snow",
    },
    "office": {
        "office", "office_scene", "office_scenes", "workplace",
    },
    "outdoor": {
        "outdoor", "outdoors", "outdoor_scene", "outdoor_scenes",
    },
    "landscape": {
        "landscape", "landscapes", "scenery", "scenic", "landscape_scene",
    },
    "sports": {
        "sport", "sports", "sport_scene", "sports_scene", "sports_scenes",
    },
    "objective_image": {
        "computer_synthesized", "computer_synthetic", "computer_generated",
        "objective", "objective_image", "test_pattern", "test_patterns",
        "resolution_chart", "resolution_charts",
    },
}

# Existing custom 13-class segmentation dataset IDs.
SEG_CLASS_IDS = {
    "sky": 1,
    "plant": 3,
    "building": 4,
    "water": 7,
    "desert": 8,
    "ice_or_snow": 9,
    "mountain": 12,
}

LABELS = [
    "indoor",
    "outdoor",
    "landscape",
    "sports",
    "food",
    "animal",
    "building",
    "sky",
    "office",
]

DISPLAY_NAMES = {
    "indoor": "室内",
    "outdoor": "户外",
    "landscape": "风景",
    "sports": "运动",
    "food": "美食",
    "animal": "动物",
    "building": "建筑",
    "sky": "蓝天/天空",
    "office": "办公",
}

# Product definition: "蓝天" means visible sky of any color, not specifically blue.

LANDSCAPE_POS = {
    "badlands", "bamboo_forest", "beach", "butte", "canyon", "cliff", "coast",
    "creek", "desert/sand", "desert/vegetation", "field/cultivated", "field/wild",
    "forest/broadleaf", "forest_path", "glacier", "harbor", "iceberg", "islet",
    "lagoon", "lake/natural", "marsh", "mountain", "mountain_path", "mountain_snowy",
    "ocean", "orchard", "pasture", "rainforest", "rice_paddy", "river", "skyline",
    "snowfield", "swamp", "tundra", "valley", "vineyard", "volcano", "waterfall",
    "wheat_field", "wind_farm",
}

# Safe negatives: clearly non-scenic indoor content.
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

OFFICE_POS = {"office", "office_cubicles", "conference_room"}
OFFICE_NEG = {
    "bathroom", "bedroom", "bookstore", "cafeteria", "classroom", "closet",
    "dining_room", "hotel_room", "kitchen", "laundromat", "living_room",
    "locker_room", "movie_theater/indoor", "museum/indoor", "restaurant",
    "shopping_mall/indoor", "supermarket", "television_room", "waiting_room",
}

COCO_FOOD = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake",
}
COCO_ANIMAL = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe",
}

# Existing 13-class segmentation mapping used by this project.
SEG_CLASS_IDS = {"sky": 1, "building": 4, "food": 6}

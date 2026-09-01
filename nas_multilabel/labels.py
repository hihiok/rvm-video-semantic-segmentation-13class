"""NAS 9-label taxonomy and zero-shot prompt ensembles."""

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

# Positive/contrast prompt pairs are used independently per label.  This avoids
# a 9-way softmax, because NAS tagging is multi-label and labels can co-exist.
PROMPTS = {
    "indoor": {
        "positive": [
            "a photo taken indoors",
            "an indoor scene",
            "the inside of a building or room",
        ],
        "negative": [
            "a photo taken outdoors",
            "an outdoor scene",
            "an open-air exterior environment",
        ],
    },
    "outdoor": {
        "positive": [
            "a photo taken outdoors",
            "an outdoor scene",
            "an open-air exterior environment",
        ],
        "negative": [
            "a photo taken indoors",
            "an indoor scene",
            "the inside of a building or room",
        ],
    },
    "landscape": {
        "positive": [
            "a landscape or scenery photo",
            "a scenic natural or city landscape",
            "a wide view of scenery",
        ],
        "negative": [
            "a close-up object photo rather than scenery",
            "a non-landscape close-up scene",
            "a photo whose main subject is not scenery",
        ],
    },
    "sports": {
        "positive": [
            "a sports scene",
            "people doing or playing sports",
            "a sporting event or sports venue",
        ],
        "negative": [
            "an everyday scene without sports",
            "a non-sports activity",
            "a scene unrelated to sports",
        ],
    },
    "food": {
        "positive": [
            "a photo containing food",
            "a meal, dish, or prepared food",
            "food is clearly visible in the image",
        ],
        "negative": [
            "a scene without food",
            "a photo of non-food objects",
            "no meal or food is visible",
        ],
    },
    "animal": {
        "positive": [
            "a photo containing an animal",
            "a pet, wild animal, bird, or livestock",
            "an animal is clearly visible in the image",
        ],
        "negative": [
            "a scene without animals",
            "a photo containing only people or inanimate objects",
            "no animal is visible",
        ],
    },
    "building": {
        "positive": [
            "a building is clearly visible",
            "a photo showing a building or architectural structure",
            "a visible building exterior or large architectural structure",
        ],
        "negative": [
            "a scene without a visible building",
            "a photo with no building or architecture visible",
            "no architectural structure is visible",
        ],
    },
    # Product UI calls this 蓝天, but the agreed semantic definition is simply
    # that sky is visibly present; it does not need to be blue.
    "sky": {
        "positive": [
            "the sky is visibly present in the image",
            "a photo with visible sky",
            "sky or clouds are clearly visible",
        ],
        "negative": [
            "a photo with no visible sky",
            "the sky is not visible",
            "an enclosed scene without visible sky",
        ],
    },
    "office": {
        "positive": [
            "an office or workplace environment",
            "an office scene",
            "a meeting room, cubicle, or professional office workspace",
        ],
        "negative": [
            "a non-office environment",
            "a scene that is not an office or workplace",
            "a home, outdoor, leisure, or non-office scene",
        ],
    },
}

# Curated Places365 category aliases used to create a reproducible probe set.
# Matching is normalized and suffix-aware, so it works for common Places365
# layouts such as data_large/a/airport_terminal/*.jpg.
PLACES_GROUPS = {
    "indoor": {
        "airport_terminal", "art_gallery", "auditorium", "bakery/shop",
        "banquet_hall", "bar", "bathroom", "bedroom", "bookstore",
        "bowling_alley", "cafeteria", "classroom", "closet",
        "conference_room", "corridor", "dining_hall", "dining_room",
        "elevator/interior", "gymnasium/indoor", "hospital_room",
        "hotel_room", "kitchen", "laboratory/wet", "laundromat",
        "library/indoor", "lobby", "locker_room", "mall/indoor",
        "movie_theater/indoor", "museum/indoor", "office",
        "office_cubicles", "restaurant", "server_room",
        "shopping_mall/indoor", "supermarket", "television_room",
        "waiting_room", "warehouse/indoor",
    },
    "outdoor": {
        "airfield", "alley", "amphitheater", "amusement_park",
        "apartment_building/outdoor", "arch", "athletic_field/outdoor",
        "badlands", "beach", "boardwalk", "bridge", "campus", "canyon",
        "castle", "cemetery", "coast", "construction_site", "courtyard",
        "desert/sand", "downtown", "field/cultivated", "field/wild",
        "forest/broadleaf", "forest_path", "garden", "golf_course",
        "harbor", "highway", "lake/natural", "mountain", "mountain_path",
        "ocean", "park", "parking_lot", "pasture", "plaza", "rainforest",
        "residential_neighborhood", "river", "road", "ski_slope", "skyline",
        "stadium/baseball", "stadium/football", "stadium/soccer", "street",
        "swimming_pool/outdoor", "valley", "waterfall", "wheat_field",
    },
    "landscape": {
        "badlands", "beach", "canyon", "coast", "desert/sand",
        "field/cultivated", "field/wild", "forest/broadleaf", "forest_path",
        "glacier", "harbor", "lake/natural", "mountain", "mountain_path",
        "ocean", "rainforest", "river", "skyline", "valley", "waterfall",
        "wheat_field",
    },
    "sports": {
        "athletic_field/outdoor", "baseball_field", "basketball_court/indoor",
        "basketball_court/outdoor", "football_field", "golf_course",
        "ice_skating_rink/indoor", "ice_skating_rink/outdoor", "racecourse",
        "ski_slope", "soccer_field", "stadium/baseball", "stadium/football",
        "stadium/soccer", "swimming_pool/indoor", "swimming_pool/outdoor",
        "tennis_court/indoor", "tennis_court/outdoor", "volleyball_court/outdoor",
    },
    "office": {"office", "office_cubicles", "conference_room"},
}

COCO_FOOD = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}
COCO_ANIMAL = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
}
COCO_SPORTS = {
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
}

# Existing 13-class semantic data mapping used by this project.
SEG_CLASS_IDS = {"sky": 1, "building": 4, "food": 6}

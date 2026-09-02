#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/pub1/z00919662/scene_multilabel/datasets_raw}"
PLACES="${ROOT}/places365"
COCO="${ROOT}/coco2017"
mkdir -p "${PLACES}" "${COCO}"

WGET=(wget -c --no-check-certificate --timeout=30 --tries=20)

# Places365-Standard 256px: ~24 GB train + ~0.5 GB val.
cd "${PLACES}"
if [[ ! -f .train_downloaded ]]; then
  "${WGET[@]}" http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
  touch .train_downloaded
fi
if [[ ! -f .val_downloaded ]]; then
  "${WGET[@]}" http://data.csail.mit.edu/places/places365/val_256.tar
  touch .val_downloaded
fi
if [[ ! -f .filelist_downloaded ]]; then
  "${WGET[@]}" http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
  touch .filelist_downloaded
fi

[[ -d data_256 ]] || tar -xf train_256_places365standard.tar
[[ -d val_256 ]] || tar -xf val_256.tar
# The file-list archive may unpack metadata into a subdirectory; the Python
# builder searches recursively, so no relocation is needed.
if ! find . -type f -name places365_val.txt -print -quit | grep -q .; then
  tar -xf filelist_places365-standard.tar
fi

# Official taxonomy and indoor/outdoor mapping.
if [[ ! -f categories_places365.txt ]]; then
  "${WGET[@]}" https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt
fi
if [[ ! -f IO_places365.txt ]]; then
  "${WGET[@]}" https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt
fi

# COCO 2017: ~18 GB train + ~1 GB val + annotations.
cd "${COCO}"
for item in \
  "http://images.cocodataset.org/zips/train2017.zip" \
  "http://images.cocodataset.org/zips/val2017.zip" \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; do
  file="$(basename "${item}")"
  [[ -f "${file}" ]] || "${WGET[@]}" "${item}"
done

[[ -d train2017 ]] || unzip -q train2017.zip
[[ -d val2017 ]] || unzip -q val2017.zip
[[ -f annotations/instances_train2017.json ]] || unzip -q annotations_trainval2017.zip

echo "DATASETS_READY"
echo "PLACES_ROOT=${PLACES}"
echo "COCO_ROOT=${COCO}"

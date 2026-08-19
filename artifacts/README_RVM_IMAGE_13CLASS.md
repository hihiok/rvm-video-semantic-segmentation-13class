# RVM image semantic segmentation: 13 classes

Prepared from `RobustVideoMatting-semantic-12class-512` with category-count-only changes.

- Immutable order: `background, sky, person, plant, building, flower, food, water, desert, ice_or_snow, text, ball, mountain`
- `mountain=12`; `255=ignore`
- Default semantic head channels: 13
- Training/data/model logic otherwise unchanged
- Dataset: `/data/pub1/y00841348/dataset/AI-Seg-12class/cocostuff164k+ade20k_13class`
- Deployment target: `/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512`

Verify the archive against the adjacent SHA256 file before extraction.

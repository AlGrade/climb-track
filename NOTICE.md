# Third-party notices

ClimbTrack itself is released under the MIT License (see [LICENSE](LICENSE)). That license covers
the source code in this repository only. The models the pipeline runs on are **not** redistributed
here — they are downloaded on demand by `climbtrack download-yolo` and `climbtrack download-sapiens`,
and each carries its own terms.

## Sapiens2 (pose estimation)

Sapiens2 materials are published by Meta and are not redistributed by this repository. Review Meta's
current Sapiens2 license before downloading or using the weights, and note that its terms govern what
you may do with the model and its outputs.

## YOLO11x (person detection)

The YOLO11x checkpoint is published by Ultralytics under **AGPL-3.0**. The AGPL is a strong copyleft
license: if you build on the detection step and make the result available to others over a network,
its terms may require you to publish your corresponding source. Ultralytics also offers a commercial
license for cases where that is not acceptable. This affects your use of the downloaded weights, not
the MIT-licensed code in this repository.

## Python dependencies

Runtime dependencies are declared in [pyproject.toml](pyproject.toml) and pinned in `uv.lock`. They
retain their own licenses, which are not reproduced here.

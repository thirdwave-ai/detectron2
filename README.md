# Steps to create a wheel and upload it.

1. Make sure you have gsutil installed or follow the install instructions in the thirdwave README.
1. Inside docker create the wheel by running:
    ```bash
    python3 setup.py sdist bdist_wheel
    ```
1. Outside docker upload the dist folder via gsutil:
    ```bash
    gsutil cp -r dist/ gs://twa-build-support/detectron_dist/$(date +"%Y_%m_%d")
    ```
1. Edit the url in the `thirdwave/requirements/bazel_requirements.in` file to point to the new path and re-generate the dependencies.
    ```bash
    /usr/local/bin/pip-compile-multi --no-upgrade --use-cache
    ```

# Steps to adding a new config variable
1. Edit [detectron2/config/defaults.py](detectron2/config/defaults.py) and add a new config variable.
1. Build and push a new wheel.

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).
* Models can be exported to torchscript format or caffe2 format for deployment.

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

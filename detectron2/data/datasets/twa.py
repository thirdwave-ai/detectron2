import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse TWA dataset directory into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_sem_seg", "convert_to_coco_json"]

def prep_merged_twa(merged_path):
    merged_description = read_proto(MergedDatasetDescriptionProto, merged_path)
    # Class names must be added in sorted order of the pixels they are labeled with.
    # TODO(katarina): better add class system.
    category_to_id_map = OrderedDict(
        sorted(dict(merged_description.class_name_to_id).items(), key=lambda kv: kv[1])
    )
    classes = ["__background__"] + list(category_to_id_map.keys())
    num_classes = len(classes)
    print("CLASSES:", classes)
    rgbd_pixel_mean = np.array(list(merged_description.meta_data.rgbd_mean))
    rgbd_pixel_stdev = np.array(list(merged_description.meta_data.rgbd_stdev))
    pixel_mean = rgbd_pixel_mean[[2, 1, 0, 3]] #RGBD -> BGRD
    pixel_stdev = rgbd_pixel_stdev[[2, 1, 0, 3]] #RGBD -> BGRD
    print("MEAN:", pixel_mean)
    print("STD:", pixel_stdev)
    return category_to_id_map, classes, num_classes, pixel_mean, pixel_stdev, merged_description

def prep_standard_twa(dataset_directory):
    dataset_description_path = os.path.join(dataset_directory, "protos", "dataset_description.pb")
    dataset_description = read_proto(DatasetDescriptionProto, dataset_description_path)
    category_to_id_map = OrderedDict()
    # Classes must be added in sorted order of the pixels they are labeled with.
    for class_id in sorted(dataset_description.class_id_to_name.keys()):
        category_to_id_map[dataset_description.class_id_to_name[class_id]] = class_id
    classes = ["__background__"] + list(category_to_id_map.keys())
    print("CLASSES:", classes)
    num_classes = len(classes)
    rgbd_pixel_mean = np.array(list(dataset_description.meta_data.rgbd_mean))
    rgbd_pixel_stdev = np.array(list(dataset_description.meta_data.rgbd_stdev))
    pixel_mean = rgbd_pixel_mean[[2, 1, 0, 3]] #RGBD -> BGRD
    pixel_stdev = rgbd_pixel_stdev[[2, 1, 0, 3]] #RGBD -> BGRD
    print("MEAN:", pixel_mean)
    print("STD:", pixel_stdev)
    return category_to_id_map, classes, num_classes, pixel_mean, pixel_stdev

def check_valid_label(path):
    datum_proto = read_proto(DatumProto, path)
    _, instance_mask, height, width = get_masks(datum_proto)
    unique_instances = np.unique(instance_mask)
    valid = len(unique_instances) > 1
    return valid, height, width

def get_masks(datum_proto):
    segmentation_mask = parse_image_field(datum_proto.segmentation_mask.png_data, cv2.IMREAD_GRAYSCALE)
    instance_mask = parse_image_field(datum_proto.instance_mask.png_data, cv2.IMREAD_ANYDEPTH)
    # check that segmentation mask and instance mask zero at same pixels
    assert np.array_equal(np.argwhere(segmentation_mask == 0), np.argwhere(instance_mask == 0))
    # check that segmentation mask and instance mask are non zero at same pixels
    assert np.array_equal(np.argwhere(segmentation_mask > 0), np.argwhere(instance_mask > 0))
    height_ins, width_ins = instance_mask.shape
    height_seg, width_seg = segmentation_mask.shape

    # check that height and width or segmentation and instance masks are equal
    assert (height_seg == height_ins) and (width_seg == width_ins)
    return segmentation_mask, instance_mask, height_seg, width_seg

def load_twa_directory(dataset_directory, subset="train"):
    """
    Given the full path to a directory with a twa dataset definition, create
    a detectron2 dataset.
    #DNS(katarina) find/create google doc and link it here.
    Args:
        dataset_directory (str): full path to the TWA dataset stored following the convention in the doc above.
        subset (str): one of ["train", "val", "test"] representing which subset of the dataset to create.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    potential_merged_path = os.path.join(dataset_directory, "merged_dataset_description.pb")
    if os.path.isfile(potential_merged_path):
        print("Using merged dataset.")
        merged_set = True
        category_to_id_map, classes, num_classes, pixel_mean, pixel_stdev, merged_description = prep_merged_twa(potential_merged_path)
    else:
        print("Using standard TWA dataset.")
        merged_set = False
        category_to_id_map, classes, num_classes, pixel_mean, pixel_stdev = prep_standard_twa(dataset_directory)
    potential_paths = sorted(glob.glob(os.path.join(dataset_directory, "protos", subset, "*.pb")))
    dataset_dicts = []
    for path in tqdm.tqdm(potential_paths):
        valid, height, width = check_valid_label(path)
        if valid:
            record = {}
            record["file_name"] = path
            record["height"] = height
            record["width"] = width
            dataset_dicts.append(record)
    print(f"Loaded {len(dataset_dicts)} images.")
    return dataset_dicts



if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)

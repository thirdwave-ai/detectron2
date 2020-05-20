import copy
import cv2
from collections import Counter, OrderedDict
import glob
import numpy as np
import os
import torch
import torch.utils.data
import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from third_party.maskrcnn_benchmark.maskrcnn_benchmark.structures.bounding_box import BoxList
from third_party.maskrcnn_benchmark.maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from third_party.maskrcnn_benchmark.maskrcnn_benchmark.utils import cv2_util

from twa.pallet_detection.datum_pb2 import DatasetDescriptionProto, DatumProto, MergedDatasetDescriptionProto
from twa.pallet_detection.proto_python_tools import (
    read_proto,
    parse_bgrd_image,
    parse_image_field,
    parse_grayscale_image,
)


def parse_image_mask_instance(datum_proto):
    """Reads in the triplet of the image, segmentation mask, and instance.

    Reads all images from datum and validates image dimensions.
    Returns:
        segmentation_mask: np.array[height, width]
        instance_mask: np.array[height, width]
        height: height of image/segmentation mask/instance mask
        width: width of image/segmentation mask/instance mask
    Raises:
        Asserts if there was a size mismatch or inconsistency in masks.
    """
    # Read in the three images.
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


class TwaDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, subset, transforms=None, depth=False, grayscale=False):
        # as you would do normally
        self.proto_directory = base_dir
        self.subset = subset
        self.use_depth = depth
        self.transforms = transforms
        self.grayscale = grayscale
        # Set up dataset classes
        potential_merged_path = os.path.join(self.proto_directory, "merged_dataset_description.pb")
        if os.path.isfile(potential_merged_path):
            print("Using merged dataset.")
            self.merged_set = True
            self._prep_merged_twa(potential_merged_path)
        else:
            print("Using standard TWA dataset.")
            self.merged_set = False
            self._prep_standard_twa()
        self.potential_paths = sorted(glob.glob(os.path.join(self.proto_directory, "protos", self.subset, "*.pb")))
        self.paths = []
        self.image_sizes = []
        print("Preparing dataset.")
        for path in tqdm.tqdm(self.potential_paths):
            valid, height, width = self.check_valid_label(path)
            if valid:
                self.paths.append(path)
                self.image_sizes.append({"height": height, "width": width})
        print("{} images used.".format(len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        datum_path = self.paths[idx]
        datum_proto = read_proto(DatumProto, datum_path)
        if self.grayscale:
            image, _ = parse_grayscale_image(datum_proto, add_depth=self.use_depth)
        else:
            image = parse_bgrd_image(datum_proto, add_depth=self.use_depth)
        image = image.astype(np.uint16)
        target = self.get_ground_truth(datum_proto, idx, self.image_sizes[idx])
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target, idx

    def check_valid_label(self, path):
        datum_proto = read_proto(DatumProto, path)
        _, instance_mask, height, width = self.get_masks(datum_proto)
        unique_instances = np.unique(instance_mask)
        valid = len(unique_instances) > 1
        return valid, height, width

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return self.image_sizes[idx]

    def get_masks(self, datum_proto):
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

    def get_ground_truth(self, datum_proto, idx, img_size):
        segmentation_mask, instance_mask, height, width = self.get_masks(datum_proto)
        boxes = []
        masks = []
        classes = []
        for instance_id in np.unique(instance_mask):
            if instance_id == 0:
                continue
            mask = (instance_mask == instance_id).astype(np.uint8)
            path = self.paths[idx]
            semantic_class = self._get_semantic_class(
                instance_mask, segmentation_mask, instance_id, datum_relative_path=path[len(self.proto_directory) :]
            )
            if not semantic_class:
                print("No semantic segmentation for this instance. Label is likely wrong.")
                continue
            contour, _ = cv2_util.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            polygons = [c.reshape(-1).tolist() for c in contour]
            if polygons == []:
                continue  # skip non-instance categories
            len_p = [len(p) for p in polygons]
            if min(len_p) <= 4:
                continue  # polygon must have at least 4 points.
            classes.append(semantic_class)
            masks.append(polygons)
            indices = np.where(mask == 1)
            x1, y1, x2, y2 = min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])
            boxes.append([x1, y1, x2, y2])
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (img_size["width"], img_size["height"]), mode="xyxy")
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        new_masks = SegmentationMask(masks, (img_size["width"], img_size["height"]))
        target.add_field("masks", new_masks)
        target = target.clip_to_image(remove_empty=True)
        return target

    def _get_semantic_class(self, instance_mask, segmentation_mask, id_num, datum_relative_path=None):
        """
        Returns a list of up to two semantic pixels occuring where the instance mask
        shows the given id number in sorted order, excluding 0, the
        background pixel. If there are no semantic pixels other than zero,
        returns an empty list.
        """
        segmentation_id_options = np.unique(segmentation_mask[instance_mask == id_num])
        assert len(segmentation_id_options) == 1
        segmentation_id = segmentation_id_options[0]
        if segmentation_id == 0:
            print("No segmentation. Something is not right!!")
        if self.merged_set:
            dataset_description = self.merged_description.dataset_descriptions[
                self.merged_description.datum_to_description[datum_relative_path]
            ]
            class_name = dataset_description.class_id_to_name[segmentation_id]
            new_pixel = self.merged_description.class_name_to_id[class_name]
            return new_pixel
        return segmentation_id

    def _prep_merged_twa(self, merged_path):
        self.merged_description = read_proto(MergedDatasetDescriptionProto, merged_path)
        self.source = "merged"
        # Class names must be added in sorted order of the pixels they are labeled with.
        # TODO(katarina/james): better add class system.
        self.category_to_id_map = OrderedDict(
            sorted(dict(self.merged_description.class_name_to_id).items(), key=lambda kv: kv[1])
        )
        self.classes = ["__background__"] + list(self.category_to_id_map.keys())
        self.num_classes = len(self.classes)
        print(self.classes)
        rgbd_pixel_mean = np.array(list(self.merged_description.meta_data.rgbd_mean))
        rgbd_pixel_stdev = np.array(list(self.merged_description.meta_data.rgbd_stdev))
        self.pixel_mean = rgbd_pixel_mean[[2, 1, 0, 3]]
        self.pixel_stdev = rgbd_pixel_stdev[[2, 1, 0, 3]]
        print(self.pixel_mean)
        print(self.pixel_stdev)

    def _prep_standard_twa(self):
        dataset_description_path = os.path.join(self.proto_directory, "protos", "dataset_description.pb")
        self.dataset_description = read_proto(DatasetDescriptionProto, dataset_description_path)
        self.source = self.dataset_description.name
        self.category_to_id_map = OrderedDict()
        # Classes must be added in sorted order of the pixels they are labeled with.
        for class_id in sorted(self.dataset_description.class_id_to_name.keys()):
            self.category_to_id_map[self.dataset_description.class_id_to_name[class_id]] = class_id
        self.classes = ["__background__"] + list(self.category_to_id_map.keys())
        print(self.classes)
        # We add 1 to include the background class.
        self.num_classes = len(self.classes)
        rgbd_pixel_mean = np.array(list(self.dataset_description.meta_data.rgbd_mean))
        rgbd_pixel_stdev = np.array(list(self.dataset_description.meta_data.rgbd_stdev))
        self.pixel_mean = rgbd_pixel_mean[[2, 1, 0, 3]]
        self.pixel_stdev = rgbd_pixel_stdev[[2, 1, 0, 3]]

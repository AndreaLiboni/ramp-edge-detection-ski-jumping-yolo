import os
import xml.etree.ElementTree as ET
import shutil
import random
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms


def parse_cvat_annotations(file_path, cvat_images_dir, output_root, augmentation_level):
    """
    Parse CVAT annotations in XML format to extract bounding boxes and image dimensions.

    Args:
        file_path (str): Path to the CVAT XML file.

    Returns:
        dict: A dictionary with keys as image names and values as annotation data.
    """
    # Create output directory structure
    labels_dir = os.path.join(output_root, "labels")
    images_dir = os.path.join(output_root, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    directorys = {
        "Train": [
            os.path.join(images_dir, "train"),
            os.path.join(labels_dir, "train")
        ],
        "Validation": [
            os.path.join(images_dir, "val"),
            os.path.join(labels_dir, "val")
        ],
        "Test": [
            os.path.join(images_dir, "test"),
            os.path.join(labels_dir, "test")
        ]
    }

    for _, (images_dir, labels_dir) in directorys.items():
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    # Parse the CVAT XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    image_idx = 1
    for image_tag in root.findall('image'):
        if len(image_tag.findall("polyline")) <= 0:
            continue
        image_name = image_tag.get('name')
        width = int(image_tag.get('width'))
        height = int(image_tag.get('height'))
        subset = image_tag.get('subset')
        boxes = []

        for polyline in image_tag.findall("polyline"):
            boxes.append(order_points([float(coord) for coord in polyline.get('points').replace(';', ',').split(',')]) + [width, height])
            break

        if subset == "Train":
            for _ in range(augmentation_level):
                image, line = augment_image(os.path.join(cvat_images_dir, subset, image_name), boxes[0].copy())
                image.save(os.path.join(directorys[subset][0], f"{image_idx:04d}.jpg"))
                save_line(line, directorys[subset][1], image_idx)
                image_idx += 1
            
        shutil.copy2(
            os.path.join(cvat_images_dir, subset, image_name),
            os.path.join(directorys[subset][0], f"{image_idx:04d}.jpg")
        )
        save_line(boxes[0], directorys[subset][1], image_idx)
        image_idx += 1
        


def save_line(line, path, idx):
    """
    Save bounding box annotations to YOLO .txt format.

    Args:
        annotations (dict): Parsed annotations.
        output_dir (str): Directory to save the label files.
        split_mapping (dict): A mapping of image names to their dataset split ('train' or 'test').
        category_id (int): Class ID to assign to all annotations (default is 0).
    """

    with open(os.path.join(path, f"{idx:04d}.txt"), "w") as f:
        x1, y1, x2, y2, width, height = line
        category_id = 0
        if y1 > y2:
            category_id = 1
        # Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height)
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        bbox_width = abs(x2 - x1) / width
        bbox_height = abs(y2 - y1) / height
        f.write(f"{category_id} " + " ".join(map(lambda x: f"{x:.6f}", (x_center, y_center, bbox_width, bbox_height))) + "\n")


def order_points(line):
    """
    If the line is not ordered from left to right, reorder the points.

    Args:
        line (list): List with coordinates of the line [x1, y1, x2, y2].
    Returns:
        list: Ordered line.
    """
    if line[0] > line[2]:
        line[0:4] = [line[2], line[3], line[0], line[1]]
        return line
    return line

def augment_image(image_path, line):
    """
    Augment an image by returning a new image with some data augmentation techniques.

    Args:
        image_path (str): Path to the image file.
        line (list): List with coordinates of the line [x1, y1, x2, y2].

    Returns:
        PIL.Image: Augmented image.
    """


    assert os.path.isfile(image_path), image_path
    image = Image.open(image_path).convert('RGB')

    img_width, img_height = image.size
    
    # random crop the image
    if random.random() < 0.5:
        crop_percent = random.uniform(0.1, 0.4)
        width_offset = int(img_width * crop_percent)
        height_offset = int(img_height * crop_percent)

        min_y, max_y = (line[1], line[3]) if line[1] < line[3] else (line[3], line[1])
        min_x, max_x = (line[0], line[2]) if line[0] < line[2] else (line[2], line[0])

        top = height_offset if height_offset < min_y else min_y
        left = width_offset if width_offset < min_x else min_x
        height = img_height - height_offset if img_height-height_offset > max_y else max_y
        width = img_width - width_offset if img_width-width_offset > max_x else max_x

        image = transforms.functional.crop(image, top, left, height, width)

        # adjust line coordinate after crop
        img_width, img_height = image.size
        line = [
            line[0] - left,
            line[1] - top,
            line[2] - left,
            line[3] - top,
            img_width,
            img_height
        ]
    
    # random color jittering
    if random.random() < 0.5:
        image = transforms.ColorJitter(
            brightness=(0.3, 1.7),
            contrast=(0.3, 1.7),
            saturation=(0.3, 1.7)
        )(image)
        
    # random horizontal filp the image
    if random.random() < 0.5:
        image = transforms.functional.hflip(image)
        line[0] = img_width - line[0]
        line[2] = img_width - line[2]
    
    # random blur the image
    if random.random() < 0.5:
        random_kernel_size = random.choice([7, 11, 17])
        image = transforms.functional.gaussian_blur(image, kernel_size=random_kernel_size)
    
    return image, order_points(line)


if __name__ == "__main__":
    parser = ArgumentParser(description='Dataset Conversion from CVAT to YOLO format')
    parser.add_argument('--root', type=str, required=True, help='root folder for the CVAT dataset')
    parser.add_argument('--output', type=str, default='./datasets', help='output directory for the YOLO dataset')
    parser.add_argument('--aug_level', type=int, default=1, help='augmentation level for the train dataset (default: 1)')
    args = parser.parse_args()

    # Output paths
    os.makedirs(args.output, exist_ok=True)
    augmentation_level = args.aug_level - 1

    # Parse annotations and organize dataset
    parse_cvat_annotations(
        file_path=os.path.join(args.root, "annotations.xml"),
        cvat_images_dir=os.path.join(args.root, "images"),
        output_root=args.output,
        augmentation_level=augmentation_level
    )

    # create the data.yaml file
    with open(os.path.join(args.output, "data.yaml"), "w") as f:
        f.write('path: ./ \ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n    0: ramp-edge-right\n    1: ramp-edge-left\n')

    print(f"YOLO dataset created at {args.output}")

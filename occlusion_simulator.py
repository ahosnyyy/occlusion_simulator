import cv2
import numpy as np
import random

class OcclusionSimulator:
    def __init__(self, bbox_file, img_width, img_height ,value=0, degree='partial'):
        self.bboxes = self._read_bboxes(bbox_file, img_width, img_height)
        self.value = value
        self.degree = degree
        self.ratios = [(0.23, 0.7), (0.36, 0.5), (0.23, 0.7), (0.23, 0.7)]

    def _read_bboxes(self, bbox_file, img_width, img_height):
        with open(bbox_file, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            # Parse the normalized bounding box
            _, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert to pixel coordinates
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            # Append the bounding box to the list
            bboxes.append((x, y, w, h))
        return bboxes

    def _calculate_box(self, x, y, w, h, height_ratio, width_ratio, index):
        box_y = y + sum(ratio[0] * h for ratio in self.ratios[:index])
        box_y -= 0.05 * h if index >= 2 else 0  # Shift up for bottom boxes
        box_h = height_ratio * h
        box_w = width_ratio * w
        box_x = x + int(0.15 * w) if index == 0 or index >= 2 else x  # Shift right for top and bottom boxes
        return (int(box_x), int(box_y), int(box_w), int(box_h))

    def divide_bbox(self):
        boxes = []
        for bbox in self.bboxes:
            x, y, w, h = bbox
            bbox_boxes = [self._calculate_box(x, y, w, h, height_ratio, width_ratio, i) for i, (height_ratio, width_ratio) in enumerate(self.ratios)]
            # Add an additional box beside the second box with the same dimensions
            if len(bbox_boxes) > 1:
                bbox_boxes.insert(2, (bbox_boxes[1][0] + bbox_boxes[1][2], bbox_boxes[1][1], bbox_boxes[1][2], bbox_boxes[1][3]))
            boxes.extend(bbox_boxes)
        return boxes

    @staticmethod
    def draw_boxes(image, boxes, ratios):
        for i, box in enumerate(boxes):
            x, y, w, h = map(int, box)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
    
    def erase_random_box(self, image, value=0, degree='partial'):
        for bbox in self.bboxes:
            self.bbox = bbox
            boxes = self.divide_bbox()
            
            # Determine the number of boxes to erase based on the degree of occlusion
            if degree == 'partial':
                num_boxes_to_erase = min(random.randint(1, 2), len(boxes))
            elif degree == 'heavy':
                num_boxes_to_erase = min(random.randint(3, 4), len(boxes))
            else:
                raise ValueError("Invalid value for 'degree'. It should be 'partial' or 'heavy'.")

            for _ in range(num_boxes_to_erase):
                box_to_erase = random.choice(boxes)
                x, y, w, h = box_to_erase

                if isinstance(value, int):
                    erase_value = value
                elif isinstance(value, tuple) and len(value) == 3:
                    erase_value = [float(v) for v in value]
                elif isinstance(value, str) and value.lower() == 'random':
                    erase_value = lambda: np.random.randint(256, size=(h, w, 3))  # Random color for each pixel
                else:
                    raise ValueError("Invalid value for 'value'. It should be an int, a tuple of length " f"{image.shape[-1]} (number of input channels), or 'random'.")

                if callable(erase_value):
                    image[y:y+h, x:x+w] = erase_value()  # Call the function to get the random values
                else:
                    image[y:y+h, x:x+w] = erase_value  # Set the pixel values in the box area to the erase value

                boxes.remove(box_to_erase)  # Remove the erased box from the list

        return image




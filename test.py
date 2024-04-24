import cv2
import numpy as np
from occlusion_simulator import OcclusionSimulator

def test_occlusion_simulator():
    # Define a bounding box (x, y, width, height)
    bbox = (50, 50, 200, 200)

    # Create an OcclusionSimulator object
    simulator = OcclusionSimulator(bbox)

    # Divide the bounding box
    boxes = simulator.divide_bbox()
    print(boxes)

    # Revised expected boxes
    revised_expected_boxes = [(80, 50, 140, 46), (50, 96, 100, 72), (150, 96, 100, 72), (80, 158, 140, 46), (80, 204, 140, 46)]

    # Check if the calculated boxes match the revised expected boxes
    assert boxes == revised_expected_boxes, f"Expected {revised_expected_boxes}, but got {boxes}"

    print("All tests passed!")

if __name__ == "__main__":
    test_occlusion_simulator()
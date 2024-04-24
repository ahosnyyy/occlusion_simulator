import cv2
import argparse
import ast
import os
from occlusion_simulator import OcclusionSimulator

def main():
    parser = argparse.ArgumentParser(description='Simulate occlusions on an image.')
    parser.add_argument('--value', default=0, help='Erasing value. Can be an int, a tuple of length 3, or "random".')
    parser.add_argument('--degree', default='partial', help='Degree of occlusion. Can be "partial" or "heavy".')
    parser.add_argument('--bbox_file', required=True, help='File containing bounding boxes.')
    parser.add_argument('--image', required=True, help='Path to the image file.')
    args = parser.parse_args()

    if args.value.lower() == 'random':
        value = 'random'
    elif '(' in args.value and ')' in args.value:
        value = ast.literal_eval(args.value)
        value = tuple(map(float, value))
    else:
        value = float(args.value)

    image = cv2.imread(args.image)
    height, width, _ = image.shape
    simulator = OcclusionSimulator(args.bbox_file, img_width=width, img_height=height, value=value, degree=args.degree)
    boxes = simulator.divide_bbox()
    image_with_boxes = simulator.draw_boxes(image.copy(), boxes, simulator.ratios)
    image_with_box_erased = simulator.erase_random_box(image.copy(), value=value, degree=args.degree)

    # Create a directory for the results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the images with more descriptive names
    cv2.imwrite(os.path.join('results', f'{os.path.basename(args.image)}_with_boxes.jpg'), image_with_boxes)
    cv2.imwrite(os.path.join('results', f'{os.path.basename(args.image)}_with_box_erased.jpg'), image_with_box_erased)

if __name__ == "__main__":
    main()
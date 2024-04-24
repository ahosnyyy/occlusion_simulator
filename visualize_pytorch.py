import argparse
import cv2
import numpy as np
import os
from torchvision import transforms
from occlusion_simulator_pytorch import OcclusionSimulator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Simulate occlusions on an image.')
    parser.add_argument('--value', default='random', help='Erasing value. Can be an int, a tuple of length 3, or "random".')
    parser.add_argument('--degree', default='heavy', help='Degree of occlusion. Can be "partial" or "heavy".')
    parser.add_argument('--bbox_file', required=True, help='File containing bounding boxes.')
    parser.add_argument('--image', required=True, help='Path to the image file.')
    args = parser.parse_args()

    # Convert the value argument to the appropriate type
    if args.value.lower() == 'random':
        value = 'random'
    elif '(' in args.value and ')' in args.value:
        value = ast.literal_eval(args.value)
        value = tuple(map(float, value))
    else:
        value = float(args.value)

    # Read the image using OpenCV
    image = cv2.imread(args.image)

    # Get the dimensions of the image
    img_height, img_width = image.shape[:2]

    # Create an OcclusionSimulator instance
    simulator = OcclusionSimulator(args.bbox_file, img_width, img_height, value=value, degree=args.degree)

    # Create a sequence of transformations
    transform = transforms.Compose([
        simulator
    ])

    # Apply the occlusion simulation to the image
    occluded_image = transform(image)

    # Create a directory for the results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the occluded image with a descriptive name in the results directory
    cv2.imwrite(os.path.join('results', f'{os.path.basename(args.image)}_occluded.png'), np.array(occluded_image))

if __name__ == "__main__":
    main()
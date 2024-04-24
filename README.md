# Image Occlusion Simulator

This repository contains Python scripts simulate occlusions on pedestrians in images using bounding boxes. The scripts read bounding box information from a file, divide each bounding box into smaller boxes based on specified ratios, and then erase a random number of these boxes to simulate occlusion.

## Dependencies

The scripts require the following Python libraries:
- OpenCV
- torchvision (for the PyTorch version)

## Usage

To use these scripts, run them from the command line with the following arguments:

- `--value`: The erasing value. Can be an int, a tuple of length 3, or "random". Default is 0 for the first script and "random" for the PyTorch version.
- `--degree`: The degree of occlusion. Can be "partial" or "heavy". Default is "partial" for the first script and "heavy" for the PyTorch version.
- `--bbox_file`: The file containing bounding boxes. This argument is required.
- `--image`: The path to the image file. This argument is required.

For example:

```bash
python visualize.py --value random --degree heavy --bbox_file sample/sample.txt --image sample/sample.png
```
or apply the occlusion simulation as a transformation in a PyTorch pipeline:

```bash
python visualize_pytorch.py --value random --degree heavy --bbox_file sample/sample.txt --image sample/sample.png
```
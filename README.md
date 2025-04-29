# OMR Sheet Response Capture

## Overview

The Optical Mark Recognition (OMR) Sheet Response Capture project is a Python application that detects marked bubbles in scanned images of sheets. This project utilizes computer vision techniques to identify and extract the marked answers, making it useful for automated grading systems.

## Features

- **Image Resizing**: Resizes the input image to a fixed width and height for consistent processing.
- **Bubble Detection**: Detects bubbles using image processing techniques such as thresholding, morphological operations, and Hough Circle Transform.
- **Answer Extraction**: Extracts the marked answers based on the intensity and black pixel count of the detected bubbles.
- **Confidence Scoring**: Calculates a confidence score for each detected answer to ensure accuracy.
- **Visualization**: Displays the final image with detected answers highlighted.

## Requirements

- Python 3.9 and above
- OpenCV (`cv2`)
- NumPy

## Installation

1. Clone the repository:
    ```sh
    https://github.com/SHU8HAM-K/OMR-Sheet-Response-Capture.git
    cd OMR-Sheet-Response-Capture
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your image file in the project directory or provide the path to the image.
2. Run the main script:
    ```sh
    python omr_sheet_marked_bubble_detect.py
    ```
3. The script will process the image, detect the marked bubbles, and display the final image with detected answers highlighted.

## Code Structure

- `omr_sheet_marked_bubble_detect.py`: The main script that runs the bubble detection process.
- `requirements.txt`: List of required Python packages.
- `Input Image`: Sample image
- `Output Result`: Output result

## Functions

### `resize_image(image, fixed_width=200, fixed_height=900)`

Resizes the input image to a fixed width and height.

- **Parameters**:
  - `image` (numpy.ndarray): The input image to resize.
  - `fixed_width` (int): The width to resize the image to.
  - `fixed_height` (int): The height to resize the image to.
- **Returns**:
  - `numpy.ndarray`: The resized image.

### `detect_marked_bubbles(image_path, confidence_threshold=50)`

Detects marked bubbles in an image and returns the detected answers.

- **Parameters**:
  - `image_path` (str): The path to the image file.
  - `confidence_threshold` (int): The threshold for confidence score to consider a bubble as marked.
- **Returns**:
  - `dict`: A dictionary containing the question numbers as keys and the detected answers as values.

### `main(image_path)`

Main function to detect marked bubbles in an image.

- **Parameters**:
  - `image_path` (str): The path to the image file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Inspired by automated grading systems.
- Built using OpenCV and NumPy.

## Contact

For any questions or feedback, please contact [7398259289sk@gmail.com]


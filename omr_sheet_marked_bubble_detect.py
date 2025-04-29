import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, fixed_width=200, fixed_height=900):
    """
    Resize the image to a fixed width and height.

    Parameters:
    image (numpy.ndarray): The input image to resize.
    fixed_width (int): The width to resize the image to.
    fixed_height (int): The height to resize the image to.

    Returns:
    numpy.ndarray: The resized image.
    """
    return cv2.resize(image, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)

def detect_marked_bubbles(image_path, confidence_threshold=50):
    """
    Detect marked bubbles in an image and return the detected answers.

    Parameters:
    image_path (str): The path to the image file.
    confidence_threshold (int): The threshold for confidence score to consider a bubble as marked.

    Returns:
    dict: A dictionary containing the question numbers as keys and the detected answers as values.
    """
    image_main = cv2.imread(image_path)
    if image_main is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    image = resize_image(image_main)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 135, 2)

    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, kernel, iterations=2)

    morph_blur = cv2.GaussianBlur(morph, (5, 5), 3)
    edges_m = cv2.Canny(morph_blur, 20, 150)

    circles = cv2.HoughCircles(morph_blur, cv2.HOUGH_GRADIENT, 1.2, 10, param1=50, param2=25, minRadius=10, maxRadius=26)

    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0, :]).astype("int")

    morph_visual = morph.copy()

    detected_answers = {}
    choices = ['a', 'b', 'c', 'd']

    if circles is not None:
        detected_circles = circles.tolist()
        sorted_circles = sorted(detected_circles, key=lambda x: x[1])
        rows = []

        current_row = [sorted_circles[0]]
        for i in range(1, len(sorted_circles)):
            if abs(sorted_circles[i][1] - current_row[-1][1]) < 30:
                current_row.append(sorted_circles[i])
            else:
                rows.append(current_row)
                current_row = [sorted_circles[i]]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x[0])

        for question_num, row in enumerate(rows, start=1):
            if len(row) != 4:
                logging.warning(f"Skipping question {question_num}, incorrect number of bubbles detected {len(row)}")
                continue

            best_choice = None
            best_mean_intensity = float("inf")
            best_black_pixels = float("inf")

            mean_intensities = []
            black_pixel_counts = []

            for idx, (x, y, r) in enumerate(row):
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)

                black_pixels = cv2.countNonZero(cv2.bitwise_and(morph, morph, mask=mask))
                mean_intensity = cv2.mean(morph, mask=mask)[0]
                mean_intensities.append(mean_intensity)
                black_pixel_counts.append(black_pixels)

                logging.debug(f"Question {question_num}, Choice {choices[idx]}: Black Pixels = {black_pixels}, Mean Intensity = {mean_intensity}")

                if mean_intensity < best_mean_intensity or \
                   (mean_intensity == best_mean_intensity and black_pixels < best_black_pixels):
                    best_choice = choices[idx]
                    best_mean_intensity = mean_intensity
                    best_black_pixels = black_pixels

            if best_choice:
                second_best_mean_intensity = min(intensity for choice_idx, intensity in enumerate(mean_intensities) if choices[choice_idx] != best_choice)
                relative_intensity_difference = (second_best_mean_intensity - best_mean_intensity) / second_best_mean_intensity
                black_pixel_difference = (max(black_pixel_counts) - black_pixel_counts[choices.index(best_choice)]) / max(black_pixel_counts)

                confidence_score = (relative_intensity_difference * 100) + (black_pixel_difference * 50)
                logging.info(f"Question {question_num}, Best Choice: {best_choice}, Confidence Score: {confidence_score}")

                if confidence_score > confidence_threshold:
                    detected_answers[question_num] = best_choice
                    cv2.putText(image, best_choice.upper(), (row[0][0] - 30, row[0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    logging.info(f"Detected answer for question {question_num}: {best_choice}")
                else:
                    logging.warning(f"No valid answer detected for question {question_num} due to low confidence.")
            else:
                logging.warning(f"No valid answer detected for question {question_num}")

        if not detected_answers:
            logging.warning("No bubbles were marked. Stopping detection process.")
            return {}

        logging.info("Answers detected successfully!")

    else:
        logging.error("No bubbles detected. Check preprocessing or parameter tuning.")
        return {}

    for (x, y, r) in circles:
        cv2.circle(morph_visual, (x, y), r, (0, 0, 0), 4)
        cv2.circle(morph_visual, (x, y), 1, (0, 0, 0), 4)

    cv2.imshow("Final Image with Detected Answers", morph_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_answers

def main(image_path):
    """
    Main function to detect marked bubbles in an image.

    Parameters:
    image_path (str): The path to the image file.
    """
    try:
        detected_answers = detect_marked_bubbles(image_path)
        logging.info(f"Detected answers: {detected_answers}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = "path_to_your_image"  # Replace with the actual path to your image
    main(image_path)

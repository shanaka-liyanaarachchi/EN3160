import cv2
import numpy as np

def get_four_points(image):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Four Points", image)
    cv2.imshow("Select Four Points", image)
    cv2.setMouseCallback("Select Four Points", mouse_callback)
    while len(points) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)

def main():
    # Load architectural image and flag image
    architectural_image = cv2.imread("Arcade_Square.jpg")
    flag_image = cv2.imread("flag.jfif", cv2.IMREAD_UNCHANGED)

    # Get four points from the user in the architectural image
    points_architectural = get_four_points(architectural_image.copy())

    # Define four points in the flag image (assuming it's a rectangle)
    points_flag = np.array([[0, 0], [0, flag_image.shape[0]], [flag_image.shape[1], flag_image.shape[0]], [flag_image.shape[1], 0]], dtype=np.float32)

    # Compute homography matrix
    homography_matrix, _ = cv2.findHomography(points_flag, points_architectural)

    # Warp the flag image onto the architectural image
    warped_flag = cv2.warpPerspective(flag_image, homography_matrix, (architectural_image.shape[1], architectural_image.shape[0]))

    # Blend the warped flag onto the architectural image
    alpha = 0.4
    blended_image = cv2.addWeighted(architectural_image, 1 - alpha, warped_flag[:, :, :3], alpha, 0)

    # Display the result
    cv2.imshow("Blended Image", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

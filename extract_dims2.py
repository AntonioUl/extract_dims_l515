import cv2
import numpy as np
from functools import partial
import pyrealsense2 as rs

# Calibration variables
scale_factor_x = 0.35
scale_factor_y = 0.35

def mouse_callback(event, x, y, flags, params):
    color_image, depth_frame, depth_intrinsics = params

    if event == cv2.EVENT_LBUTTONDOWN:
        depth_mm = get_distance(x, y, depth_frame)
        print(f"Depth at ({x}, {y}): {depth_mm:.2f} mm")


def get_distance(x, y, depth_frame):
    depth = depth_frame.get_distance(x, y)
    return depth * 1000  # Convert to millimeters

# Initialize camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Get depth sensor and depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get depth intrinsics
depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

# Define the size of the region of interest
size = 100

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        print("Received depth frame.")

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
 
        # Set mouse callback to display depth information when clicking
        cv2.namedWindow("Color Stream")
        cv2.setMouseCallback("Color Stream", mouse_callback, (color_image, depth_frame, depth_intrinsics))

        # Allow the user to select the region of interest
        roi = cv2.selectROI("Color Stream", color_image, False, False)
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])


        # Extract the region of interest from the color image
        roi = color_image[y:y + h, x:x + w]


        print("Region of interest selected.")

        # Convert the region of interest to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Detect edges using the Canny edge detection algorithm
        edges = cv2.Canny(blurred_roi, 50, 150)

        # Find contours in the binary image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("Detecting edges and extracting contours.")

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Get the length of the sides of the rectangle
            (x, y), (side_one, side_two), _ = rect

            # Calculate depth_x and depth_y
            depth_x = int(min(x + size // 2, depth_frame.get_width() - 1))
            depth_y = int(min(y + size // 2, depth_frame.get_height() - 1))

            # Convert the dimensions from pixels to millimeters
            depth_mm = get_distance(depth_x, depth_y, depth_frame)
            side_one_mm = side_one * depth_mm * scale_factor_x / depth_intrinsics.fx
            side_two_mm = side_two * depth_mm * scale_factor_y / depth_intrinsics.fy

            print(f"Dimensions calculated: Side 1 = {side_one_mm:.1f} mm, Side 2 = {side_two_mm:.1f} mm")

            # Draw the green contour and red rectangle
            cv2.drawContours(roi, [cnt], 0, (0, 255, 0), 2)
            cv2.drawContours(roi, [box], 0, (0, 0, 255), 2)

            # Display the dimensions on the color image
            cv2.putText(color_image, f"Side 1: {side_one_mm:.1f} mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(color_image, f"Side 2: {side_two_mm:.1f} mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print("Displaying color image with dimensions.")

        # Display the images
        cv2.imshow("Color Stream", color_image)

        # Create an empty image with the same dimensions as the color image
        edges_overlay = np.zeros_like(color_image)

        # Copy the edges (in blue) to the edges_overlay image
        edges_overlay[int(y):int(y + h), int(x):int(x + w), 0] = edges.astype(edges_overlay.dtype)

        # Combine the color_image and edges_overlay
        combined_image = cv2.addWeighted(color_image, 1, edges_overlay, 0.5, 0)

        # Display the combined image
        cv2.imshow("Color Stream with Edges Overlay", combined_image)


        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


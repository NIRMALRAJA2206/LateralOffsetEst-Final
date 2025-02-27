# Import necessary libraries
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import socket
import yaml
import struct

# Function to find the intersection of two lines
def find_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    # Calculate coefficients for line1 (from lane boundary)
    A1 = line1_p2[1] - line1_p1[1]
    B1 = line1_p1[0] - line1_p2[0]
    C1 = A1 * line1_p1[0] + B1 * line1_p1[1]

    # Calculate coefficients for line2 (vehicle reference line)
    A2 = line2_p2[1] - line2_p1[1]
    B2 = line2_p1[0] - line2_p2[0]
    C2 = A2 * line2_p1[0] + B2 * line2_p1[1]

    matrix_A = np.array([[A1, B1], [A2, B2]])
    matrix_C = np.array([C1, C2])

    if np.linalg.det(matrix_A) == 0:
        return None  # Lines are parallelwq

    intersection = np.linalg.solve(matrix_A, matrix_C)
    return (int(intersection[0]), int(intersection[1]))


# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# Separated function to process boundary points and calculate offset and direction
def find_offset_and_direction(all_boundary_points, img_width, y_threshold, x_limit):
    all_boundary_points_array = np.array(all_boundary_points)
    boundary_points_sorted = all_boundary_points_array[
        np.lexsort((all_boundary_points_array[:, 0], all_boundary_points_array[:, 1]))]

    filtered_points = boundary_points_sorted[boundary_points_sorted[:, 1] > y_threshold]

    unique_ys = np.unique(filtered_points[:, 1])
    first_valid_pair = None

    # Find the first valid pair of points
    for y in unique_ys:
        points_with_y = filtered_points[filtered_points[:, 1] == y]
        valid_points_with_y = points_with_y[points_with_y[:, 0] < x_limit]

        if valid_points_with_y.size > 0:
            max_x_point = valid_points_with_y[valid_points_with_y[:, 0].argmax()]

            if first_valid_pair is None:
                first_valid_pair = (max_x_point, y)
            else:
                break

    if first_valid_pair is not None:
        y_value = first_valid_pair[1]
        points_with_same_y = filtered_points[filtered_points[:, 1] == y_value]
        points_with_same_y_and_x_condition = points_with_same_y[points_with_same_y[:, 0] > 0]

        if points_with_same_y_and_x_condition.size > 0:
            point1 = points_with_same_y_and_x_condition[points_with_same_y_and_x_condition[:, 0].argmin()]
            point2 = first_valid_pair[0]
        else:
            point1, point2 = None, None
    else:
        point1, point2 = None, None

    # If valid points were found, compute the intersection and offset
    if point1 is not None and point2 is not None:
        # Vehicle reference line endpoints (pre-calibrated)
        CENTER_LINE = [tuple(config["calibration"]["center_line"][0]), tuple(config["calibration"]["center_line"][1])]
        LEFT_LINE = [tuple(config["calibration"]["left_line"][0]), tuple(config["calibration"]["left_line"][1])]
        RIGHT_LINE = [tuple(config["calibration"]["right_line"][0]), tuple(config["calibration"]["right_line"][1])]

        # Find intersections with all three vehicle reference lines
        center_intersect = find_intersection(point1, point2, *CENTER_LINE)
        left_intersect = find_intersection(point1, point2, *LEFT_LINE)
        right_intersect = find_intersection(point1, point2, *RIGHT_LINE)

        if None in [center_intersect, left_intersect, right_intersect]:
            return None, None, None, None, None

        line_center = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

        # Calculate distances from lane center to each intersection
        d_C = calculate_distance(center_intersect, line_center)
        d_L = calculate_distance(left_intersect, line_center)
        d_R = calculate_distance(right_intersect, line_center)

        # Calculate intra-vehicle distances
        d_h1_L = calculate_distance(left_intersect, center_intersect)
        d_h1_R = calculate_distance(right_intersect, center_intersect)

        # Calculate component offsets
        d_o_C = d_C
        d_o_L = d_L - d_h1_L
        d_o_R = d_R - d_h1_R

        # Average the three offsets
        offset_pix = (d_o_C + d_o_L + d_o_R) / 3

        # Convert to real-world distance
        lane_width_pix = calculate_distance(point1, point2)
        offset_m = (offset_pix / lane_width_pix) * 3.6

        # Determine direction
        direction = "left" if center_intersect[0] < line_center[0] else "right"

        return offset_m, direction, point1, point2, center_intersect


# Updated function to process each frame and analyze it using YOLO model
def process_frame(img, model):
    img = cv2.resize(img, (1280, 720))  # Resize the frame to 1280x720
    H, W, _ = img.shape

    results = model(img)  # Run YOLO model

    max_contour_length = 0
    max_contour = None
    segmentation_found = False

    # Iterate through model results and extract masks
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask_cpu = mask.cpu().numpy()
                mask_cpu = cv2.resize(mask_cpu, (W, H))
                contours, _ = cv2.findContours((mask_cpu * 255).astype('uint8'), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                for contour in contours:
                    if len(contour) > max_contour_length:
                        max_contour_length = len(contour)
                        max_contour = contour
                segmentation_found = True

    if not segmentation_found:
        return img, None, None  # Return original frame if no segmentation found
    # Create a red mask to overlay on the segmented area
    red_mask = np.zeros_like(img)
    if max_contour is not None:
        cv2.drawContours(red_mask, [max_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Overlay the red mask on the original image
    overlay_image = cv2.addWeighted(img, 1.0, red_mask, 0.5, 0)

    # Get boundary points of the largest contour
    all_boundary_points = []
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        for px in range(x, x + w):
            for py in range(y, y + h):
                if cv2.pointPolygonTest(max_contour, (px, py), False) == 0:
                    all_boundary_points.append((px, py))

    # Calculate offset and direction using boundary points for 3 set of lines
    #line 1
    offset_m_a, direction_a, point1_a, point2_a, centerCar_intersect_a = find_offset_and_direction(all_boundary_points, W,360,960)

    #line 2
    offset_m_b, direction_b, point1_b, point2_b, centerCar_intersect_b = find_offset_and_direction(all_boundary_points,W, 460, 960)

    #line 3
    offset_m_c, direction_c, point1_c, point2_c, centerCar_intersect_c = find_offset_and_direction(all_boundary_points,W, 500, 960)

    offset_m_values = [offset_m_a, offset_m_b, offset_m_c]
    valid_offsets = [offset for offset in offset_m_values if offset is not None]  # Handle None values

    if valid_offsets:
        offset_m_final = sum(valid_offsets) / len(valid_offsets)
    else:
        return img, None, None  # If no valid offsets, return None for all values

    # Step 2: Find the offset that is closest to the final averaged offset
    closest_offset = min(valid_offsets, key=lambda x: abs(x - offset_m_final))

    # Step 3: Select point1, point2, centerCar_intersect, and direction based on the closest offset
    if closest_offset == offset_m_a:
        point1_final, point2_final, centerCar_intersect_final, final_direction = point1_a, point2_a, centerCar_intersect_a, direction_a
    elif closest_offset == offset_m_b:
        point1_final, point2_final, centerCar_intersect_final, final_direction = point1_b, point2_b, centerCar_intersect_b, direction_b
    else:  # closest_offset == offset_m_c
        point1_final, point2_final, centerCar_intersect_final, final_direction = point1_c, point2_c, centerCar_intersect_c, direction_c


    offset_m, direction, point1, point2, centerCar_intersect = offset_m_final, final_direction, point1_final, point2_final, centerCar_intersect_final

    corrected_offset_m = offset_m - (E_SCALE * offset_m) + E_BIAS
    # Draw visual cues: contour, lines, intersection points
    if point1 is not None and point2 is not None:

        if centerCar_intersect is not None:
            # Draw circles at the intersection and midpoint
            #cv2.circle(overlay_image, centerCar_intersect, 5, (0, 255, 255), -1)
            line_center = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
            #cv2.circle(overlay_image, line_center, 5, (255, 255, 0), -1)

            # Annotate the frame with offset information
            cv2.putText(overlay_image, f"Offset: {corrected_offset_m:.4f} m towards {direction}", (460, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #Draw specified white lines (example lines, can be adjusted)
            cv2.fillPoly(overlay_image,
                         [np.array(((656, 450), (658, 450), (685, 560), (677, 560)), dtype=np.int32)],
                         (255, 255, 255))
            cv2.fillPoly(overlay_image,
                         [np.array(((614, 450), (616, 450), (570, 560), (560, 560)), dtype=np.int32)],
                         (255, 0, 0))
            cv2.fillPoly(overlay_image,
                         [np.array(((709, 450), (711, 450), (809, 560), (799, 560)), dtype=np.int32)],
                         (255, 0, 0))

        return overlay_image, corrected_offset_m, direction

    return overlay_image, 0, "right"


#Load config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load configurations
MODEL_PATHX = config["model"]["path"]
SOURCE_TYPE = config["input"]["type"]
VIDEO_PATHX = config["input"]["path"]
INPUT_SOURCE = config["input"]["source"]
UDP_IP = config["network"]["UDP_IP"]
UDP_PORT = config["network"]["UDP_PORT"]
ENABLE_VISUALIZATION = config["visualization"]["enable"]
# Load error correction parameters
E_SCALE = config["error_correction"]["e_scale"]
E_BIAS = config["error_correction"]["e_bias"]

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



# Initialize video capture
input_option = SOURCE_TYPE  # 1 for video file, 2 for camera feed
video_path = VIDEO_PATHX
cap = cv2.VideoCapture(video_path if input_option == 1 else INPUT_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model_path = MODEL_PATHX
model = YOLO(model_path).to(device)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = frame[:, frame.shape[1] // 2:]

    processed_frame, c_offset_m, direction = process_frame(frame, model)
    if c_offset_m is not None:
        # Convert direction to +1 or -1
        direction_value = 1 if direction == "right" else -1

        # Pack float (4 bytes) and signed char (1 byte) using struct
        message = struct.pack("!f b", c_offset_m, direction_value)

        # Send the structured message over UDP
        sock.sendto(message, (UDP_IP, UDP_PORT))

    if ENABLE_VISUALIZATION:
        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
if ENABLE_VISUALIZATION:
    cv2.destroyAllWindows()

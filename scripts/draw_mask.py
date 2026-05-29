import cv2
import numpy as np

START_LINES = [(984, 346), (961, 256), (1101, 163), (1027, 101), (881, 126), (684, 165), (499, 128), (433, 126),
               (306, 198), (72, 297), (2, 352), (0, 7), (1278, 4), (1273, 953), (1059, 957), (1218, 806), (831, 517)]

# List to store coordinates
points = list(START_LINES)
undo_stack = []
dragged_point_index = None
POINT_HIT_RADIUS = 12


def save_undo_state():
    undo_stack.append(list(points))


def undo():
    global points, dragged_point_index
    if not undo_stack:
        print("Nothing to undo.")
        return

    points = undo_stack.pop()
    dragged_point_index = None
    print("Undo.")


def find_nearest_point_index(x, y):
    if not points:
        return None

    distances = [
        (index, (px - x) ** 2 + (py - y) ** 2)
        for index, (px, py) in enumerate(points)
    ]
    nearest_index, nearest_distance = min(distances, key=lambda item: item[1])
    if nearest_distance <= POINT_HIT_RADIUS ** 2:
        return nearest_index

    return None

def draw_polygon(event, x, y, flags, param):
    global points, dragged_point_index

    if event == cv2.EVENT_LBUTTONDOWN:
        dragged_point_index = find_nearest_point_index(x, y)
        save_undo_state()
        if dragged_point_index is not None:
            points[dragged_point_index] = (x, y)
            print(f"Moving point {dragged_point_index}: ({x}, {y})")
        else:
            points.append((x, y))
            print(f"Point added: ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE and dragged_point_index is not None:
        points[dragged_point_index] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragged_point_index is not None:
            points[dragged_point_index] = (x, y)
            print(f"Point moved: ({x}, {y})")
        dragged_point_index = None

    # Right click to reset points if you mess up
    elif event == cv2.EVENT_RBUTTONDOWN:
        save_undo_state()
        points = list(START_LINES)
        dragged_point_index = None
        print("Resetting points to START_LINES.")


def main():
    global points
    video_path = r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video. Check the path.")
        return

    ret, img = cap.read()
    cap.release()

    if not ret or img is None:
        print("Error: Could not read the first frame from the video.")
        return

    cv2.namedWindow("Polygon Mask Drawer")
    cv2.setMouseCallback("Polygon Mask Drawer", draw_polygon)

    print("\n--- INSTRUCTIONS ---")
    print("1. Left-Click empty space to add points.")
    print("2. Drag existing points to move them.")
    print("3. Right-Click to reset points to START_LINES.")
    print("4. Ctrl+Z to undo.")
    print("5. Press 'q' to quit and print the final array.")
    print("---------------------\n")

    while True:
        display_img = img.copy()

        # Draw lines between points
        if len(points) > 0:
            # Draw dots at each click
            for index, pt in enumerate(points):
                color = (0, 0, 255) if index == dragged_point_index else (0, 255, 0)
                cv2.circle(display_img, pt, 5, color, -1)

            # Draw lines connecting the dots
            if len(points) > 1:
                cv2.polylines(display_img, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.imshow("Polygon Mask Drawer", display_img)

        key = cv2.waitKeyEx(1)
        key_code = key & 0xFF
        if key_code == ord('q'):
            break
        if key_code == 26:
            undo()

    # Final Output
    print("\nFinal Polygon Coordinates:")
    print(points)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np

def create_drawing_animation(image_path, output_video_path="drawing_animation.mp4", duration_sec=6, fps=30):
    # Load image[span_3](start_span)[span_3](end_span)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Resize image for smooth processing and display
    height, width = img.shape[:2]
    max_dim = 800
    if max(height, width) > max_dim:
        scale = max_dim / float(max(height, width))
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection to create white outline on black background
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Sort contours by length to draw main outlines first
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Canvas for accumulated drawing
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate total points across all contours
    total_points = sum(len(cnt) for cnt in contours)
    total_frames = duration_sec * fps
    points_per_frame = max(1, total_points // total_frames)

    current_point_count = 0

    print("Generating animation frames...")

    for contour in contours:
        for i in range(1, len(contour)):
            pt1 = tuple(contour[i - 1][0])
            pt2 = tuple(contour[i][0])

            # Draw white line segment on black background
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
            current_point_count += 1

            # Write frame periodically
            if current_point_count % points_per_frame == 0:
                out.write(canvas)
                cv2.imshow("Drawing Animation", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Hold the final completed image for 2 minutes
    for _ in range(fps * 2):
        out.write(canvas)

    out.release()
    cv2.destroyAllWindows()
    print(f"Animation saved successfully as {output_video_path}")

# Run the function using your image[span_4](start_span)[span_4](end_span)
create_drawing_animation("Rean1.jpg")

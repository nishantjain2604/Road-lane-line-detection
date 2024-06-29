import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1) Load image
img = cv2.imread(r"F:\M.tech\internship\road-lane-detection-master\Data\TonkRoad.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2) Preprocessing
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Step 3) Canny Edge Detection
canny = cv2.Canny(gray_img, 50, 150)

# Step 4) Define ROI Vertices
roi_vertices = np.array([[(100, img.shape[0]), (550, 250), (800, 250), (img.shape[1], img.shape[0])]], dtype=np.int32)

# Step 5) Define ROI function
def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

# Step 6) ROI Image
roi_image = roi(canny, roi_vertices)

# Step 7) Apply Hough Lines P Method on ROI Image
lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)

# Step 8) Draw Hough lines
def draw_lines(image, hough_lines):
    if hough_lines is not None:
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

final_img = draw_lines(np.copy(img), lines)  # Result

plt.imshow(final_img)
plt.xticks([])
plt.yticks([])
plt.show()

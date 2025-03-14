
import cv2
import os
import numpy as np  



# ffa72a rgb(255 167 42) bgr(42 167 255)
# lower_limits = (32, 157, 245)
lower_limits = (0, 0, 245)
upper_limits = (0, 0, 255)


img_annotated = cv2.imread(filename='image_annotated.jpg')
# img_annotated = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2Lab) # convert to Lab color space

original_image = cv2.imread(filename='image.jpg')
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab) # convert to Lab color space

mask = cv2.inRange(img_annotated,lower_limits, upper_limits)





mean, std = cv2.meanStdDev(original_image, mask = mask) # mean and standard deviation of the image in the mask where red color is present


# lower_limits = (int(mean[0] - std[0]), int(mean[1] - std[1]), int(mean[2] - std[2]))
# upper_limits = (int(mean[0] + std[0]), int(mean[1] + std[1]), int(mean[2] + std[2]))
mean = mean.flatten()  # Convert (3,1) to (3,)
std = std.flatten()    # Convert (3,1) to (3,)

lower_limits = (int(mean[0] - std[0]), int(mean[1] - std[1]), int(mean[2] - std[2]))
upper_limits = (int(mean[0] + std[0]), int(mean[1] + std[1]), int(mean[2] + std[2]))
print(f"Lower Limits: {lower_limits}")
print(f"Upper Limits: {upper_limits}")

original_image = cv2.imread("image.jpg")


mean_mask = cv2.inRange(original_image, lower_limits, upper_limits)
contours, _ = cv2.findContours(mean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) >= 20]

# cv2.drawContours(image=original_image, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                
for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(original_image, (cx, cy), 3, (0, 0, 255), 2)


cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Annotated Image", cv2.cvtColor(img_annotated, cv2.COLOR_Lab2BGR)
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 






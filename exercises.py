
import cv2
import os



# ffa72a rgb(255 167 42) bgr(42 167 255)
# lower_limits = (32, 157, 245)
lower_limits = (0, 0, 245)
upper_limits = (0, 0, 255)
img_annotated = cv2.imread(filename='image_annotated.jpg')



original_image = cv2.imread(filename='image.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab) # convert to Lab color space
mask = cv2.inRange(img_annotated,lower_limits, upper_limits)





mean, std = cv2.meanStdDev(original_image, mask = mask) # mean and standard deviation of the image in the mask where red color is present


# lower_limits = (int(mean[0] - std[0]), int(mean[1] - std[1]), int(mean[2] - std[2]))
# upper_limits = (int(mean[0] + std[0]), int(mean[1] + std[1]), int(mean[2] + std[2]))
mean = mean.flatten()  # Convert (3,1) to (3,)
std = std.flatten()    # Convert (3,1) to (3,)

lower_limits = (int(mean[0] - std[0]), int(mean[1] - std[1]), int(mean[2] - std[2]))
upper_limits = (int(mean[0] + std[0]), int(mean[1] + std[1]), int(mean[2] + std[2]))


number_of_contours = 0
directory = './Images for first miniproject'

for filename in os.listdir(directory):
    # print(f"Processing {filename}")
    if filename.endswith('.JPG'):
        image_path = os.path.join(directory, filename)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)  # convert to Lab color space
        
        
        segmented_image = cv2.inRange(original_image, lower_limits, upper_limits)
        contours, _ = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        number_of_contours += len(contours)
        # print(f"Number of contours found in {filename}: {len(contours)}")

#displ
# segmented_image = cv2.inRange(original_image,lower_limits, upper_limits)


# contours, _ = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {number_of_contours}")
# print(f"Number of contours found: {len(contours)}")


# cv2.drawContours(image=original_image, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                

# cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Original Image", original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 






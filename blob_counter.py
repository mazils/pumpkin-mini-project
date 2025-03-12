import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
input_rel = "../figures"
annotated_rel = "../annotated-images"  
output_rel = "../processed-images"
input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)
annotated_path = os.path.join(script_dir, annotated_rel)


def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)


def main():
    # filename = "./Gyldensteensvej-9-19-2017-orthophoto.tif"
    # filename =  input_path + "/EB-02-660_0595_0435.JPG"
    # img = cv2.imread(filename)
    



    # Convert to HSV
    image = input_path + "/EB-02-660_0595_0435.JPG"
    image_annoted = annotated_path + "/EB-02-660_0595_0435.png"  
    # Load images
    image = cv2.imread(image)
    dst = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite("./ex05-1-smoothed.jpg", dst)
    image_annoted = cv2.imread(image_annoted)

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
    image_annoted = cv2.cvtColor(image_annoted, cv2.COLOR_BGR2RGB).astype("float")
    
        # Find annotated color
    tmp0 = image_annoted[:, :, 0] == 255
    tmp1 = image_annoted[:, :, 1] == 0
    tmp2 = image_annoted[:, :, 2] == 0
    annotated_pixels = tmp0 & tmp1 & tmp2

    # determine color statistics
    pumkin_colors = image[annotated_pixels]
    refColorMean= pumkin_colors.mean(axis=0)
    refColorCov= np.cov(pumkin_colors.T)

    # Hendriks math magic for mahalanobis distance
    diff = image - refColorMean
    inv_cov = np.linalg.inv(refColorCov)
    moddotproduct = diff * (diff @ inv_cov)
    mahalanobis_dist = np.sum(moddotproduct, axis=2)

    # Thresholding
    pumkings = np.zeros(image.shape[0:2])
    pumkings[mahalanobis_dist < 20] = 1
    segmented_image = pumkings.astype("uint8")*255
    cv2.imwrite("./ex05-2-segmented.jpg", segmented_image)
    
    # Morphological filtering the image
    kernel = np.ones((20, 20), np.uint8)
    closed_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("./ex05-3-closed.jpg", closed_image)

    # Locate contours.
    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    image = image.astype("uint8")*255
    # Draw a circle above the center of each of the detected contours.
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), 2)
        else:
            # Handle the case where m00 is zero if necessary
            print("Contour with zero area detected, skipping.")

    print("Number of detected balls: %d" % len(contours))

    cv2.imwrite("./ex05-4-located-objects.jpg", image)



main()
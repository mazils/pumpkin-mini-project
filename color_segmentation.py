import cv2
import numpy as np
from file_path import input_path, annotated_path, output_path
class PumpkinCounter():
    def __init__(self,verbose=False):
        image,image_anotated=self._LoadReferenceImage()
        self._FindReferenceColorStats(image,image_anotated)
        self._verbose=verbose
        if self._verbose:
            print("Finish Loading Annotations")

    def ProcessImage(self, image: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> int:
        binary_Image = self._SegmentColors(image)
        if self._verbose:
            print("Finish Segmentation")
            cv2.imwrite("binary_image.png", binary_Image.astype("uint8")*255)

        numBlobs, blobMap = self._FindBlobs(binary_Image)
        if self._verbose:
            print("Finish Finding Blobs")
            print("Number of Blobs: ", numBlobs)
            cv2.imwrite("blob_map.png", blobMap.astype("uint8")*255)    

        blobCounts, Blobs = self._ExtractBlobList(numBlobs, blobMap)
        if self._verbose:
            print("Finish Processing Blobs")
            # np.savetxt("blob_counts.csv", blobCounts, delimiter=",")
            # np.savetxt("blob_map.csv", blobMap, delimiter=",")

        blobCounts = self._ProcessBlobCounts(blobCounts)
        print("blobCounts: ", len(blobCounts))
        blobCount = self._ProcessBlobList(blobCounts, Blobs, mins, maxs)
        if self._verbose:
            print("Finish Counting")

        return int(blobCount)
    
    def processImageContours(self,image:np.ndarray) -> int:
        binary_Image = self._SegmentColors(image)
        closed_image=self.__filterMorphological(binary_Image)
        contours=self.__locateContours(closed_image)
        annotated_image=self.__annotateImage(image,contours)
        # annotated_image=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        if self._verbose:
            print("Image Annotated")
        return annotated_image, len(contours)
        
        
        
        
    def _LoadReferenceImage(self):
        # image_name = "EB-02-660_0595_0435"
        # image = "Data/Images/" + image_name + ".JPG"
        # image_annoted = "Data/Images_annotated/" + image_name + ".png"

        image = input_path + "/EB-02-660_0595_0435.JPG"
        image_annoted = annotated_path + "/EB-02-660_0595_0435.png"  
        # Load images
        image = cv2.imread(image)
        image_annoted = cv2.imread(image_annoted)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
        image_annoted = cv2.cvtColor(image_annoted, cv2.COLOR_BGR2RGB).astype("float")
        return image, image_annoted


    def _FindReferenceColorStats(self,image, image_annoted):
        # Find annotated color
        tmp0 = image_annoted[:, :, 0] == 255
        tmp1 = image_annoted[:, :, 1] == 0
        tmp2 = image_annoted[:, :, 2] == 0
        annotated_pixels = tmp0 & tmp1 & tmp2

        # determine color statistics
        pumkin_colors = image[annotated_pixels]
        self.refColorMean= pumkin_colors.mean(axis=0)
        self.refColorCov= np.cov(pumkin_colors.T)


    def _SegmentColors(self,image,threshold=20):
        # Hendriks math magic for mahalanobis distance
        diff = image - self.refColorMean
        inv_cov = np.linalg.inv(self.refColorCov)
        moddotproduct = diff * (diff @ inv_cov)
        mahalanobis_dist = np.sum(moddotproduct, axis=2)

        # Thresholding
        pumkings = np.zeros(image.shape[0:2])
        pumkings[mahalanobis_dist < threshold] = 1
        return pumkings


    def _FindBlobs(self,binary_image):
        num_labels, labels_im = cv2.connectedComponents(binary_image.astype("uint8"))
        return num_labels, labels_im
    

    def _ExtractBlobList(self,numlabels, labels_im):
        img_shape = labels_im.shape
        BlobCounts = np.zeros(numlabels)
        Blobs=dict()
        for x in range(img_shape[0]):
            for y in range(img_shape[1]):
                if labels_im[x, y] == 0:
                    continue

                blob_label = labels_im[x, y]
                if blob_label in Blobs.keys():
                    Blobs[blob_label].append(np.array([x, y]))
                else:
                    Blobs[blob_label] = [np.array([x, y])]
                BlobCounts[blob_label] = BlobCounts[blob_label] + 1

        return BlobCounts,Blobs
    
    def _ProcessBlobCounts(self,blobCounts):
        # filter out small blobs
        blobCounts[blobCounts < 20] = 0
        blobCounts[np.logical_and(blobCounts > 0, blobCounts < 100)] = 1
        blobCounts[blobCounts > 100] = (blobCounts[blobCounts > 100] / 100).astype(np.int16)
        return blobCounts
    
    def _ProcessBlobList(self,blobCounts,blobs,mins,maxs):
        for i in range(len(blobCounts)):
            if blobCounts[i]==0:
                continue

            pixels=np.array(blobs[i])
            mean=pixels.mean(axis=0)
            if (mean<mins).any() or (mean>maxs).any():
                blobCounts[i]=0
        return blobCounts.sum()

    def __filterMorphological(self,segmented_image,kernel_size=(20,20)):
        if segmented_image.dtype != "uint8":
            segmented_image = segmented_image.astype("uint8")*255
            
        # Morphological filtering the image
        kernel = np.ones(kernel_size, np.uint8)
        closed_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite("./ex05-3-closed.jpg", closed_image)
        return closed_image
    
    def __locateContours(self,closed_image):
        # Locate contours.
        contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    
    def __annotateImage(self,annotated_image,contours):
        
        if annotated_image.dtype != "uint8":
            annotated_image = annotated_image.astype("uint8")
    
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Draw a circle above the center of each of the detected contours.
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(annotated_image, (cx, cy), 10, (0, 0, 255), 2)
            else:
                if self._verbose:
                # Handle the case where m00 is zero if necessary
                    print("Contour with zero area detected, skipping.")

        # print("Number of detected balls: %d" % len(contours))

        return annotated_image


def __DebugLoadTestImage(filename=None,tiff=False):
    
    if filename is  None:
        filename = input_path + "/EB-02-660_0595_0435.JPG"
        # image_name = "EB-02-660_0595_0435"
        # image = "Data/Images/" + image_name + ".JPG"
        # image = "./Gyldensteensvej-9-19-2017-orthophoto.tif"

    # Load images
    image = cv2.imread(filename)
    print(image.shape)
    
    if tiff:
        temp = image.transpose(1, 2, 0)
        t2 = cv2.split(temp)
        img_cv = cv2.merge([t2[2], t2[1], t2[0]])
        img_tiff = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype("float")
        return img_tiff
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
    return image

if __name__=="__main__":
    Pc=PumpkinCounter(verbose=True)
    testImg=__DebugLoadTestImage()
    mins=np.array((1000,1000))
    maxs=np.array((testImg.shape[0]-1000,testImg.shape[1]-1000))
    print(Pc.ProcessImage(testImg,mins,maxs))
    # annotated_image = Pc.processImageContours(testImg)
    # cv2.imwrite("annotated_image.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print("Done")
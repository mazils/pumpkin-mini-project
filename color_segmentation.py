import cv2
import numpy as np
import os

script_dir = os.path.dirname(__file__)
input_rel = "../figures"
annotated_rel = "../annotated-images"  
output_rel = "../processed-images"
input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)
annotated_path = os.path.join(script_dir, annotated_rel)

class PumpkinCounter():
    def __init__(self,verbose=False):
        image,image_anotated=self._LoadReferenceImage()
        self._FindReferenceColorStats(image,image_anotated)
        self._verbose=verbose
        if self._verbose:
            print("Finish Loading Annotations")

    def ProcessImage(self,image):
        binary_Image=self._SegmentColors(image)
        if self._verbose:
            print("Finish Segmentation")

        numBlobs,blobMap=self._FindBlobs(binary_Image)
        if self._verbose:
            print("Finish Finding Blobs")

        BlobCount,EdgeBlobs=self._ProcessBlobs(numBlobs,blobMap)
        if self._verbose:
            print("Finish Processing Image")

        return BlobCount,EdgeBlobs

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


    def _ProcessBlobs(self,numlabels, labels_im):
        img_shape = labels_im.shape
        EdgeBlobs = dict()
        BlobCounts = np.zeros(numlabels)
        for x in range(img_shape[0]):
            for y in range(img_shape[1]):
                if labels_im[x, y] == 0:
                    continue

                blob_label = labels_im[x, y]
                if self._isEdge(x, y, img_shape):
                    if blob_label in EdgeBlobs.keys():
                        EdgeBlobs[blob_label].append(np.array([x, y]))
                    else:
                        EdgeBlobs[blob_label] = [np.array([x, y])]
                else:
                    BlobCounts[blob_label] = BlobCounts[blob_label] + 1

        return self._CleanUp(BlobCounts, EdgeBlobs)


    def _isEdge(self,x, y, img_shape):
        if x == 0 or y == 0:
            return True
        elif x == img_shape[0] or y == img_shape[1]:
            return True
        else:
            return False


    def _CleanUp(self,BlobCounts, EdgeBlobs):
        # filter out small blobs
        BlobCounts[BlobCounts < 20] = 0
        BlobCounts[np.logical_and(BlobCounts > 0, BlobCounts < 100)] = 1
        BlobCounts[BlobCounts > 100] = (BlobCounts[BlobCounts > 100] / 100).astype(np.int16)

        Pumking_Count = int(BlobCounts.sum())

        EdgeBlobsOut = list()
        for blob in EdgeBlobs.keys():
            tmp = np.array(EdgeBlobs[blob])
            EdgeBlobsOut.append(tmp)
        return Pumking_Count, EdgeBlobsOut

def __DebugLoadTestImage():
    
    image = input_path + "/EB-02-660_0595_0435.JPG"
    # image_name = "EB-02-660_0595_0435"
    # image = "Data/Images/" + image_name + ".JPG"

    # Load images
    image = cv2.imread(image)

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
    return image

if __name__=="__main__":
    Pc=PumpkinCounter(verbose=True)
    print(Pc.ProcessImage(__DebugLoadTestImage()))
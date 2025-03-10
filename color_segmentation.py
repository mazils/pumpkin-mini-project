import cv2
import numpy as np

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

        numBlobs, blobMap = self._FindBlobs(binary_Image)
        if self._verbose:
            print("Finish Finding Blobs")

        blobCounts, Blobs = self._ExtractBlobList(numBlobs, blobMap)
        if self._verbose:
            print("Finish Processing Blobs")

        blobCounts = self._ProcessBlobCounts(blobCounts)
        blobCount = self._ProcessBlobList(blobCounts, Blobs, mins, maxs)
        if self._verbose:
            print("Finish Counting")

        return int(blobCount)

    def _LoadReferenceImage(self):
        image_name = "EB-02-660_0595_0435"
        image = "Data/Images/" + image_name + ".JPG"
        image_annoted = "Data/Images_annotated/" + image_name + ".png"

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


def __DebugLoadTestImage():
    image_name = "EB-02-660_0595_0435"
    image = "Data/Images/" + image_name + ".JPG"

    # Load images
    image = cv2.imread(image)

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
    return image

if __name__=="__main__":
    Pc=PumpkinCounter(verbose=True)
    testImg=__DebugLoadTestImage()
    mins=np.array((1000,1000))
    maxs=np.array((testImg.shape[0]-1000,testImg.shape[1]-1000))
    print(Pc.ProcessImage(testImg,mins,maxs))
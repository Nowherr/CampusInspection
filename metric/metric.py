import numpy as np

__all__ = ['SegmentationMetric']


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        denominator = self.confusionMatrix.sum(axis=1)
        denominator = np.where(denominator == 0, 1e-12, denominator)
        classAcc = np.diag(self.confusionMatrix) / denominator
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        union = np.where(union == 0, 1e-12, union)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        denominator1 = np.sum(self.confusionMatrix)
        denominator1 = np.where(denominator1 == 0, 1e-12, denominator1)
        freq = np.sum(self.confusionMatrix, axis=1) / denominator1
        denominator2 = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        denominator2 = np.where(denominator2 == 0, 1e-12, denominator2)
        iu = np.diag(self.confusionMatrix) / denominator2
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def classF1Score(self):
        tp = np.diag(self.confusionMatrix)
        fp = self.confusionMatrix.sum(axis=0) - tp
        fn = self.confusionMatrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return f1

    def meanF1Score(self):
        f1 = self.classF1Score()
        mean_f1 = np.nanmean(f1)
        return mean_f1

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def get_scores(self):
        scores = {
            'Pixel Accuracy': self.pixelAccuracy(),
            'Class Pixel Accuracy': self.classPixelAccuracy(),
            'Intersection over Union': self.IntersectionOverUnion(),
            'Class F1 Score': self.classF1Score(),
            'Frequency Weighted Intersection over Union': self.Frequency_Weighted_Intersection_over_Union(),
            'Mean Pixel Accuracy': self.meanPixelAccuracy(),
            'Mean Intersection over Union(mIoU)': self.meanIntersectionOverUnion(),
            'Mean F1 Score': self.meanF1Score()
        }
        return scores

import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """

    def update(self, gt, pred):
        """ Overridden by subclasses """

    def get_results(self):
        """ Overridden by subclasses """

    def to_str(self, metrics):
        """ Overridden by subclasses """

    def reset(self):
        """ Overridden by subclasses """    

class StreamSegMetrics(_StreamMetrics):
    """Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        super(StreamSegMetrics, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        """    Pred
        True   5  4
               1  9
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        """
        5 0 0 2     acc: scalar
        0 5 0 0     acc_cls: scalar
        0 0 5 0     fwavacc: scalar
        0 0 0 3     iu: iou per class, mean_iu: scalar, cls_iu: K-dim vector
        """
        hist = self.confusion_matrix
        precision = np.divide(
            np.diag(hist), 
            hist.sum(axis=0), 
            out=np.zeros_like(np.diag(hist)), 
            where=(hist.sum(axis=0)!=0)
        )
        recall = np.divide(
            np.diag(hist), 
            hist.sum(axis=1), 
            out=np.zeros_like(np.diag(hist)), 
            where=(hist.sum(axis=1)!=0)
        )
        specificity = np.flip(np.divide(
            np.diag(hist), 
            hist.sum(axis=1), 
            out=np.zeros_like(np.diag(hist)), 
            where=(hist.sum(axis=1)!=0)
        ))
        iu = np.divide(
            np.diag(hist),
            (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)),
            out=np.zeros_like(np.diag(hist)),
            where=((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))!=0)
        )
        dice = np.divide(
            np.diag(hist) * 2,
            (hist.sum(axis=1) + hist.sum(axis=0)),
            out=np.zeros_like(np.diag(hist)),
            where=((hist.sum(axis=1) + hist.sum(axis=0))!=0)
        )
        cls_iu = dict(zip(range(self.n_classes), iu))
        cls_dice = dict(zip(range(self.n_classes), dice))
        
        # Legacy
        #acc = np.diag(hist).sum() / hist.sum()
        #acc_cls = np.diag(hist) / hist.sum(axis=1)
        #freq = hist.sum(axis=1) / hist.sum()
        #fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        #print(hist)
        return {
                "Class IoU": cls_iu,
                "Class Dice": cls_dice,
                "Class Precision": precision,
                "Class Recall": recall,
                "Class Specificity": specificity
            }

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


if __name__ == "__main__":
    import torch

    metrics = StreamSegMetrics(n_classes=2)

    true = torch.ones((2, 3, 4), dtype=torch.int64).numpy()
    pred = torch.randint(high=2, size=(2, 3, 4)).numpy()
    #true = torch.ones((2, 3, 4, 5), dtype=torch.int64).numpy()
    print( true )
    print( pred )
    metrics.update(label_trues=true, label_preds=pred)
    print(metrics.get_results())
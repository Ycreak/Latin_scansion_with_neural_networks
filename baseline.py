import utilities as util
import configparser

from sklearn.metrics import precision_recall_fscore_support
from sklearn_crfsuite import metrics

class Base_line:

    cf = configparser.ConfigParser()
    cf.read("config.ini")

    y_true = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'hmm_y_true'))

    def __init__(self):
      
        y_pred = [] # fill this with long
        for sentence in self.y_true:
            y_pred.append(['long']*len(sentence))        
        
        print(self.get_metrics_report(self.y_true, y_pred))

    def get_metrics_report(self, y_true, y_pred):
        sorted_labels = sorted(
            ['long', 'short', 'elision'],
            key=lambda name: (name[1:], name[0])
        )
        metrics_report = metrics.flat_classification_report(
            y_true, y_pred, labels=sorted_labels, digits=3
        )        
        return metrics_report        

base_line = Base_line()
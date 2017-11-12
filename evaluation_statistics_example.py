import evaluation_statistics as eval_stats
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    target = [2, 1, 3, 2, 2, 1, 3]
    predicted = [1, 1, 2, 2, 2, 1, 3]

    # Class names
    # 1 - SED
    # 2 - LPA
    # 3 - MVPA
    class_names = ['SED', 'LPA', 'MVPA']

    # Derive the confusion matrix using sklearn package
    cnf_matrix = confusion_matrix(target, predicted)

    stats = eval_stats.EvaluationStatistics.calculate(cnf_matrix)

    print(stats)

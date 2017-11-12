"""
The confusion matrix
When referring to the performance of a classification model, we are interested in the model’s ability to
correctly predict or separate the classes. When looking at the errors made by a classification model,
the confusion matrix gives the full picture. Consider e.g. a three class problem with the classes A, B, and C.

The rows correspond to the known class of the data, i.e. the labels in the data. The columns correspond to
the predictions made by the model. The value of each of element in the matrix is the number of predictions
made with the class corresponding to the column for examples with the correct value as represented by the row.
Thus, the diagonal elements show the number of correct classifications made for each class,
and the off-diagonal elements show the errors made.

 	 	Predicted class
 	 	 A 	 B 	 C
Truth	 A 	tpA	eAB	eAC
(Target) B 	eBA	tpB	eBC
class    C 	eCA	eCB	tpC

Accuracy
Accuracy is the overall correctness of the model and is calculated as the sum of correct classifications divided
by the total number of classifications.

Precision
Precision is a measure of the accuracy provided that a specific class has been predicted. It is defined by:
Precision = tp/(tp + fp)
where tp and fp are the numbers of true positive and false positive predictions for the considered class.

Recall
Recall is a measure of the ability of a prediction model to select instances of a certain class from a data set.
It is commonly also called sensitivity, and corresponds to the true positive rate. It is defined by the formula:

Recall = Sensitivity = tp/(tp+fn)
RecallA = SensitivityA = tpA/(tpA+eAB+eAC)

Specificity
Recall/sensitivity is related to specificity, which is a measure that is commonly used in two class problems
where one is more interested in a particular class. Specificity corresponds to the true-negative rate.

Specificity = tn/(tn+fp)
For class A, the specificity would correspond to the true-negative rate for class A
(as in not being a member of class A) and be calculated as:
SpecificityA = tnA/(tnA+eBA+eCA), where tnA = tpB + eBC + eCB + tpC

Source of above explanation: Online (http://www.compumine.com/web/public/newsletter/20071/precision-recall)
"""


class EvaluationStatistics:

    @staticmethod
    def calculate(confusion_matrix):

        # Name the values
        tpa = confusion_matrix[0, 0]
        tpb = confusion_matrix[1, 1]
        tpc = confusion_matrix[2, 2]
        eab = confusion_matrix[0, 1]
        eac = confusion_matrix[0, 2]
        eba = confusion_matrix[1, 0]
        ebc = confusion_matrix[1, 2]
        eca = confusion_matrix[2, 0]
        ecb = confusion_matrix[2, 1]

        # Calculate accuracy for label 1
        total_classifications = sum(sum(confusion_matrix))
        accuracy = (tpa + tpb + tpc) / total_classifications

        # Calculate Precision for label 1
        precisionA = tpa / (tpa + eba + eca)

        # Calculate Sensitivity for label 1
        sensitivityA = tpa / (tpa + eab + eac)

        # Calculate Specificity for label 1
        tna = tpb + ebc + ecb + tpc
        specificityA = tna / (tna + eba + eca)

        # Calculate Precision for label 2
        precisionB = tpb / (tpb + eab + ecb)

        # Calculate Sensitivity for label 2
        sensitivityB = tpb / (tpb + eba + ebc)

        # Calculate Specificity for label 2
        tnb = tpa + eac + eca + tpc
        specificityB = tnb / (tnb + eab + ecb)

        # Calculate Precision for label 2
        precisionC = tpc / (tpc + eac + ebc)

        # Calculate Sensitivity for label 2
        sensitivityC = tpc / (tpc + eca + ecb)

        # Calculate Specificity for label 2
        tnc = tpa + eab + eba + tpb
        specificityC = tnc / (tnc + eac + ebc)

        return {
            'accuracy': accuracy,
            'precision': [precisionA, precisionB, precisionC],
            'recall': [sensitivityA, sensitivityB, sensitivityC],
            'sensitivity': [sensitivityA, sensitivityB, sensitivityC],
            'specificity': [specificityA, specificityB, specificityC]
        }

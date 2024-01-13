from metrics.metrics import calculate_roc_auc, calculate_fpr_fnr, find_best_threshold
import numpy as np
def AUC_gap_matrix_class_based(list_list_of_probabilities, list_list_of_labels, class_names):
        # list_list_of_probabilities: list of list of probabilities each is for one subgroup
        # list_list_of_labels: list of list of labels each is for one subgroup
        # returns: ROC gap matrix for each subgroup comaared to the other subgroups for all classes it should be dict{Class1: Subgroup * Subgroup, Class2: Subgroup * Subgroup, ...}
        # ROC gap is the difference between the AUC of two subgroups for a class

        gaps = {}
        for cls_index,cls in enumerate(class_names):
            gap_temp = np.zeros((len(list_list_of_probabilities), len(list_list_of_probabilities)))
            for i in range(len(list_list_of_probabilities)):
                for j in range(len(list_list_of_probabilities)):
                    if i != j:
                        fpr, tpr, roc_auc = calculate_roc_auc(list_list_of_probabilities[i], list_list_of_labels[i])
                        fpr2, tpr2, roc_auc2 = calculate_roc_auc(list_list_of_probabilities[j], list_list_of_labels[j])
                        roc_gap = roc_auc[cls_index] - roc_auc2[cls_index]
                        gap_temp[i][j] = roc_gap
            gaps[cls] = gap_temp
        return gaps
# def calculate_fpr_fnr(probabilities, all_labels, thresholds):
def FPR_gap_matrix_class_based(list_list_of_probabilities, list_list_of_labels, class_names, thresholds):
        # list_list_of_probabilities: list of list of probabilities each is for one subgroup
        # list_list_of_labels: list of list of labels each is for one subgroup
        # thresholds: dict of thresholds for each class
        # returns: FPR gap matrix for each subgroup comaared to the other subgroups for all classes it should be dict{Class1: Subgroup * Subgroup, Class2: Subgroup * Subgroup, ...}
        # FPR gap is the difference between the FPR of two subgroups for a class

        gaps = {}
        for cls_index,cls in enumerate(class_names):
            gap_temp = np.zeros((len(list_list_of_probabilities), len(list_list_of_probabilities)))
            for i in range(len(list_list_of_probabilities)):
                for j in range(len(list_list_of_probabilities)):
                    if i != j:
                        fpr, fnr = calculate_fpr_fnr(list_list_of_probabilities[i], list_list_of_labels[i], thresholds)
                        fpr2, fnr2 = calculate_fpr_fnr(list_list_of_probabilities[j], list_list_of_labels[j], thresholds)
                        fpr_gap = fpr[cls_index] - fpr2[cls_index]
                        gap_temp[i][j] = fpr_gap
            gaps[cls] = gap_temp
        return gaps
def AUC_of_subgroups(list_list_of_probabilities, list_list_of_labels, class_names):
        # list_list_of_probabilities: list of list of probabilities each is for one subgroup
        # list_list_of_labels: list of list of labels each is for one subgroup
        # returns: AUC for each subgroup for each class it should be dict{Class1: Subgroup, Class2: Subgroup, ...}
     
        aucs = {}
        for cls_index,cls in enumerate(class_names):
            auc_temp = np.zeros(len(list_list_of_probabilities))
            for i in range(len(list_list_of_probabilities)):
                fpr, tpr, roc_auc = calculate_roc_auc(list_list_of_probabilities[i], list_list_of_labels[i])
                auc_temp[i] = roc_auc[cls_index]
            aucs[cls] = auc_temp
        return aucs
def FPR_of_subgroups(list_list_of_probabilities, list_list_of_labels, class_names, thresholds):
        # list_list_of_probabilities: list of list of probabilities each is for one subgroup
        # list_list_of_labels: list of list of labels each is for one subgroup
        # thresholds: dict of thresholds for each class
        # returns: FPR for each subgroup for each class it should be dict{Class1: Subgroup, Class2: Subgroup, ...}
     
        fprs = {}
        for cls_index,cls in enumerate(class_names):
            fpr_temp = np.zeros(len(list_list_of_probabilities))
            for i in range(len(list_list_of_probabilities)):
                fpr, fnr = calculate_fpr_fnr(list_list_of_probabilities[i], list_list_of_labels[i], thresholds)
                fpr_temp[i] = fpr[cls_index]
            fprs[cls] = fpr_temp
        return fprs




    


if __name__ == "__main__":
#test class

    # random list of probabilities and labels
    list_list_of_probabilities = np.random.rand(5, 1000, 14)
    list_list_of_labels = np.random.randint(0, 2, size=(5, 1000, 14))
    class_names = [i for i in range(14)]
    gaps = AUC_gap_matrix_class_based(list_list_of_probabilities, list_list_of_labels, class_names)
    for cls in class_names:
        print("Class: ", cls)
        print(gaps[cls])
        print()



                        



        

    
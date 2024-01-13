import lightning as L
import torch.nn
import numpy as np
import torch.nn as nn
import warnings
import torch.nn.functional as F
from metrics.metrics import calculate_roc_auc, calculate_fpr_fnr, find_best_threshold
import os

class CLS(L.LightningModule):

    def __init__(self, model: nn.Module, criterion=nn.CrossEntropyLoss(), lr = 0.001, weight_decay = 0.0005 , save_probabilities_path = None):
        super().__init__()
        print("CLS init", "*"*50)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_probabilities_path = save_probabilities_path



        self.criterion = criterion


        self.probabilities_val_test = []
        self.labels_val_test = []
        self.loss_val_test = []


    def training_step(self, batch, batch_idx):

        data, target = batch['data'].to(self.device), batch['labels'].to(self.device)


        
        output = self.model(data.float().squeeze())
        loss = self.criterion(output, target)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):


        loss = self.share_val(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        

    def test_step(self, batch, batch_idx):


        loss = self.share_val(batch)
        self.log('test_loss', loss, prog_bar=True, logger=True)


    def on_test_epoch_end(self):


        avg_loss = np.mean(self.loss_val_test)

        _,_,class_auc = calculate_roc_auc(self.probabilities_val_test, self.labels_val_test)
        average_auc = sum(class_auc.values()) / len(class_auc)

        print("-"*50)
        print("Test loss = {}".format(avg_loss))
        print("Average AUC = {}".format(average_auc))
        if self.save_probabilities_path is not None:
            # Ensure the directory exists, create it if necessary
            os.makedirs(self.save_probabilities_path, exist_ok=True)

            # Convert to numpy and save
            probabilities_val_test_np = np.array(self.probabilities_val_test)
            labels_val_test_np = np.array(self.labels_val_test)


            np.save(os.path.join(self.save_probabilities_path, "probabilities.npy"), probabilities_val_test_np)
            np.save(os.path.join(self.save_probabilities_path, "labels.npy"), labels_val_test_np)



        self.probabilities_val_test = []
        self.labels_val_test = []
        self.loss_val_test = []
        self.info = []

    def on_validation_epoch_end(self):
            
            avg_loss = np.mean(self.loss_val_test)
    
            _,_,class_auc = calculate_roc_auc(self.probabilities_val_test, self.labels_val_test)
            # thresholds_test = find_best_threshold(self.probabilities_val_test, self.labels_val_test)
            # print("Best thresholds for validation set: {}".format(thresholds_test))
            # fpr_numbers_test, fnr_numbers_test = calculate_fpr_fnr(self.probabilities_val_test, self.labels_val_test, thresholds_test)

            # # print fpr and fnr as a dictionary in a nice format in a table like this:

            # # | Class | FPR    | FNR    |
            # # |-------|--------|--------|
            # # |   0   |  0.00  |  0.00  |


            
            # print("| Class | FPR    | FNR    |")
            # print("|-------|--------|--------|")
            # for i in range(len(fpr_numbers_test)):
                
            #     print("|   {}   |   {:.3f}  |   {:.3f}  |".format(i, fpr_numbers_test[i], fnr_numbers_test[i]))


            
            
            average_auc = sum(class_auc.values()) / len(class_auc)

            print("-"*50)
            print("Validation loss = {}".format(avg_loss))
            print("Average AUC = {}".format(average_auc))
    
    
    
            self.probabilities_val_test = []
            self.labels_val_test = []
            self.loss_val_test = []
            self.info = []





    def share_val(self, batch):




        data, target  = batch['data'].to(self.device), batch['labels'].to(self.device) 
        output = self.model(data.float())


        loss = self.criterion(output, target).item()
        # append loss as numpy value
        self.loss_val_test.append(loss)
        self.labels_val_test.extend(target.data.cpu().numpy())



        if isinstance(self.criterion, nn.CrossEntropyLoss):
           self.probabilities_val_test.extend(torch.softmax(output, dim=1).data.cpu().numpy())

        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
           self.probabilities_val_test.extend(torch.sigmoid(output).data.cpu().numpy())


        else:
            warnings.warn("Loss is not supported for validation step. No probabilities will be returned.")
            self.probabilities_val_test.extend(output.data.cpu().numpy())

        return loss
    
    
    





    def configure_optimizers(self):
        
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
import numpy as np

import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def forecasting_loss(T_test, preds_edge, labels_edge, preds_negedge, labels_negedge, preds_attr, labels_attr, preds_negattr, labels_negattr):
    list_error_edge = []
    list_error_attr = []
    print("Edge NLL")
    for t in range(T_test):
        error_edge  = F.binary_cross_entropy(preds_edge[t], labels_edge[t], reduction='sum')
        error_edge += F.binary_cross_entropy(preds_negedge[t], labels_negedge[t], reduction='sum')
        list_error_edge.append(error_edge.to('cpu').detach().numpy().copy())
        print(error_edge.to('cpu').detach().numpy().copy())
    print()
    print("Average: ",np.average(list_error_edge))
    print()
    print("Attribute NLL")
    for t in range(T_test):
        error_attr  = F.binary_cross_entropy(preds_attr[t], labels_attr[t], reduction='sum')
        error_attr += F.binary_cross_entropy(preds_negattr[t], labels_negattr[t], reduction='sum')
        list_error_attr.append(error_attr.to('cpu').detach().numpy().copy())
        print(error_attr.to('cpu').detach().numpy().copy())
    print()
    print("Average: ",np.average(list_error_attr))
    print()
    return list_error_edge, list_error_attr


def forecasting_auc(T_test, preds_edge, labels_edge, preds_negedge, labels_negedge, preds_attr, labels_attr, preds_negattr, labels_negattr):
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    list_auc_edge = []
    list_auc_attr = []
    print('Edge AUC')
    for t in range(T_test):
        y_true = list(labels_edge[t].to('cpu').detach().numpy().copy())
        y_pred = list(preds_edge[t].to('cpu').detach().numpy().copy())
        y_true2 = list(labels_negedge[t].to('cpu').detach().numpy().copy())
        y_pred2 = list(preds_negedge[t].to('cpu').detach().numpy().copy())
        y_true.extend(y_true2)
        y_pred.extend(y_pred2)
        auc = roc_auc_score(y_true, y_pred)
        print(auc)
        list_auc_edge.append(auc)
    print()
    print("Average: ",np.average(list_auc_edge))
    print()
    print('Attribute AUC')
    for t in range(T_test):
        y_true = list(labels_attr[t].to('cpu').detach().numpy().copy())
        y_pred = list(preds_attr[t].to('cpu').detach().numpy().copy())
        y_true2 = list(labels_negattr[t].to('cpu').detach().numpy().copy())
        y_pred2 = list(preds_negattr[t].to('cpu').detach().numpy().copy())
        y_true.extend(y_true2)
        y_pred.extend(y_pred2)
        auc = roc_auc_score(y_true, y_pred)
        list_auc_attr.append(auc)
        print(auc)
    print()
    print("Average: ",np.average(list_auc_attr))
    return list_auc_edge, list_auc_attr


def loss_function_dynamic(model, preds_edge, labels_edge, preds_negedge, labels_negedge, predicted_U, predicted_V, predicted_X, preds_attr, labels_attr, preds_negattr, labels_negattr):
    cost_edge = 0
    cost_attr = 0
    cost_pred = 0
    cost_reg = 0
    for t in range(model.T_train):
        cur_cost_edge = 0
        cur_cost_attr = 0
        cur_cost_pred = 0
        cur_cost_reg = 0
        cur_cost_edge += F.binary_cross_entropy(preds_edge[t], labels_edge[t], reduction='sum')
        cur_cost_attr += F.binary_cross_entropy(preds_attr[t], labels_attr[t], reduction='sum')
        cur_cost_edge += F.binary_cross_entropy(preds_negedge[t], labels_negedge[t], reduction='sum')
        cur_cost_attr += F.binary_cross_entropy(preds_negattr[t], labels_negattr[t], reduction='sum')
        if t > 0:
            weight = 10.
            diff = predicted_U[t-1] - model.U[t]
            cost_pred += weight * torch.sum(diff*diff)
            diff = predicted_V[t-1] - model.V[t]
            cost_pred += weight * torch.sum(diff*diff)
            diff = predicted_X[t-1] - model.X[t]
            cost_pred += weight * torch.sum(diff*diff)
        dec_t = 1.
        cost_edge += dec_t * cur_cost_edge
        cost_attr += dec_t * cur_cost_attr
        cost_pred += dec_t * cur_cost_pred
        cost_reg  += dec_t * cur_cost_reg

    return cost_edge + cost_attr + cost_pred + cost_reg
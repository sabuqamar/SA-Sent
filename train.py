#!/usr/bin/python
from __future__ import division
from model import *
from config import config
import _pickle as cPickle
import copy
from sklearn.metrics import precision_recall_fscore_support

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (2 ** (epoch // config.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_opt(parameters, config):
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr)
    return optimizer

def load_data(data_path, if_utf=False):
    f = open(data_path, "rb")
    obj = cPickle.load(f)
    f.close()
    return obj

id2word = load_data(config.dic_path)
id2label = ["positive", "neutral", "negative"]

def train():
    print(config)
    best_acc = 0
    best_model = None

    train_batch, test_batch = load_data(config.data_path)
    model = AspectSent(config)
    if config.if_gpu: model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_opt(parameters, config)
    
    num_batches = len(train_batch)
    bound = num_batches // config.k_fold * (config.k_fold - 1) 
    print("{0} batches and {1} for dev".format(num_batches, num_batches - bound))
    
    cv_train = train_batch[ : bound]
    cv_test = train_batch[ bound : ]

    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  adjust_learning_rate(optimizer, e_)

        for triple_list in cv_train:
            model.zero_grad() 
            if len(triple_list) == 0: continue
            for sent, mask, label in triple_list:
                cls_loss = model(sent, mask, label)
                cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        acc = evaluate_dev(cv_test, model)
        print("Dev acc ", acc)
        if acc > best_acc: 
            best_acc = acc
            best_model = copy.deepcopy(model)
    evaluate_test(test_batch, best_model)
    print("Finish with best dev acc {0}".format(best_acc))
    torch.save(model, "model_params.json")

def visualize(sent, mask, best_seq, pred_label, gold):
    try:
        acc_sent = []
        for i, x in enumerate(sent):
            acc_sent.append(id2word[x])
        print(u" ".join(acc_sent))
    except:
        print("unknow char..")
        return
    print("Mask", mask)

    print("Seq", best_seq)
    print("Predict: {0}, Gold: {1}".format(id2label[pred_label], id2label[gold]))
    print("") 

def evaluate_test(test_batch, model):
    print("Evaluting")
    model.eval()
    all_counter = 0
    correct_count = 0
    print("transitions matrix ", model.inter_crf.transitions.data)
    # Initialize empty lists to store true labels and predicted labels
    y_true = []
    y_pred = []
    for sent, mask, label in test_batch:
        pred_label, best_seq = model.predict(sent, mask) 
        visualize(sent, mask, best_seq, pred_label, label)

        all_counter += 1
        if pred_label == label:  correct_count += 1
        y_true.append(label)
        y_pred.append(pred_label)
    acc = correct_count * 1.0 / all_counter
    print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
   # Calculate precision, recall, F1-score, and support using sklearn's precision_recall_fscore_support function
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1_score))
    
    return f1_score

def evaluate_dev(dev_batch, model):
    print("Evaluting")
    model.eval()
    all_counter = 0
    correct_count = 0
    for triple_list in dev_batch:
        for sent, mask, label in triple_list:
            pred_label, best_seq = model.predict(sent, mask) 

            all_counter += 1
            if pred_label == label:  correct_count += 1
    acc = correct_count * 1.0 / all_counter
    print("Sentiment Accuracy {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

if __name__ == "__main__":
    train()
import pickle as pkl
import argparse
from nn_layers import NNComp
import random
import numpy as np
from features import Features
import math
from classify import classify
from sklearn.metrics import f1_score, mean_squared_error, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def train(train_data_path, l_rate, n_epoch, save_path, validation_data_path, val_result_path):
    # network=NNComp(feature_dim=args.f,h_units=args.u,output_dim=n_outputs)
    learning_rate = l_rate
    with open(train_data_path, "rb") as f:
        X_train, y_train = pkl.load(f)
        ### Flatten the embeddings to 1d array ###
        X_train = np.asarray([np.asarray(row).flatten() for row in X_train])

    with open(validation_data_path, "rb") as f:
        X_val, y_val = pkl.load(f)
        X_val = np.asarray([np.asarray(row).flatten() for row in X_val])

    n_inputs = len(X_train[0])
    n_outputs = len(set(y_train))

    model = NNComp(feature_dim=n_inputs, h_units=2, output_dim=n_outputs)
    print("Start training...")

    print("Shape of training data", np.shape(X_train))
    print("Shape of test data",np.shape(X_val))
    val_results = open(val_result_path, "w")

    for epoch in range(n_epoch):
        y_true=[]
        y_pred=[]
        sum_error = 0
        # if epoch%15==0:
        #     learning_rate = step_decay(learning_rate, epoch)
        for i, row in enumerate(X_train):
            if i%2000==0:
                print("Sample %d" %(i))
            outputs = model.forward(row)
            # t_pred = [0 for _ in range(n_outputs)]
            # t_pred[np.argmax(outputs)] = 1
            y_pred.append(np.argmax(outputs))
            expected = [0 for _ in range(n_outputs)]
            expected[y_train[i]] = 1
            y_true.append(expected)


            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            model.backward(expected)
            model.update_weights(row, learning_rate)

        f1 = f1_score(y_true=y_train, y_pred=y_pred, average="macro")
        print('Train log : epoch=%d, lrate=%.3f, error=%.3f, f1=%.3f' % (epoch, learning_rate, sum_error,f1))
        ### validate on validation set and print the loss on val set and f1 score on val set ###
        print("Start validating...")
        validate(X_val, y_val, model, epoch, val_results)

    ### save the model ##
    val_results.close()
    model.save_model(save_path)


def validate(X_val, y_val, model, epoch, val_results):
    """
    validate the so far trained model on the validation set and save the f1 score ,loss and the epoch
    :param X_val:
    :param y_val:
    :param model:
    :param epoch:
    :return:
    """
    ## first classify the X_val and then get the loss and the f1 score iwth y_val
    ## and save it with the epoch number in the results path
    y_preds = []
    n_outputs = len(set(y_val))
    y_expected = [0 for _ in range(n_outputs)]
    # y_preds[y_train[i]] = 1
    y_true = []
    for i, sample in enumerate(X_val):
        pred = model.forward(sample)
        # t_pred = [0 for _ in range(n_outputs)]
        # t_pred[np.argmax(pred)] = 1
        # y_preds.append(t_pred)
        # y_expected = [0 for _ in range(n_outputs)]
        # y_expected[y_val[i]] = 1
        # y_true.append(y_expected)
        # y_true.append(y_val[i])
        y_preds.append(np.argmax(pred))

    # print(y_val)
    # print(y_preds)
    f1 = f1_score(y_true=y_val, y_pred=y_preds, average="weighted")
    mse = mean_squared_error(y_true=y_val, y_pred=y_preds)
    val_results.write("Epoch: %d, Val_loss : %.3f, Val_f1: %.3f" % (epoch, mse, f1))
    print("Epoch: %d, Val_loss : %.3f, Val_f1: %.3f" % (epoch, mse, f1))
    print("===============")
    val_results.write("\n")


def exp_decay(rate, epoch):
    initial_lrate = rate
    k = 0.1
    lrate = initial_lrate * math.exp(-k * epoch)
    return lrate


def step_decay(rate, epoch):
    initial_lrate = rate
    drop = 0.5
    epochs_drop = 15
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Neural net training arguments.')
    #
    # parser.add_argument('-u', type=int, help='number of hidden units')
    # parser.add_argument('-l', type=float, help='learning rate')
    # parser.add_argument('-f', type=int, help='max sequence length')
    # parser.add_argument('-b', type=int, help='mini-batch size')
    # parser.add_argument('-e', type=int, help='number of epochs to train for')
    # parser.add_argument('-E', type=str, help='word embedding file')
    # parser.add_argument('-i', type=str, help='training file')
    # parser.add_argument('-o', type=str, help='model file to be written')
    #
    # args = parser.parse_args()

    # train(train_data_path=args.i,l_rate=args.l,n_epoch=args.e,save_path=args.o)

    # f1 = Features(data_file="datasets/4dim.train.txt", stopwords_file="stopwords.txt")
    # X_train = f1.get_features("4dim", f1.preprocessed_text,40)
    # f1.save_features(vectors=X_train, save_path="4dim_embeds.pkl")
    # y_train = f1.numeric_classes
    # print("loading the embeddings...")
    # with open("4dim_embeds.pkl", "rb") as f:
    #     X_train, _, y_train = pkl.load(f)
    #     ### Flatten the embeddings to 1d array ###
    #     X_train = np.asarray([np.asarray(row).flatten() for row in X_train])
    # n_inputs = len(X_train[0])
    # n_outputs = len(set(y_train))
    # network = NNComp(feature_dim=n_inputs, h_units=2, output_dim=n_outputs)
    train(train_data_path="products_train.pkl", l_rate=0.3, n_epoch=100, save_path="products_V0.model",
          validation_data_path="products_test.pkl", val_result_path="products_val_results.txt")

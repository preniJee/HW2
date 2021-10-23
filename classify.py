import pickle
import argparse


def classify(X, model):
    outputs = model.forward(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()

    classify(args)

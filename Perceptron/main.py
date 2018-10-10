import numpy as np
import csv
import sys
import os
import matplotlib.pyplot as plt

from mlp import MLP

def read_csv(filePath):
    elements = []
    with open(filePath, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        rows.__next__()
        for row in rows:
            elements.append(row)
    return elements

def get_inputs(elements):
    inputs = []
    for row in elements:
            [inputs.append(x) for x in row[:-1]]
    return inputs

def get_outputs(elements):
    outputs = []
    for row in elements:
            outputs.append(row[-1])
    return outputs

def main(argv):
     
  #  trainFilePath = argv[1]
  #  testFilePath = argv[2]
    trainFilePath = os.getcwd() + "\data\\regression\data.activation.train.100.csv"
    testFilePath = os.getcwd() + "\data\\regression\data.activation.test.100.csv"

    regression = True


    train_elements = read_csv(trainFilePath)
    test_elements = read_csv(testFilePath)

    inputs = get_inputs(train_elements)
    outputs = get_outputs(train_elements)

    n = train_elements.count

    if(regression):
        print("not ready")
    else: #classification
        print("not ready")
        


if __name__ == "__main__":
    main(sys.argv[1:])
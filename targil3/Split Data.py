#write a program that recieves a file and splits the data in a 80 to 20 ratio
#and writes the data to two files train.txt and test.txt
#the data is in the format of:
#<label> <feature1>\n <label> <feature2>\n <label> <feature3>\n
#<label> <feature4>\n <label> <feature5>\n <label> <feature6>\n

import sys
import random
import os

RATIO = 0.8


def process_data(data_file, train_file, test_file):
    #open the file and read the lines
    with open(data_file, 'r') as file:
        lines = file.readlines()
        #shuffle the lines
        random.shuffle(lines)
        #split the lines into 80% and 20%
        train_lines = lines[:int(len(lines)*RATIO)]
        test_lines = lines[int(len(lines)*RATIO):]
    #write the lines to the train file
    with open(train_file, 'w') as file:
        for line in train_lines:
            file.write(line)
    #write the lines to the test file
    with open(test_file, 'w') as file:
        for line in test_lines:
            file.write(line)
    print(f"{data_file} split successfully")

def main():
    #recieve argument entered through the command line
    nn0 = "nn0.txt"
    nn1 = "nn1.txt"
    # add check that file exists
    if os.path.exists(nn0):
        process_data(nn0, "train_file0.txt", "test_file0.txt")
    else:
        print("nn0 not found")

    if os.path.exists(nn1):
        process_data(nn1, "nn1/train_file1.txt", "nn1/test_file1.txt")
    else:
        print("nn1 not found")

if __name__ == "__main__":
    main()
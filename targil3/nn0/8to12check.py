import random


def generate_binary_string(length):
    binary_string = ""
    for _ in range(length):
        binary_string += random.choice(['0', '1'])
    return binary_string


def label_binary_string(binary_string):
    count_ones = binary_string.count('1')
    if 12 >= count_ones >= 8:
        return "1"
    else:
        return "0"


def generate_and_label_strings(num_strings, file_name):
    with open(file_name, "w") as file:
        for _ in range(num_strings):
            binary_string = generate_binary_string(16)
            label = label_binary_string(binary_string)
            if label is not None:
                file.write(f"{binary_string}{label}\n")


# Generate and label 10 binary strings and write them to "testnet13.txt"
generate_and_label_strings(100000, "testnet0.txt")

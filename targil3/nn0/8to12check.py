import random


def generate_binary_string(length):
    binary_string = ""
    for _ in range(length):
        binary_string += random.choice(['0', '1'])
    return binary_string


def label_binary_string(binary_string, over_count):
    count_ones = binary_string.count('1')
    if 12 >= count_ones >= 8:
        return "1"
    elif count_ones > 12:
        over_count[0] += 1
        return "0"
    else:
        return "0"


def generate_and_label_strings(num_strings, file_name):
    with open(file_name, "w") as file:
        over_count = [0]  # Use a list to make over_count mutable
        for _ in range(num_strings):
            binary_string = generate_binary_string(16)
            label = label_binary_string(binary_string, over_count)
            if label is not None:
                file.write(f"{binary_string}\n")
        print(f"Over count: {over_count[0]}")


# Generate and label 10 binary strings and write them to "testnet13.txt"
generate_and_label_strings(100000, "testnet0.txt")


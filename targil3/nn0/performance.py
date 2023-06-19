import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics
import matplotlib.pyplot as plt


def read_csv(file_name):
    df = pd.read_csv(file_name)
    df['Generations'] = df['Generations'].replace(0, 300)
    run_time = df['Run Time'].tolist()
    generations = df['Generations'].tolist()
    train_accuracy = df['Train Accuracy'].tolist()
    test_accuracy = df['Test Accuracy'].tolist()

    return run_time, generations, train_accuracy, test_accuracy


def plot_comparison(run_time, generations, train_accuracy, test_accuracy, file_name):
    plt.figure(figsize=(12, 6))

    # Plot Run Time
    plt.subplot(221)
    plt.plot(run_time, 'b')
    plt.xlabel('Iterations')
    plt.ylabel('Run Time (sec)')
    plt.title('Run Time')

    # Plot Generations
    plt.subplot(222)
    plt.plot(generations, 'g')
    plt.xlabel('Iterations')
    plt.ylabel('Generations')
    plt.title('Generations')

    # Plot Train Accuracy
    plt.subplot(223)
    plt.plot(train_accuracy, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy')

    # Plot Test Accuracy
    plt.subplot(224)
    plt.plot(test_accuracy, 'm')
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy')

    plt.suptitle(f'Comparison for {file_name}')
    plt.tight_layout()
    plt.show()


def calculate_averages(run_time, generations, train_accuracy, test_accuracy):
    avg_run_time = statistics.mean(run_time)
    avg_generations = statistics.mean(generations)
    avg_train_accuracy = statistics.mean(train_accuracy)
    avg_test_accuracy = statistics.mean(test_accuracy)

    return avg_run_time, avg_generations, avg_train_accuracy, avg_test_accuracy


# Example usage
file_name = 'run_data250.csv'
run_time, generations, train_accuracy, test_accuracy = read_csv(file_name)
plot_comparison(run_time, generations, train_accuracy, test_accuracy, file_name)
avg_run_time, avg_generations, avg_train_accuracy, avg_test_accuracy = calculate_averages(
    run_time, generations, train_accuracy, test_accuracy
)

print(f'Average Run Time: {avg_run_time} sec')
print(f'Average Generations: {avg_generations}')
print(f'Average Train Accuracy: {avg_train_accuracy}')
print(f'Average Test Accuracy: {avg_test_accuracy}')

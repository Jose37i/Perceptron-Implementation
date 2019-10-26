import math
import random
import csv
import matplotlib.pyplot as plt


def normalize(graph_selection):
    if graph_selection == 'A':
        file = 'groupA.txt'
        graphs_epsilon = .00005
    elif graph_selection == 'B':
        file = 'groupB.txt'
        graphs_epsilon = 100
    else:
        file = 'groupC.txt'
        graphs_epsilon = 1450

    normalized_data = [[], [], [], []]
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        heights_arr, weights_arr, gender_arr = [], [], []

        for line in csv_reader:
            height = line[0]
            weight = line[1]
            gender = line[2]

            heights_arr.append(height)
            weights_arr.append(weight)
            gender_arr.append(gender)

        min_height = min(heights_arr)
        max_height = max(heights_arr)
        max_weight = max(weights_arr)
        min_weight = min(weights_arr)

        length = len(weights_arr)
        for i in range(length):
            height2 = heights_arr[i]
            weight2 = weights_arr[i]
            gender2 = gender_arr[i]

            normalized_height_numerator = float(height2) - float(min_height)
            normalized_height_denominator = float(max_height) - float(min_height)
            normalized_weight_numerator = float(weight2) - float(min_weight)
            normalized_weight_denominator = float(max_weight) - float(min_weight)
            normalized_height = float(normalized_height_numerator) / float(normalized_height_denominator)
            normalized_weight = float(normalized_weight_numerator) / float(normalized_weight_denominator)

            normalized_data[0].append(normalized_height)
            normalized_data[1].append(normalized_weight)
            normalized_data[2].append(int(1))
            normalized_data[3].append(gender2)

    return normalized_data, graphs_epsilon


def get_training_testing_data(inputs, passed_training_percentage):
    men_dictionary = {}
    female_dictionary = {}
    training_data = [[], [], [], []]
    testing_data = [[], [], [], []]
    passed_heights = inputs[0]
    passed_weights = inputs[1]
    passed_bias = inputs[2]
    passed_genders = inputs[3]

    if passed_training_percentage == 75:
        training_data_amount = 1500
    else:
        training_data_amount = 500

    for i in range(4000):
        if i < 2000:
            men_dictionary[i] = True
        else:
            female_dictionary[i] = True

    for _ in range(training_data_amount):
        men_random_number = random.choice(list(men_dictionary.keys()))
        training_data[0].append(passed_heights[men_random_number])
        training_data[1].append(passed_weights[men_random_number])
        training_data[2].append(passed_bias[men_random_number])
        training_data[3].append(passed_genders[men_random_number])
        men_dictionary.pop(men_random_number)

        female_random_number = random.choice(list(female_dictionary.keys()))
        training_data[0].append(passed_heights[female_random_number])
        training_data[1].append(passed_weights[female_random_number])
        training_data[2].append(passed_bias[female_random_number])
        training_data[3].append(passed_genders[female_random_number])
        female_dictionary.pop(female_random_number)

    whole_dictionary = men_dictionary
    whole_dictionary.update(female_dictionary)

    for key in whole_dictionary:
        testing_data[0].append(passed_heights[key])
        testing_data[1].append(passed_weights[key])
        testing_data[2].append(passed_bias[key])
        testing_data[3].append(passed_genders[key])

    return training_data, testing_data


def activation_function(data_point, passed_weights, function_type):
    net_val = 0
    net_val += data_point[0] * passed_weights[0]
    net_val += data_point[1] * passed_weights[1]
    net_val += data_point[2] * passed_weights[2]

    if function_type == 'Hard':
        if net_val > 0:
            return 1
        else:
            return 0
    else:
        gain_val = 0.1
        soft_result = 1 / (1 + (math.exp(-1 * gain_val * net_val)))
        return soft_result


def train(function_type, passed_data, p_weights, alpha, constant):
    height_data = passed_data[0]
    weight_data = passed_data[1]
    bias_data = passed_data[2]
    gender_data = passed_data[3]
    length = len(height_data)
    for iteration in range(5000):
        total_error_calc = 0
        for p in range(length):
            height_point = height_data[p]
            actual_weight = weight_data[p]
            bias_point = bias_data[p]
            arr = [height_point, actual_weight, bias_point]
            output = activation_function(arr, p_weights, function_type)
            error = int(gender_data[p]) - output
            total_error_calc += math.pow(error, 2)
            p_weights[0] += alpha * error * height_data[p]
            p_weights[1] += alpha * error * weight_data[p]
            p_weights[2] += alpha * error * bias_data[p]
        if total_error_calc <= constant:
            break
    return p_weights


def print_confusion_matrix(women_above_line, women_below_line, men_above_line, men_below_line):
    # true positives = women below line = a
    true_positives = len(women_below_line)
    # true negatives = men above line = d
    true_negatives = len(men_above_line)
    # false positives = men below line = c
    false_positives = len(men_below_line)
    # false negatives = women above line = b
    false_negatives = len(women_above_line)
    # true positive rate = a/a+b
    # false positive rate = c/c+d
    # true negative rate = d/c+d
    # false negative rate = b/a+b
    # accuracy = a+d/a+b+c+d
    # error = 1 - accuracy

    print("               Predicted Female   Predicted Male")
    print("------------------------------------------------")
    print("Actual Female     " + str(true_positives) + "                 " + str(false_negatives))
    print("------------------------------------------------")
    print("Actual Male       " + str(false_positives) + "                 " + str(true_negatives))

    print("\n")
    print("True Positives: " + str(true_positives))
    print("True Negatives: " + str(true_negatives))
    print("False Positives: " + str(false_positives))
    print("False Negatives: " + str(false_negatives))
    print("True Positive Rate: " + str((true_positives / (true_positives + false_negatives))))
    print("False Positive Rate: " + str((false_positives / (false_positives + true_negatives))))
    print("True Negative Rate: " + str((true_negatives / (false_positives + true_negatives))))
    print("False Negative Rate: " + str((false_negatives / (true_positives + false_negatives))))
    print("Accuracy: " + str(((true_positives + true_negatives) / (true_positives + false_negatives + false_positives
                                                                   + true_negatives)) * 100) + "%")
    print("Error: " + str(((1 - ((true_positives + true_negatives) / (true_positives + false_negatives + false_positives
                                                                      + true_negatives))) * 100)) + "%")


def make_graphs(data_passed, title_for_graph, graph_name, passed_slope, passed_y_intercept, passed_x_intercept):
    n_array = [data_passed[0], data_passed[1], data_passed[3]]
    gender_vals = n_array[2]
    heights_vals = n_array[0]
    weights_vals = n_array[1]
    male = []
    female = []

    for i in range(len(gender_vals)):
        if gender_vals[i] == '0':
            male.append(i)
        else:
            female.append(i)
    plt.figure(0)
    plt.title(title_for_graph)
    plt.xlabel('Height (ft)')
    plt.ylabel('Weight (lbs)')

    men_above_line = []
    men_below_line = []
    for index in male:
        male_plot = plt.scatter(heights_vals[index], weights_vals[index], color='#1D5DEC', s=7)
        if weights_vals[index] > passed_slope * heights_vals[index] + passed_y_intercept:
            men_above_line.append(weights_vals[index])
        elif weights_vals[index] < passed_slope * heights_vals[index] + passed_y_intercept:
            men_below_line.append(weights_vals[index])
        else:
            # Since arbitrary line was chosen, points that fell on the line were added to the "below line" list
            men_below_line.append(weights_vals[index])

    women_above_line = []
    women_below_line = []
    for index in female:
        female_plot = plt.scatter(heights_vals[index], weights_vals[index], color='#FE01B1', s=7)
        if weights_vals[index] > passed_slope * heights_vals[index] + passed_y_intercept:
            women_above_line.append(weights_vals[index])
        elif weights_vals[index] < passed_slope * heights_vals[index] + passed_y_intercept:
            women_below_line.append(weights_vals[index])
        else:
            # Since arbitrary line was chosen, points that fell on the line were added to the "above line" list
            women_above_line.append(weights_vals[index])

    print(title_for_graph)
    print_confusion_matrix(women_above_line, women_below_line, men_above_line, men_below_line)

    plt.legend((male_plot, female_plot),
               ('Male', 'Female'),
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=8,
               markerscale=2.0)

    plt.rcParams["figure.figsize"] = (10, 10)

    plt.plot([0, passed_y_intercept], [passed_x_intercept, 0], color='black', linewidth=1)
    plt.savefig(graph_name)


what_graph = str(input("Type A, B, C as the data set you want to use:"))
what_graph = what_graph.strip().upper()
function_selection = str(input("Type hard or soft as the choice of functionality:"))
function_selection = function_selection.strip().lower().capitalize()
training_percentage = str(input("What do you want the training percentage to be 75% or 25%"))
training_percentage.strip()
testing_percentage = 100 - int(training_percentage)
training_graph_title = 'Group ' + what_graph + ': ' + function_selection + ' Activation ' \
                       + str(training_percentage) + '% Training'
training_graph_name = function_selection + '.' + what_graph.upper() + '.Train' + str(training_percentage) + '.png'
testing_graph_title = 'Group ' + what_graph + ': ' + function_selection + ' Activation ' \
                      + str(testing_percentage) + '% Testing'
testing_graph_name = function_selection + '.' + what_graph.upper() + '.Test' + str(testing_percentage) + '.png'

# --------------------------------------------------------------------------------------------

data, epsilon = normalize(what_graph)

TrainingData, TestingData = get_training_testing_data(data, training_percentage)

weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
perceptron = train(function_selection, TrainingData, weights, 0.1, epsilon)
print('Final Weights are as follows: [Height_weight, Weight_weight, Bias_weight]')
print('[' + str(perceptron[0]) + ' , ' + str(perceptron[1]) + ' , ' + str(perceptron[2]) + ']\n\n')

slope = perceptron[0] / perceptron[1] * -1
y_intercept = (perceptron[2] / (-1 * perceptron[1]))
x_intercept = ((-y_intercept) / slope)
# ----------------------------------------------------------------------------------------------

make_graphs(TrainingData, training_graph_title, training_graph_name, slope, y_intercept, x_intercept)

print('\n')
make_graphs(TestingData, testing_graph_title, testing_graph_name, slope, y_intercept, x_intercept)

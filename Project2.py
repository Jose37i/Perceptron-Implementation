import math
import random
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize():
    files = ['groupA.txt', 'groupB.txt', 'groupC.txt']
    group_a_data = [[], [], [], []]
    group_b_data = [[], [], [], []]
    group_c_data = [[], [], [], []]

    for entry in files:
        data = [[], [], [], []]
        with open(entry, 'r') as csv_file:
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

                data[0].append(normalized_height)
                data[1].append(normalized_weight)
                data[2].append(int(1))
                data[3].append(gender2)
        if entry == 'groupA.txt':
            group_a_data = data
        elif entry == 'groupB.txt':
            group_b_data = data
        else:
            group_c_data = data

    return group_a_data, group_b_data, group_c_data


def get_training_testing_data(inputs, training_percentage):
    men_dictionary = {}
    female_dictionary = {}
    training_data = [[], [], [], []]
    testing_data = [[], [], [], []]
    passed_heights = inputs[0]
    passed_weights = inputs[1]
    passed_bias = inputs[2]
    passed_genders = inputs[3]

    if training_percentage == 75:
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


def activation_function(data_point, weights, function_type):
    net_val = 0
    net_val += data_point[0] * weights[0]
    net_val += data_point[1] * weights[1]
    net_val += data_point[2] * weights[2]

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
    wieht_data = passed_data[1]
    bias_data = passed_data[2]
    gender_data = passed_data[3]
    length = len(height_data)
    for iteration in range(5000):
        total_error_calc = 0
        for p in range(length):
            height_point = height_data[p]
            actual_weight = wieht_data[p]
            bias_point = bias_data[p]
            arr = [height_point, actual_weight, bias_point]
            output = activation_function(arr, p_weights, function_type)
            error = int(gender_data[p]) - output
            total_error_calc += math.pow(error, 2)
            p_weights[0] += alpha * error * height_data[p]
            p_weights[1] += alpha * error * wieht_data[p]
            p_weights[2] += alpha * error * bias_data[p]
        if total_error_calc <= constant:
            break
    return p_weights


what_graph = str(input("Type A, B, C as the data set you want to use:"))
what_graph = what_graph.strip().upper()
function_selection = str(input("Type hard or soft as the choice of functionality:"))
function_selection = function_selection.strip().lower().capitalize()
training_percentage = str(input("What do you want the training percentage to be 75% or 25%"))
training_percentage.strip()
testing_percentage = 100 - int(training_percentage)
training_file_name = what_graph + '.' + function_selection + '.Training' + training_percentage
testing_file_name = what_graph + '.' + function_selection + '.Testing' + str(testing_percentage)


# --------------------------------------------------------------------------------------------
a_data, b_data, c_data = normalize()
if what_graph == 'A':
    epsilon = .00005
    data = a_data
elif what_graph == 'B':
    epsilon = 100
    data = b_data
else:
    epsilon = 1450
    data = c_data
TrainingData, TestingData = get_training_testing_data(data, training_percentage)

weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
perceptron = train(function_selection, TrainingData, weights, 0.1, epsilon)  # -------------------------change this
print 'Final Weights are as follows: [Height_weight, Weight_weight, Bias_weight]'
print '[' + str(perceptron[0]) + ' , ' + str(perceptron[1]) + ' , ' + str(perceptron[2]) + ']\n\n'

slope = perceptron[0] / perceptron[1] * -1
y_intercept = (perceptron[2] / (-1 * perceptron[1]))
x_intercept = ((-y_intercept) / slope)
# ----------------------------------------------------------------------------------------------

nArray = [TrainingData[0], TrainingData[1], TrainingData[3]]
gender_for_training = nArray[2]
heights_for_training = nArray[0]
weightsVals_for_training = nArray[1]
male = []
female = []

for i in range(len(gender_for_training)):
    if gender_for_training[i] == '0':
        male.append(i)
    else:
        female.append(i)
plt.figure(0)
plt.title('Group ' + what_graph + ': Activation ' + str(training_percentage) + '% Training')
plt.xlabel('Height (ft)')
plt.ylabel('Weight (lbs)')

menAboveLine1 = []
menBelowLine1 = []
for index in male:
    male_plot = plt.scatter(heights_for_training[index], weightsVals_for_training[index], color='#1D5DEC', s=7)
    if weightsVals_for_training[index] > slope * heights_for_training[index] + y_intercept:
        menAboveLine1.append(weightsVals_for_training[index])
    elif weightsVals_for_training[index] < slope * heights_for_training[index] + y_intercept:
        menBelowLine1.append(weightsVals_for_training[index])
    else:
        # Since arbitrary line was chosen, points that fell on the line were added to the "below line" list
        menBelowLine1.append(weightsVals_for_training[index])

womenAboveLine = []
womenBelowLine = []
for index in female:
    female_plot = plt.scatter(heights_for_training[index], weightsVals_for_training[index], color='#FE01B1', s=7)
    if weightsVals_for_training[index] > slope * heights_for_training[index] + y_intercept:
        womenAboveLine.append(weightsVals_for_training[index])
    elif weightsVals_for_training[index] < slope * heights_for_training[index] + y_intercept:
        womenBelowLine.append(weightsVals_for_training[index])
    else:
        # Since arbitrary line was chosen, points that fell on the line were added to the "above line" list
        womenAboveLine.append(weightsVals_for_training[index])

# true positives = women below line = a
a = float(len(womenBelowLine))
# true negatives = men above line = d
d = float(len(menAboveLine1))
# false positives = men below line = c
c = float(len(menBelowLine1))
# false negatives = women above line = b
b = float(len(womenAboveLine))
# true positive rate = a/a+b
# false positive rate = c/c+d
# true negative rate = d/c+d
# false negative rate = b/a+b
# accuracy = a+d/a+b+c+d
# error = 1 - accuracy


print 'Group ' + what_graph + ': Activation ' + str(training_percentage) + '% Training'
print("               Predicted Female   Predicted Male")
print("------------------------------------------------")
print("Actual Female     " + str(a) + "                 " + str(b))
print("------------------------------------------------")
print("Actual Male       " + str(c) + "                 " + str(d))

print("\n")
print("True Positives: " + str(a))
print("True Negatives: " + str(d))
print("False Positives: " + str(c))
print("False Negatives: " + str(b))
print("True Positive Rate: " + str(float(a) / (float(a) + float(b))))
print("False Positive Rate: " + str((float(c) / (float(c) + float(d)))))
print("True Negative Rate: " + str((float(d) / (float(c) + float(d)))))
print("False Negative Rate: " + str((float(b) / (float(a) + float(b)))))
print("Accuracy: " + str(((float(a) + float(d)) / (float(a) + float(b) + float(c) + float(d))) * 100) + "%")
print("Error: " + str(((1 - ((float(a) + float(d)) / (float(a) + float(b) + float(c) + float(d)))) * 100)) + "%\n\n")

# ## Group B

# In[22]:

plt.legend((male_plot, female_plot),
           ('Male', 'Female'),
           scatterpoints=1,
           loc='lower right',
           ncol=2,
           fontsize=8,
           markerscale=2.0)

plt.rcParams["figure.figsize"] = (10, 10)

plt.plot([0, y_intercept], [x_intercept, 0], color='black', linewidth=1)

plt.savefig(function_selection + '.' + what_graph.upper() + '.Train' + str(training_percentage) + '.png')

nArray2 = [TestingData[0], TestingData[1], TestingData[3]]
gender_Testing = nArray2[2]
height_Testing = nArray2[0]
weightVals_Testing = nArray2[1]
male_Testing = []
female_Testing = []

for i in range(len(gender_Testing)):
    if gender_Testing[i] == '0':
        male_Testing.append(i)
    else:
        female_Testing.append(i)
plt.figure(1)
plt.title('Group ' + what_graph + ': Activation ' + str(training_percentage) + '% Testing')
plt.xlabel('Height (ft)')
plt.ylabel('Weight (lbs)')

menAboveLine_Testing = []
menBelowLine_Testing = []
for index in male_Testing:
    male_plot = plt.scatter(height_Testing[index], weightVals_Testing[index], color='#1D5DEC', s=7)
    if weightVals_Testing[index] > slope * height_Testing[index] + y_intercept:
        menAboveLine_Testing.append(weightVals_Testing[index])
    elif weightVals_Testing[index] < slope * height_Testing[index] + y_intercept:
        menBelowLine_Testing.append(weightVals_Testing[index])
    else:
        # Since arbitrary line was chosen, points that fell on the line were added to the "below line" list
        menBelowLine_Testing.append(weightVals_Testing[index])

womenAboveLine_Testing = []
womenBelowLine_Testing = []
for index in female_Testing:
    female_plot = plt.scatter(height_Testing[index], weightVals_Testing[index], color='#FE01B1', s=7)
    if weightVals_Testing[index] > slope * height_Testing[index] + y_intercept:
        womenAboveLine_Testing.append(weightVals_Testing[index])
    elif weightVals_Testing[index] < slope * height_Testing[index] + y_intercept:
        womenBelowLine_Testing.append(weightVals_Testing[index])
    else:
        # Since arbitrary line was chosen, points that fell on the line were added to the "above line" list
        womenAboveLine_Testing.append(weightVals_Testing[index])

# true positives = women below line = a
a_Testing = float(len(womenBelowLine_Testing))
# true negatives = men above line = d
d_Testing = float(len(menAboveLine_Testing))
# false positives = men below line = c
c_Testing = float(len(menBelowLine_Testing))
# false negatives = women above line = b
b_Testing = float(len(womenAboveLine_Testing))
# true positive rate = a/a+b
# false positive rate = c/c+d
# true negative rate = d/c+d
# false negative rate = b/a+b
# accuracy = a+d/a+b+c+d
# error = 1 - accuracy

print 'Group ' + what_graph + ': Activation ' + str(training_percentage) + '% Testing'
print("               Predicted Female   Predicted Male")
print("------------------------------------------------")
print("Actual Female     " + str(a_Testing) + "                 " + str(b_Testing))
print("------------------------------------------------")
print("Actual Male       " + str(c_Testing) + "                 " + str(d_Testing))

print("\n")
print("True Positives: " + str(a_Testing))
print("True Negatives: " + str(d_Testing))
print("False Positives: " + str(c_Testing))
print("False Negatives: " + str(b_Testing))
print("True Positive Rate: " + str(float(a_Testing) / (float(a_Testing) + float(b_Testing))))
print("False Positive Rate: " + str((float(c_Testing) / (float(c_Testing) + float(d_Testing)))))
print("True Negative Rate: " + str((float(d_Testing) / (float(c_Testing) + float(d_Testing)))))
print("False Negative Rate: " + str((float(b_Testing) / (float(a_Testing) + float(b_Testing)))))
print("Accuracy: " + str(((float(a_Testing) + float(d_Testing)) / (
        float(a_Testing) + float(b_Testing) + float(c_Testing) + float(d_Testing))) * 100) + "%")
print("Error: " + str(((1 - ((float(a_Testing) + float(d_Testing)) / (
        float(a_Testing) + float(b_Testing) + float(c_Testing) + float(d_Testing)))) * 100)) + "%")

plt.rcParams["figure.figsize"] = (10, 10)
y_intercept = (perceptron[2] / (-1 * perceptron[1]))
x_intercept = ((-y_intercept) / (-1 * perceptron[0] / perceptron[1]))
plt.plot([0, y_intercept], [x_intercept, 0], color='black', linewidth=1)

plt.savefig(function_selection + '.' + what_graph.upper() + '.Test' + str(testing_percentage) + '.png')

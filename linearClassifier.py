def find_centroid(numFeatures, numExamples, start, training_input, numActual):
    centroid = []
    #count = 1
    for j in range(numFeatures):
        curSum = 0
        for i in range(start,numExamples):
            curSum += training_input[i][j]
        centroid.append(curSum / numActual)

    return centroid

def find_midPoint(x, y):
    midPoint = []
    curMid = 0
    for i in range(len(x)):
        curMid = (x[i] + y[i])/2
        midPoint.append(curMid)

    return midPoint

def find_t(x,y):
    a = []
    b = []
    sum = 0
    for i in range(len(x)):
        cur = x[i] + y[i]
        a.append(cur)
        cur = x[i] - y[i]
        b.append(cur)

    for i in range(len(a)):
        cur = a[i] * b[i]
        sum += cur

    return sum/2

def dot(x,y):
    sum = 0
    for i in range(len(x)):
        cur = x[i] * y[i]
        sum += cur

    return sum

def difference(x,y):
    w = []
    for i in range(len(x)):
        w.append(x[i]-y[i])

    return w

def findBias(x,y):
    bias = dot(x,y)
    return bias

def run_train_test(training_input, testing_input):

    Acentroid = find_centroid(training_input[0][0], 1+training_input[0][1], 1, training_input, training_input[0][1])
    Bcentroid = find_centroid(training_input[0][0], training_input[0][2] + 1+training_input[0][1], 1+training_input[0][1], training_input, training_input[0][2])
    Ccentroid = find_centroid(training_input[0][0], training_input[0][3] + 1+training_input[0][1]+training_input[0][2], 1+training_input[0][1]+training_input[0][2], training_input, training_input[0][3])

    w_AB = difference(Acentroid, Bcentroid)
    w_AC = difference(Acentroid, Ccentroid)
    w_BC = difference(Bcentroid, Ccentroid)

    midpoint_AB = find_midPoint(Acentroid, Bcentroid)
    midpoint_AC = find_midPoint(Acentroid, Ccentroid)
    midpoint_BC = find_midPoint(Bcentroid, Ccentroid)

    bias_AB = findBias(w_AB, midpoint_AB)
    bias_AC = findBias(w_AC, midpoint_AC)
    bias_BC = findBias(w_BC, midpoint_BC)
    
    w_AB.append(-bias_AB)
    w_AC.append(-bias_AC)
    w_BC.append(-bias_BC)

    # 'A or B'
    values = []
    curDataPoint = []


    for i in range(1,1+testing_input[0][1]):
        curDataPoint = testing_input[i]
        curDataPoint.append(1)
        # print(curDataPoint)
        cur = dot(curDataPoint, w_AB)
        values.append(cur)

    for i in range(1+testing_input[0][1], 1+testing_input[0][1]+testing_input[0][2]):
        curDataPoint = testing_input[i]
        curDataPoint.append(1)
        cur = dot(curDataPoint, w_AB)
        values.append(cur)

    for i in range(1+testing_input[0][1]+testing_input[0][2],1+testing_input[0][1]+testing_input[0][2]+testing_input[0][3]):
        curDataPoint = testing_input[i]
        curDataPoint.append(1)
        cur = dot(curDataPoint, w_AB)
        values.append(cur)

    predicted_labels = []
    for i in range(0,len(values)):
        if values[i] >= 0:
            #check if A or C
            
            curDataPoint = testing_input[i+1]
            cur = dot(curDataPoint, w_AC)
            if cur >= 0:
                predicted_labels.append(0)
            else:
                predicted_labels.append(2)

        else:
            curDataPoint = testing_input[i+1]
            cur = dot(curDataPoint, w_BC)
            if cur >= 0:
                predicted_labels.append(1)
            else:
                predicted_labels.append(2)


    confusion_matrix = [[0,0,0,],[0,0,0,],[0,0,0,]]
    for i in range(0,testing_input[0][1]):
        if predicted_labels[i] == 0:
            confusion_matrix[0][0] += 1
        elif predicted_labels[i] == 1:
            confusion_matrix[1][0] += 1
        else:
            confusion_matrix[2][0] +=1

    for i in range(testing_input[0][1], testing_input[0][1]+testing_input[0][2]):
        if predicted_labels[i] == 0:
            confusion_matrix[0][1] += 1
        elif predicted_labels[i] == 1:
            confusion_matrix[1][1] += 1
        else:
            confusion_matrix[2][1] +=1

    for i in range(testing_input[0][1]+testing_input[0][2], testing_input[0][1]+testing_input[0][2]+testing_input[0][3]): #FIX INDEXING
        if predicted_labels[i] == 0:
            confusion_matrix[0][2] += 1
        elif predicted_labels[i] == 1:
            confusion_matrix[1][2] += 1
        else:
            confusion_matrix[2][2] +=1

    # list = [TP, FN, FP, TN]

    resultsA = []
    resultsA.append(confusion_matrix[0][0])
    resultsA.append(confusion_matrix[1][0]+confusion_matrix[2][0])
    resultsA.append(confusion_matrix[0][1]+confusion_matrix[0][2])
    resultsA.append(confusion_matrix[1][1]+confusion_matrix[1][2]+confusion_matrix[2][1]+confusion_matrix[2][2])

    resultsB = []
    resultsB.append(confusion_matrix[1][1])
    resultsB.append(confusion_matrix[0][1]+confusion_matrix[2][1])
    resultsB.append(confusion_matrix[1][0]+confusion_matrix[1][2])
    resultsB.append(confusion_matrix[0][0]+confusion_matrix[0][2]+confusion_matrix[2][0]+confusion_matrix[2][2])
    
    resultsC = []
    resultsC.append(confusion_matrix[2][2])
    resultsC.append(confusion_matrix[0][2]+confusion_matrix[1][2])
    resultsC.append(confusion_matrix[2][0]+confusion_matrix[2][1])
    resultsC.append(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])


    Astats = []
    Astats.append(resultsA[0]/testing_input[0][1])
    Astats.append(resultsA[2]/(testing_input[0][2]+testing_input[0][3]))
    Astats.append((resultsA[2]+resultsA[1])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Astats.append((resultsA[0]+resultsA[3])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Astats.append(resultsA[0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2]))

    Bstats = []
    Bstats.append(resultsB[0]/testing_input[0][2])
    Bstats.append(resultsB[2]/(testing_input[0][1]+testing_input[0][3]))
    Bstats.append((resultsB[2]+resultsB[1])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Bstats.append((resultsB[0]+resultsB[3])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Bstats.append(resultsB[0]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2]))

    Cstats = []
    Cstats.append(resultsC[0]/testing_input[0][3])
    Cstats.append(resultsC[2]/(testing_input[0][1]+testing_input[0][2]))
    Cstats.append((resultsC[2]+resultsC[1])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Cstats.append((resultsC[0]+resultsC[3])/(testing_input[0][1] + testing_input[0][2]+testing_input[0][3]))
    Cstats.append(resultsC[0]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2]))

    averages = []
    for i in range(len(Cstats)):
        averages.append( (Astats[i] + Bstats[i] + Cstats[i]) / 3)

    testResults = {"tpr": averages[0],
                   "fpr": averages[1],
                   "error_rate": averages[2],
                   "accuracy": averages[3],
                   "precision": averages[4]
                    }

    return testResults


def parse_file(filename):
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    print(run_train_test(training_input, testing_input))


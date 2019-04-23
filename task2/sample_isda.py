import json
from collections import defaultdict
import nltk
import en_core_web_sm
nlp = en_core_web_sm.load()

DATAPOINTS = ["delivery_currency", "delivery_amount", "delivery_rounding",
              "return_currency", "return_amount", "return_rounding"]


def read_json():
    """The function is used to read the isda data provided as part of the test

    Returns:
        list: list of dictionary values which contains the text and the datapoints as keys.
    """
    with open("isda_data.json") as f_p:
        list_isda_data = json.load(f_p) #loads data from the json file

    return list_isda_data


def extract(input_data):
    """The function is used to build the extraction logic for ISDA datapoint
    extraction from the list of text inputs. Write your code logic in this function.

    Args:
        input_data (list): The input is a list of text or string from which the datapoints need to be extracted

    Returns:
        list: the function returns a list of dictionary values which contains the predictions for the input
              text. Note: The predictions should not be in a misplaced order.
              Example Output:
              [
                  {
                        "delivery_currency": "USD",
                        "delivery_amount": "10,000",
                        "delivery_rounding": "nearest",
                        "return_currency": "USD",
                        "return_amount": "10,000",
                        "return_rounding": "nearest"
                  },
                  ...
              ]
    """

    predicted_output = []
    words = []
    curr = []
    amt = []
    round = []
    overall = []
    for data in input_data:
        words = data.split(" ") #getting the words from each sentence
        for x in range(0,len(words)):
            if words[x].isupper(): #finding the currency and amount. Since the currency is always in capital letters and followed by a number, this pattern is matched in the words.
                if x <(len(words)-1): #currency can't be the last word assumption
                    clw=""
                    cln = ""
                    for char in words[x+1]: #removing punctuation marks from the amount
                        if char not in ",;":
                            clw=clw+char
                    if clw.isdigit():
                        #print(words[x])
                        for char in words[x + 1]:
                            if char not in ";":
                                cln = cln + char
                        curr.append(words[x]) #adding currency and amount if the pattern is satisfied
                        amt.append(cln)

            if "round" in words[x]:
                if x <len(words)-1:
                    if words[x+1]=="up" or words[x+1]=="down": #rounding is usually mentioned after "round" in sentence, therefore adding that if encountered
                        round.append(words[x+1])
                    else:
                        round.append("nearest") #taking the rounding as nearest by default if up/down is not found

        #if delivery and recieving values are not mentioned seperately then consdering them the same and copying
        if len(curr) == 1:
            curr.append(curr[0])
        if len(amt) == 1:
            amt.append(amt[0])
        if len(round) == 1:
            round.append(round[0])

        #creating the full output of each sentence as a list
        for i in range(0,2):
            overall.append(curr[i])
            overall.append(amt[i])
            overall.append(round[i])
        temp_dict = {}

        #creating a dictionary from the output list
        y=0
        for key in DATAPOINTS:
            temp_dict[key] = overall[y]
            #predicted_output.append(overall[y])

            y=y+1

        predicted_output.append(temp_dict)
        curr.clear()
        amt.clear()
        round.clear()
        overall=[]
        print("Predicted output of current sentence: ")
        print(temp_dict)
        temp_dict = {}

    print("Predicted output of all sentences combined: ")
    print(predicted_output)
    return predicted_output


def evaluate(input_data, predicted_output):
    """The function computes the accuracy for each of the datapoints
    that are part of the information extraction problem.

    Args:
        input_data (list): The input is a list of dictionary values from the isda json file
        predicted_output (list): This is a list which the information extraction algorithm should extract from the text

    Returns:
        dict: The function returns a dictionary which contains the number of exact between the input and the output
    """

    result = defaultdict(lambda: 0)
    for i, input_instance in enumerate(input_data):
        for key in DATAPOINTS:
            if input_instance[key] == predicted_output[i][key]: #comparing the values of output in the original json file and the values predicted by my algorithm
                result[key] += 1


    # compute the accuracy for each datapoint
    print("Accuracy scores: ")
    for key in DATAPOINTS:
        #print("accr",result[key])
        print(key, 1.0 * result[key] / len(input_data)) #calculating the accuracy score

    return result


if __name__ == "__main__":
    json_data = read_json()
    text_data = [data['text'] for data in json_data] #extracting the sentences from json and sending them to extraction function
    predicted_output = extract(text_data)

    # extracting the actual outputs from the json file for each sentence
    actual_output = []
    temp = {}
    for rec in json_data:
        for key in DATAPOINTS:
            temp[key] = rec[key]
        actual_output.append(temp)
        temp={}
    print("Actual output of all sentences combined(extracted from json):: ")
    print(actual_output)
    result = evaluate(actual_output, predicted_output)

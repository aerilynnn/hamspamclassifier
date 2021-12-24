import glob
import math
import re

# Model hyperparamters
alpha = 0.1
total_vocab = 200000

# Paths to files
HAM_PATH = r'HamSpam/ham/*.words'
SPAM_PATH = r'HamSpam/spam/*.words'
TEST_PATH = r'HamSpam/test/*.words'
TRUTHFILE = r'HamSpam/truthfile'

# Initializing spam and ham dictionaries
spam_dict = {}
ham_dict = {}

# Create the ham word probability dictionary
total_ham_words = 0
for file in glob.glob(HAM_PATH):
    ham_files = ham_files + 1
    f = open(file, 'r')
    while True:
        total_ham_words += 1
        line = f.readline().strip()
        if line in ham_dict:
            ham_dict[line] = ham_dict[line] + 1
        else:
            ham_dict[line] = 1
        if not line:
            break

# Create the spam word probability dictionary
total_spam_words = 0
for file in glob.glob(SPAM_PATH):
    spam_files = spam_files + 1
    f = open(file, 'r')
    while True:
        total_spam_words += 1
        line = f.readline().strip()
        if line in spam_dict:
            spam_dict[line] = spam_dict[line] + 1
        else:
            spam_dict[line] = 1
        if not line:
            break

# Total words (ham and spam)
n = total_ham_words + total_spam_words

# Calculate the smooth probability for all ham words 
for key in ham_dict.keys():
    ham_dict[key] = math.log((ham_dict[key] + 1)/(n + (alpha * total_vocab)))

# Calculate the smoothed probability for all spam words 
for key in spam_dict.keys():
    spam_dict[key] = math.log((spam_dict[key] + 1)/(n + (alpha * total_vocab)))

# Calculate probability of ham and spam words 
prob_ham = math.log(total_ham_words/n)
prob_spam = math.log(total_spam_words/n)

# Probability of an unseen word
unseen_word_prob = math.log(alpha/(n + (alpha * total_vocab)))

# Classifies an email (in this case a file) as spam or not spam.
# If spam, 1 is returned otherwise 0 if ham. 
def classify(email):
    # Open the email
    f = open(email, 'r')

    # Calculate the probability of the email being spam or spam 
    total_spam_probability = 0
    total_ham_probability = 0
    while True:
        line = f.readline().strip()
        # Calculate probability of spam
        if line in spam_dict:
            total_spam_probability = total_spam_probability + spam_dict[line]
        else:
            total_spam_probability = total_spam_probability + unseen_word_prob
        # Calculate probability of ham 
        if line in ham_dict:
            total_ham_probability = total_ham_probability + ham_dict[line]
        else:
            total_ham_probability = total_ham_probability + unseen_word_prob
        if not line: # EOF
            break
    total_spam_probability = prob_spam + total_spam_probability
    total_ham_probability = prob_ham + total_ham_probability

    # Classify 
    if total_spam_probability > total_ham_probability:
        return 1 # Classified as spam
    else:
        return 0 # Classified as ham 


# Classify the test data
test_dict = {}
for file in glob.glob(TEST_PATH):
    file_num = re.search("(\d{1,3})", file).group(0)
    test_dict[file_num] = classify(file)


# Evaluate the classifer
TN = 0
TP = 0
FP = 0
FN = 0
f = open(TRUTHFILE, 'r')
while True:
    line = f.readline().strip()
    if not line:
        break
    res = test_dict[line]
    del test_dict[line] # Remove key value pair from dictionary so it's not double counted later
    if res == 1:
        TP += 1
    else:
        FN += 1

for key in test_dict.keys():
    if test_dict[key] == 1:
        FP += 1
    else:
        TN += 1

accuracy = (TP + TN)/(TP + FP + FN + TN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
fscore = 2 * ((precision * recall)/(precision + recall))

# Print the results out 
print("TP: %d" %TP)
print("TN: %d" %TN)
print("FP: %d" %FP)
print("FN: %d" %FN)
print("Accuracy: %f" %accuracy)
print("Precision: %f" %precision)
print("Recall: %f" %recall)
print("f-score: %f" %fscore)

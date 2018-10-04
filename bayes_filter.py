"""
Naïve Bayes models are commonly used for classification. 
They are a special form of Bayesian networks, therefore we could 
represent them with the same verbose data structure as the Belief Networks.

For a network with n binary input features X[1] to X[n], we represent the 
conditional probability tables (CPTs) that are required in the network, with the following two objects:

prior: a real number representing p(Class=true). The probability p(Class=false) 
       can be obtained by 1 - prior.

likelihood: a tuple of length n where each element is a pair of real numbers such 
            that likelihood[i][False] is p(X[i]=true|C=false) and likelihood[i][True] is p(X[i]=true|C=true ). 
            That is, likelihood contains the 2*n CPTs that are required at leaf nodes.

"""

from csv import *


def posterior(prior, likelihood, observation):
    """Returns the posterior probability of the class variable being true, 
    given the observation; that is, it returns p(Class=true|observation). 
    The argument observation is a tuple of n Booleans such that observation[i] is 
    the observed value (True or False) for the input feature X[i]. 
    The arguments prior and likelihood are as described above.
    """

    prob = []
    prob_true, prob_false = 1, 1

    for i in zip(likelihood, observation):
        prob.append(i)

    for i in prob:
        temp = i[0]
        if i[1]:
            prob_true *= (temp[True])
            prob_false *= (temp[False])
        else:
            prob_true *= (1 - temp[True])
            prob_false *= (1 - temp[False])

    true_prob = prob_true * prior
    false_prob = prob_false * (1 - prior)
    sol = true_prob / (true_prob + false_prob)

    return sol


def learn_prior(file_name, pseudo_count=0):
    """Takes the file name of the training set and an optional 
    pseudo-count parameter and returns a real number that is the 
    prior probability of spam being true. The parameter pseudo_count 
    is a non-negative integer and it will be the same for all the attributes 
    and all the values.
    """

    file = open(file_name)
    data = reader(file)
    training = []

    for line in data:
        training.append(tuple(line))

    training = training[1:]
    spam = 0

    for stuff in training:
        if stuff[-1] == '1':
            spam += 1

    spam += pseudo_count
    total = len(training) + (pseudo_count * 2)

    return spam / total


def learn_likelihood(file_name, pseudo_count=0):
    """Takes the file name of a training set (for the spam detection problem) 
    and an optional pseudo-count parameter and returns a sequence of pairs of 
    likelihood probabilities. As described in the representation of likelihood, 
    the length of the returned sequence (list or tuple) must be 12. Each element 
    in the sequence is a pair (tuple) of real numbers such that likelihood[i][False] 
    is P(X[i]=true|Spam=false) and likelihood[i][True] is P(X[i]=true|Spam=true ).
    """    
    
    file = open(file_name)
    data = reader(file)

    training, likelihood = [], []
    len_returned = 12
    ptr = 0

    for line in data:
        training.append(tuple(line))

    file.close()

    training = training[1:]

    for i in range(len_returned):
        temp = [pseudo_count, pseudo_count]
        likelihood.append(temp)

    for line in training:
        spam = int(line[-1])

        for i in range(len_returned):
            likelihood[i][spam] += int(line[i])

        ptr += spam

    for i in range(len(likelihood)):
        likelihood[i][True] /= (ptr + pseudo_count * 2)
        likelihood[i][False] /= (len(training) - ptr + pseudo_count * 2)

    return likelihood


def nb_classify(prior, likelihood, input_vector):
    """Takes the learnt prior and likelihood probabilities and classifies 
    an (unseen) input vector. The input vector will be a tuple of 12 integers 
    (each 0 or 1) corresponding to attributes X1 to X12. The function should 
    return a pair (tuple) where the first element is either "Spam" or "Not Spam" 
    and the second element is the certainty. The certainty is the (posterior) 
    probability of spam when the instance is classified as spam, or the probability 
    of 'not-spam' otherwise. If spam and 'not spam' are equally likely (i.e. p=0.5) 
    then choose 'not spam'.
    
    This is a very simple function to implement as it only wraps the posterior 
    function developed earlier.
    """

    sol = tuple() #placeholder for final solution tuple

    posterior_true = posterior(prior, likelihood, input_vector)
    posterior_false = 1 - posterior_true

    if posterior_true <= posterior_false:
        sol = ('Not Spam', posterior_false)

    else:
        sol = ("Spam", posterior_true)

    return sol


def accuracy(predicted_labels, correct_labels):
    """Returns the accuracy of a classifier based on the given arguments. 
    Both arguments are tuples of the same length and contain class labels. 
    Class labels may be of any type as long as they can be tested for equality.
    """

    match = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == correct_labels[i]:
            match += 1
    return match / len(correct_labels)

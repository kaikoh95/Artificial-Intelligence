def learn_perceptron(weights, bias, training_examples, learning_rate, 
                     max_epochs):
    """Adjusts the weights and bias so that all the training examples are 
    classified correctly (if possible). If the function succeeds, it returns
    a perceptron function that takes a vector of size n and returns either 
    0 or 1. If the training data cannot be learnt (i.e. it does not seem to 
    be linearly separable), then the learner returns None.
    """
    
    for epoch in range(1, max_epochs + 1):
        seen_error = False
        for input, target in training_examples:
            a = bias + sum([input[i] * weights[i] for i in range(len(input))])
            output = 0 if a < 0 else 1
            if output != target:
                seen_error = True
                change = learning_rate * (target - output)
                # Now update the weights and bias
                weights = [input[i] * change + weights[i] for i in range(len(weights))]
                bias += change

        if not seen_error:
            def perceptron(input_vector):
                a = bias + sum([input_vector[i] * weights[i] for i in range(len(input))])
                output = 0 if a < 0 else 1
                return output
            return perceptron

# Function that takes the epoch as input and returns the desired learning rate
def adaptive_lr(input_int):
    # input_int: the epoch that is currently being entered

    # define the learning rate (quite arbitrarily decaying)
    lr = 0.1**input_int

    #return the learning rate
    return lr
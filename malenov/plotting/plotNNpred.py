import matplotlib.pyplot as plt

def plotNNpred(pred,im_per_line,line_num,section):
    # pred: 4D-numpy array with the features in the 4th dimension
    # im_per_line: How many sub plot images to have in each row of the display
    # line_num: what xline to use as a reference
    # section: the section that was used for prediction

    # Define some initial parameters, like the number of features and plot size, etc.
    features = pred.shape[3]
    plt.figure(2, figsize=(20,20))
    n_columns = im_per_line
    n_rows = math.ceil(features / n_columns) + 1

    # Itterate through the sub-plots and fill them with the features, do some simple formatting
    for i in range(features):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Feature ' + str(i+1))
        plt.imshow(pred[:,line_num-1,:,i].T, interpolation="nearest", cmap="rainbow",\
                   extent=[section[0],section[1],-section[5],-section[4]])
        plt.colorbar()
import numpy as np

def convert(file_list, save = False, savename = 'adress_list', ex_adjust = False):
    # file_list: list of file names(strings) of adresses for the different classes
    # save: boolean that determines if a new ixz file should be saved with adresses and class numbers
    # savename: desired name of new .ixz-file
    # ex_adjust: boolean that determines if the amount of each class should be approximately equalized

    # Make an array of that holds the number of each example provided, if equalization is needed
    if ex_adjust:
        len_array = np.zeros(len(file_list),dtype = np.float32)
        for i in range(len(file_list)):
            len_array[i] = len(np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.float32))

        # Cnvert this array to a multiplier that determines how many times a given class set needs to be
        len_array /= max(len_array)
        multiplier = 1//len_array


    # preallocate space for the adr_list, the output containing all the adresses and classes
    adr_list = np.empty([0,4], dtype = np.int32)

    # Itterate through the list of example adresses and store the class as an integer
    for i in range(len(file_list)):
        a = np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.int32)
        adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

        # If desired copy the entire list by the multiplier calculated
        if ex_adjust:
            for k in range(int(multiplier[i])-1):
                adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename + '.ixz', adr_list, fmt = '%i')

    # Return the list of adresses and classes as a numpy array
    return adr_list
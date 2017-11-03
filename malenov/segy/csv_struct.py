# Convert a numpy-cube and seismic specs into a csv file/numpy-csv-format,
import numpy as np

def csv_struct(inp_numpy,spec_obj,section,inp_res=np.float64,save=False,savename='default_write.ixz'):
    # inp_numpy: array that should be converted to csv
    # spec_obj: object containing the seismic specifications, like starting depth, inlines, etc.
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # save: whether or not to save the output of the function
    # savename: what to name the newly saved csv-file

    # Get some initial parameters of the data
    (ilen,xlen,zlen) = inp_numpy.shape
    i = 0

    # Preallocate the array that we want to make
    full_np = np.empty((ilen*xlen*zlen,4),dtype = inp_res)

    # Itterate through the numpy-cube and convert each trace individually to a section of csv
    for il in range(section[0]*spec_obj.inl_step,(section[1]+1)*spec_obj.inl_step,spec_obj.inl_step):
        j = 0
        for xl in range(section[2]*spec_obj.xl_step,(section[3]+1)*spec_obj.xl_step,spec_obj.xl_step):
            # Make a list of the inline number, xline number, and depth for the given trace
            I = (il+spec_obj.inl_start)*(np.ones((zlen,1)))
            X = (xl+spec_obj.xl_start)*(np.ones((zlen,1)))
            Z = np.expand_dims(np.arange(section[4]*spec_obj.t_step+spec_obj.t_start,\
                                         (section[5]+1)*spec_obj.t_step+spec_obj.t_start,spec_obj.t_step),\
                               axis=1)

            # Store the predicted class/probability at each og the given depths of the trace
            D = np.expand_dims(inp_numpy[i,j,:],axis = 1)

            # Concatenate these lists together and insert them into the full array
            inp_li = np.concatenate((I,X,Z,D),axis=1)
            full_np[i*xlen*zlen+j*zlen:i*xlen*zlen+(j+1)*zlen,:] = inp_li
            j+=1
        i+=1

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename, full_np, fmt = '%f')

    # Return the list of adresses and classes as a numpy array
    return full_np
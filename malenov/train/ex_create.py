import random
import numpy as np
# Function for example creating
# Outputs a dictionary with pairs of cube tuples and labels
def ex_create(adr_arr,seis_arr,seis_spec,num_examp,cube_incr,inp_res=np.float64,sort_adr = False,replace_illegals = True):
    # adr_arr: 4-column numpy matrix that holds a header in the first row, then adress and class information for examples
    # seis_arr: 3D numpy array that holds a seismic cube
    # seis_spec: object that holds the specifications of the seismic cube;
    # num_examp: the number of output mini-cubes that should be created
    # cube_incr: the number of increments included in each direction from the example to make a mini-cube
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # sort_adr: boolean; whether or not to sort the randomly drawn adresses before making the example cubes
    # replace_illegals: boolean; whether or not to draw a new sample in place for an illegal one, or not

    # Define the cube size
    cube_size = 2*cube_incr+1

    # Define some boundary parameters given in the input object
    inline_start = seis_spec.inl_start
    inline_end = seis_spec.inl_end
    inline_step = seis_spec.inl_step
    xline_start = seis_spec.xl_start
    xline_end = seis_spec.xl_end
    xline_step = seis_spec.xl_step
    t_start = seis_spec.t_start
    t_end = seis_spec.t_end
    t_step = seis_spec.t_step
    num_channels = seis_spec.cube_num

    # Define the buffer zone around the edge of the cube that defines the legal/illegal adresses
    inl_min = inline_start + inline_step*cube_incr
    inl_max = inline_end - inline_step*cube_incr
    xl_min = xline_start + xline_step*cube_incr
    xl_max = xline_end - xline_step*cube_incr
    t_min = t_start + t_step*cube_incr
    t_max = t_end - t_step*cube_incr

    # Print the buffer zone edges
    print('Defining the buffer zone:')
    print('(inl_min,','inl_max,','xl_min,','xl_max,','t_min,','t_max)')
    print('(',inl_min,',',inl_max,',',xl_min,',',xl_max,',',t_min,',',t_max,')')
    # Also give the buffer values in terms of indexes
    print('(',cube_incr,',',((inline_end-inline_start)//inline_step) - cube_incr,\
          ',',cube_incr,',',((xline_end-xline_start)//xline_step) - cube_incr,\
          ',',cube_incr,',',((t_end-t_start)//t_step) - cube_incr,')')

    # We preallocate the function outputs; a list of examples and a list of labels
    examples = np.empty((num_examp,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
    labels = np.empty(num_examp,dtype=np.int8)

    # If we want to stack the examples in the third dimension we use the following example preallocation in stead
    # examples = np.empty((num_examp*(cube_size),(cube_size),(cube_size)),dtype=inp_res)

    # Generate a random list of indexes to be drawn, and make sure it only takes a legal amount of examples
    try:
        max_row_idx = len(adr_arr)-1
        rand_idx = random.sample(range(0, max_row_idx), num_examp)
        # NOTE: Could be faster to sort indexes before making examples for algorithm optimization
        if sort_adr:
            rand_idx.sort()
    except ValueError:
        print('Sample size exceeded population size.')

    # Make an iterator for when the lists should become shorter(if we have replacement of illegals or not)
    n=0
    for i in range(num_examp):
        # Get a random in-line, x-line, and time value, and store the label
        # Make sure there is room for an example at this index
        for j in range(50):
            adr = adr_arr[rand_idx[i]]
            # Check that the given example is within the legal zone
            if (adr[0]>=inl_min and adr[0]<inl_max) and \
                (adr[1]>=xl_min and adr[1]<xl_max) and \
                (adr[2]>=t_min and adr[2]<t_max):
                # Make the example for the given address
                # Convert the adresses to indexes and store the examples in the 4th dimension
                idx = [(adr[0]-inline_start)//inline_step,(adr[1]-xline_start)//xline_step,(adr[2]-t_start)//t_step]


                examples[i-n,:,:,:,:] = seis_arr[idx[0]-cube_incr:idx[0]+cube_incr+1,\
                              idx[1]-cube_incr:idx[1]+cube_incr+1,\
                              idx[2]-cube_incr:idx[2]+cube_incr+1,:]

                # Put the cube and label into the lists
                labels[i-n] = adr[-1]

                # Alternatively; stack the examples in the third dimension
                #datasets[(i-n)*(cube_size):(i-n+1)*(cube_size),:,:] = ex
                break
            else:
                # If we want to replace the illegals, draw again
                if replace_illegals:
                    rand_idx[i] = random.randint(0,max_row_idx)
                else:
                    # if not, just make the output lists shorter
                    n += 1
                    break

            if j == 50:
                # If we can't get a proper cube in 50 consequtive tries
                print('Badly conditioned dataset!')

    # Slice the data if desired
    #labels = labels[0:i-n+1]
    #examples = examples[0:i-n+1,:,:,:]

    # Return the output list/tuple (slice it if it has been shortened)
    return (examples[0:i-n+1,:,:,:,:], labels[0:i-n+1])
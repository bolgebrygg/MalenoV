# Make a function that adds another layer to a segy-cube
import segyio
import numpy as np

def segy_adder(segy_file, inp_cube, read_direc='xline', inp_res = np.float64):
    # segy_file: filename of the segy-cube to be imported
    # inp_cube: the existing cube that we should add a layer to
    # cube_num: which chronological number of cube is this
    # read_direc: which way the SEGY-cube should be read; 'xline', or 'inline'
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)

    # Make a variable to hold the shape of the input cube and preallocate a data holder
    print('Starting SEG-Y adder')
    cube_shape = inp_cube.shape
    dataholder = np.empty(cube_shape[0:-1])

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.xline.len):
                dataholder[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'xline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for xl_index in range(segyfile.iline.len):
                dataholder[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            ## NOTE: 'full' for some reason invokes float32 data
            # Potentially time this to find the "fast" direction
            #start = time.time()
            dataholder[:,:,:] = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')


        # Convert the numpy array to span between -127 and 127 and convert to the desired format
        factor = 127/np.amax(np.absolute(dataholder))
        if inp_res == np.float32:
            dataholder = (dataholder*factor)
        else:
            dataholder = (dataholder*factor).astype(dtype = inp_res)


    # Return the output object
    print('Finished adding a SEG-Y layer')
    return dataholder
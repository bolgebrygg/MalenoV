# ---- Functions for the prediction part of the program ----
# Parse the cube into sub-cubes suitable as model input
import numpy as np


def cube_parse(seis_arr: object, cube_incr: object, inp_res: object = np.float64,
               mode: object = 'trace', padding: object = False, conc: object = False,
               inline_num: object = 0, xline_num: object = 0, depth: object = 0) -> object:
    # seis_arr: a 3D numpy array that holds a seismic cube
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # mode: how much of the 3D-cube should be converted to examples ('full','xline','inline','trace', or 'point')
    # padding: do we want to pad the zone which is outside our buffer with zeroes?
    # conc: do we want to concattenate the examples, or store them in the same matrix they were fed to us?
    # inline_num: if mode is inline or point; what inline do we use?
    # xline_num: if mode is xline or point; what xline do we use?
    # depth: if mode is point; what depth do we use?
    
    # Make some initial definitions wrt. dimensionality
    inls = seis_arr.shape[0]
    xls = seis_arr.shape[1]
    zls = seis_arr.shape[2]
    num_channels = seis_arr.shape[3]
    cube_size = 2 * cube_incr + 1

    # Define the indent where the saved data will start, if user wants padding this is 0, else it is cube_incr
    if padding:
        i_re = 0
        x_re = 0
        z_re = 0
        # Preallocate the output array, if concatenated it's 4 dimensional, if not it's 6 dimensional
        if conc:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.zeros((inls * xls * zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
            elif mode == 'inline':
                examples = np.zeros((xls * zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                x_re = cube_incr
            elif mode == 'xline':
                examples = np.zeros((inls * zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
            elif mode == 'trace':
                examples = np.zeros((zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
            elif mode == 'point':
                examples = np.zeros((1, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
                z_re = cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
            # Take into account that we will have a total smaller dimensionality of data due to illegals
            inls -= 2 * cube_incr
            xls -= 2 * cube_incr
            zls -= 2 * cube_incr
        else:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.zeros((inls, xls, zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
            elif mode == 'inline':
                examples = np.zeros((1, xls, zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                x_re = cube_incr
            elif mode == 'xline':
                examples = np.zeros((inls, 1, zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
            elif mode == 'trace':
                examples = np.zeros((1, 1, zls, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
            elif mode == 'point':
                examples = np.zeros((1, 1, 1, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
                z_re = cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
    else:
        i_re = cube_incr
        x_re = cube_incr
        z_re = cube_incr
        # Preallocate the output array, if concatenated it's 5 dimensional, if not it's 7 dimensional
        if conc:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.empty(((inls - 2 * cube_incr) * (xls - 2 * cube_incr) * (zls - 2 * cube_incr), cube_size,
                                     cube_size, cube_size, num_channels),
                                    dtype=inp_res)
            elif mode == 'inline':
                examples = np.empty(
                    ((xls - 2 * cube_incr) * (zls - 2 * cube_incr), cube_size, cube_size, cube_size, num_channels),
                    dtype=inp_res)
                inline_num -= cube_incr
                xline_num = 0
                depth = 0
            elif mode == 'xline':
                examples = np.empty(
                    ((inls - 2 * cube_incr) * (zls - 2 * cube_incr), cube_size, cube_size, cube_size, num_channels),
                    dtype=inp_res)
                inline_num = 0
                xline_num -= cube_incr
                depth = 0
            elif mode == 'trace':
                examples = np.empty((zls - 2 * cube_incr, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                inline_num -= cube_incr
                xline_num -= cube_incr
                depth = 0
            elif mode == 'point':
                examples = np.empty((1, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
                inline_num -= cube_incr
                xline_num -= cube_incr
                depth -= cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
            # Take into account that we will have a total smaller dimensionality of data due to illegals
            inls -= 2 * cube_incr
            xls -= 2 * cube_incr
            zls -= 2 * cube_incr
        else:
            if mode == 'full':
                examples = np.empty(((inls - 2 * cube_incr), (xls - 2 * cube_incr), (zls - 2 * cube_incr), cube_size,
                                     cube_size, cube_size, num_channels),
                                    dtype=inp_res)
            elif mode == 'inline':
                examples = np.empty(
                    (1, (xls - 2 * cube_incr), (zls - 2 * cube_incr), cube_size, cube_size, cube_size, num_channels),
                    dtype=inp_res)
            elif mode == 'xline':
                examples = np.empty(
                    ((inls - 2 * cube_incr), 1, (zls - 2 * cube_incr), cube_size, cube_size, cube_size, num_channels),
                    dtype=inp_res)
            elif mode == 'trace':
                examples = np.empty((1, 1, (zls - 2 * cube_incr), cube_size, cube_size, cube_size, num_channels),
                                    dtype=inp_res)
            elif mode == 'point':
                examples = np.empty((1, 1, 1, cube_size, cube_size, cube_size, num_channels), dtype=inp_res)
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
    # Iterate through the desired section of the 3D input array, create the example cubes, and store them as desired
    if conc:
        # Make the cubes
        for i in range(cube_incr, inls + cube_incr):
            if mode == 'xline':
                j = xline_num
                for k in range(cube_size, zls + cube_size):
                    examples[inls * (i - i_re) + k - z_re, :, :, :, :] = \
                        seis_arr[
                        i - cube_incr + inline_num:i + cube_incr + inline_num + 1, \
                        j - cube_incr:j + cube_incr + 1, \
                        k - cube_incr + depth:k + cube_incr + depth + 1,
                        :]
            else:
                for j in range(cube_incr, xls + cube_incr):
                    for k in range(cube_incr, zls + cube_incr):
                        examples[(i - i_re) * inls + (j - x_re) * xls + k - z_re, :, :, :, :] = \
                            seis_arr[i - cube_incr + inline_num:i + cube_incr + inline_num + 1, \
                            j - cube_incr + xline_num:j + cube_incr + xline_num + 1, \
                            k - cube_incr + depth:k + cube_incr + depth + 1, :]

                        # Make sure we stop after the appropriate number of iterations
                        if mode == 'point':
                            break
                    if mode == 'point' or mode == 'trace':
                        break
                if mode == 'point' or mode == 'trace' or mode == 'inline':
                    break
    else:
        # Make the cubes
        for i in range(cube_incr, inls - cube_incr):
            if mode == 'xline':
                for k in range(cube_incr, zls - cube_incr):
                    examples[i - i_re, 1, k - z_re, :, :, :, :] = seis_arr[i - cube_incr:i + cube_incr + 1, \
                                                                  xline_num - cube_incr:xline_num + cube_incr + 1, \
                                                                  k - cube_incr:k + cube_incr + 1, :]
            else:
                for j in range(cube_incr, xls - cube_incr):
                    for k in range(cube_incr, zls - cube_incr):
                        examples[i - i_re, j - x_re, k - z_re, :, :, :, :] = seis_arr[
                                                                             i + inline_num - cube_incr:i + inline_num + cube_incr + 1, \
                                                                             j + xline_num - cube_incr:j + xline_num + cube_incr + 1, \
                                                                             k + depth - cube_incr:k + depth + cube_incr + 1,
                                                                             :]
                        # Make sure we stop after the appropriate number of iterations
                        if mode == 'point':
                            break
                    if mode == 'point' or mode == 'trace':
                        break
                if mode == 'point' or mode == 'trace' or mode == 'inline':
                    break
    # Return the list of examples stored as the desired type of array
    return examples

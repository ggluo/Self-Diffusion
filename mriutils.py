from numpy.fft import fft, fft2 as fft2d, ifft2 as ifft2d, ifft, ifftshift, fftshift
import torch.fft as torch_fft
import torch
import numpy as np
import os
import tempfile
import subprocess

def fftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x), axis=axis, norm="ortho"))


def ifftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x), axis=axis, norm="ortho"))


def fft2c(x, axes=(- 2, - 1)):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res


def ifft2c(x, axes=(- 2, - 1)):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res

def sos(x, axis=- 1):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res
    
def rsos(x, axis=0):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res

def zpad(array_in, outshape):
    import math
    #out = np.zeros(outshape, dtype=array_in.dtype)
    oldshape = array_in.shape
    assert len(oldshape)==len(outshape)
    #kspdata = np.array(kspdata)
    pad_list=[]
    for iold, iout in zip(oldshape, outshape):
        left = math.floor((iout-iold)/2)
        right = math.ceil((iout-iold)/2)
        pad_list.append((left, right))

    zfill = np.pad(array_in, pad_list, 'constant')
    return zfill

def crop(img, bounding):
    import operator
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices].copy()

def _ifft(x, dim=(-2,-1)):
    x = torch_fft.ifftshift(x, dim=dim)
    x = torch_fft.ifft2(x, dim=dim, norm='ortho')
    x = torch_fft.fftshift(x, dim=dim)
    return x

def _fft(x, dim=(-2,-1)):
    x = torch_fft.fftshift(x, dim=dim)
    x = torch_fft.fft2(x, dim=dim, norm='ortho')
    x = torch_fft.ifftshift(x, dim=dim)
    return x

def float2cplx(float_in):
    return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')

def A_adjoint(kspace, coils):
    return torch.sum(coils.conj() * _ifft(kspace), axis=0)

def A_adjoint_3D(kspace, coils):
    return torch.sum(coils.conj() * _ifft(kspace, dim=(-3,-2,-1)), axis=0)

def A_forward(x, coils):
    return _fft(coils * x)

def A_forward_3d(x, coils):
    return _fft(coils * x, dim=(-3,-2,-1))

def mri_l2_norm(x, y, mask, coils):
    x_ = torch.view_as_complex(x.permute(0,2,3,1).contiguous())
    mask = mask[None, None, ...]
    loss = torch.linalg.norm(mask * A_forward(x_[:,None,...], coils[None]) - mask * y)
    return loss

def mri_3d_l2_norm(x, y, mask, coils):
    x_ = torch.view_as_complex(x.permute(0,2,3,4,1).contiguous())
    mask = mask[None, None, ...]
    # please 
    loss = torch.linalg.norm(mask * A_forward_3d(x_[:,None,...], coils[None]) - y)
    return loss


def cplx2float(cplx_in):
    return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')

def equal_mask(nx, ny, factor):
    mask = np.zeros((nx, ny), dtype=np.float32)
    mask[:, ::factor] = 1
    return mask


def random_mask(nx, ny, factor, seed=1234):
    np.random.seed(seed)
    mask = np.zeros([nx,ny],dtype=np.float32)
    lines = np.random.choice(range(0, ny), ny//factor, replace=False)
    mask[:,lines] = 1
    return mask

def custom_probability_density(x, start, end, rho=1):
    mid = (start + end) / 2
    slope = 1 / (mid - start)
    flat_zone = 1
    
    if x > mid - flat_zone or x < mid + flat_zone:
        return slope * 1
    else:
        return slope * abs(mid - x)**(1/rho)

def generate_random_numbers(start, end, count):
    probabilities = [custom_probability_density(x, 0, end, rho=2) for x in range(0, end)]
    probabilities /= np.sum(probabilities)
    random_seed = 42
    np.random.seed(random_seed)
    numbers = np.arange(start, end)
    return np.random.choice(numbers, count, p=probabilities, replace=False)

def high_mask(nx, ny, factor):
    mask = np.zeros([nx,ny],dtype=np.complex_)
    lines = generate_random_numbers(0, ny, ny//factor)
    mask[:,lines] = 1
    return mask

def readcfl(name):
    """
    Read a cfl file and return the data as a NumPy array.

    Parameters:
    name (str): The name of the cfl file (without the extension).

    Returns:
    numpy.ndarray: The data stored in the cfl file, reshaped according to the dimensions specified in the corresponding .hdr file.

    """
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F')

def writecfl(name, array):
    """
    Write a NumPy array to a file in the .cfl format.

    Parameters:
    name (str): The base name of the output file.
    array (ndarray): The NumPy array to be written.

    Returns:
    None
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
        h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def check_out(cmd, split=True):
    """ utility to check_out terminal command and return the output"""

    strs = subprocess.check_output(cmd, shell=True).decode()

    if split:
        split_strs = strs.split('\n')[:-1]
    return split_strs

def bart(nargout, cmd, *args, return_str=False):
    """
    Call bart from the system command line.

    Args:
        nargout (int): The number of output arguments expected from the command.
        cmd (str): The command to be executed by bart.
        *args: Variable number of input arguments for the command.
        return_str (bool, optional): Whether to return the output as a string. Defaults to False.

    Returns:
        list or str: The output of the command. If nargout is 1, returns a single element list.
                     If return_str is True, returns the output as a string.

    Raises:
        Exception: If the command exits with an error.

    Usage:
        bart(<nargout>, <command>, <arguments...>)
    """

    BART_PATH = os.environ.get('BART_PATH')
    if BART_PATH is None:
        raise ValueError("BART_PATH not set")

    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return None

    name = tempfile.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = BART_PATH + ' ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = subprocess.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print("Make sure bart is properly installed")
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

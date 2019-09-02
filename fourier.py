import numpy as np
import uuid
import tensorflow as tf
from tensorflow.python.framework import ops


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, name=None, grad=None):
    if grad is None:
        return tf.py_func(func, inp, Tout, stateful=False, name=name)
    else:
        override_name = 'PyFuncStateless'

        # Need to generate a unique name to avoid duplicates:
        rnd_name = override_name + 'Grad' + str(uuid.uuid4())

        tf.RegisterGradient(rnd_name)(grad)
        g = tf.get_default_graph()

        with g.gradient_override_map({override_name: rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=False,
                              name=name)


def fft_layer(mask, op_name='forward'):
    def forward_op(imgs):
        fft_coeffs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            fft_coeffs[i] = mask * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img[..., 0]), norm='ortho'))[..., None]
        return fft_coeffs

    def adj_op(kspaces):
        imgs = np.empty_like(kspaces)
        for i, kspace in enumerate(kspaces):
            masked_fft_coeffs = mask * kspace[..., 0]
            imgs[i] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_fft_coeffs), norm='ortho'))[..., None]
        return imgs

    if op_name == 'forward':
        op = forward_op
        grad_op = adj_op
    else:
        op = adj_op
        grad_op = forward_op

    def tf_grad_op(x, dy, name):
        with tf.name_scope(name):
            out_shape = x.get_shape()
            with ops.name_scope(name + '_pyfunc', values=[x, dy]) as name_call:
                result = py_func(
                    grad_op,
                    [dy],
                    [tf.complex64],
                    name=name_call,
                )

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                result = result[0]
                result.set_shape(out_shape)
                return result

    # Def custom square function using np.square instead of tf.square:
    def tf_op(x, name=None):
        with tf.name_scope(name, op_name, values=[x]) as name:
            x_shape = x.get_shape()
            def tensorflow_layer_grad(op, grad):
                """Thin wrapper for the gradient."""
                x = op.inputs[0]
                return tf_grad_op(x, grad, name=name + '_grad')
            with ops.name_scope(name + '_pyfunc', values=[x]) as name_call:
                result = py_func(
                    op,
                    [x],
                    [tf.complex64],
                    name=name_call,
                    grad=tensorflow_layer_grad,
                )
                # We must manually set the output shape since tensorflow cannot
                # figure it out
                result = result[0]
                result.set_shape(x_shape)
                return result

    return tf_op

class FFT2:
    def __init__(self, mask):
        self.mask = mask
        self.shape = mask.shape

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 2-D image.

        Parameters
        ----------
        img: np.ndarray
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        fft_coeffs = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img), norm='ortho'))
        return self.mask * fft_coeffs

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        masked_fft_coeffs = self.mask * x
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_fft_coeffs), norm='ortho'))

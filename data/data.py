from functools import lru_cache
import glob
import random
import time

import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

from fourier import FFT2
from utils.utils import crop_center, gen_mask, normalize, normalize_instance

def from_test_file_to_mask_and_kspace(filename):
    with  h5py.File(filename) as h5_obj:
        masks = h5_obj['mask'][()]
        kspaces = h5_obj['kspace'][()]
        return masks, kspaces


@lru_cache(maxsize=128)
def from_train_file_to_image_and_kspace(filename):
    with h5py.File(filename) as h5_obj:
        images = h5_obj['reconstruction_esc'][()]
        kspaces = h5_obj['kspace'][()]
        return images, kspaces


def from_file_to_kspace(filename):
    with h5py.File(filename) as h5_obj:
        kspaces = h5_obj['kspace'][()]
        return kspaces

def fft(image):
    fourier_op = FFT2(np.ones_like(image))
    kspace = fourier_op.op(image)
    return kspace

def ifft(kspace):
    fourier_op = FFT2(np.ones_like(kspace))
    image = fourier_op.adj_op(kspace)
    return image

def zero_filled(kspace):
    fourier_op = FFT2(np.ones_like(kspace))
    im_recon = np.abs(fourier_op.adj_op(kspace))
    im_cropped = crop_center(im_recon, 320)
    return im_cropped


class fastMRI2DSequence(Sequence):
    train_modes = ('training', 'validation')

    def __init__(self, path, mode='training', af=4, norm=False):
        self.path = path
        self.mode = mode
        self.af = af
        self.norm = norm

        self.filenames = glob.glob(path + '*.h5')
        if not self.filenames:
            raise ValueError('No h5 files at path {}'.format(path))
        self.filenames.sort()
        if mode == 'testing':
            af_filenames = list()
            for filename in self.filenames:
                mask, _ = from_test_file_to_mask_and_kspace(filename)
                mask_af = len(mask) / sum(mask)
                if af == 4 and mask_af < 5.5 or af == 8 and mask_af > 5.5:
                    af_filenames.append(filename)
            self.filenames = af_filenames


    def __len__(self):
        """From fastMRI paper"""
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if self.mode in type(self).train_modes:
            return self.get_item_train(filename)
        else:
            return self.get_item_test(filename)


    def get_item_train(self, filename):
        pass

    def get_item_test(self, filename):
        pass


class SingleSliceSequence(fastMRI2DSequence):
    train_modes = ('training', 'validation')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_filename_and_position = list()
        for filename in self.filenames:
            kspaces = from_file_to_kspace(filename)
            filename_and_position = [(filename, i) for i in range(kspaces.shape[0])]
            self.idx_to_filename_and_position += filename_and_position

    def __len__(self):
        return len(self.idx_to_filename_and_position)

    def __getitem__(self, idx):
        if self.mode in type(self).train_modes:
            return self.get_item_train(idx)
        else:
            return self.get_item_test(idx)


class MaskShiftedSingleImage2DSequence(SingleSliceSequence):
    def get_item_train(self, idx):
        filename, position = self.idx_to_filename_and_position[idx]
        kspaces = from_file_to_kspace(filename)
        kspace = kspaces[position]
        k_shape = kspace.shape
        mask = gen_mask(kspace, accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        masked_kspace = kspace * fourier_mask
        masked_kspace *= np.sqrt(np.prod(k_shape))
        shifted_masked_kspace = np.fft.ifftshift(masked_kspace)
        shifted_mask = np.fft.ifftshift(fourier_mask)[None, ...]
        image = np.abs(ifft(kspace))
        image_shifted = np.fft.fftshift(image)
        image_shifted = image_shifted[None, ..., None]
        shifted_masked_kspace = shifted_masked_kspace[None, ..., None]
        return ([shifted_masked_kspace, shifted_mask], image_shifted)

    def get_item_test(self, idx):
        filename, position = self.idx_to_filename_and_position[idx]
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        kspace = kspaces[position]
        k_shape = kspace.shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        kspace *= np.sqrt(np.prod(k_shape))
        shifted_kspace = np.fft.ifftshift(kspace)
        shifted_mask = np.fft.ifftshift(fourier_mask)[None, ...]
        shifted_kspace = shifted_kspace[None, ..., None]
        return [shifted_kspace, shifted_mask]



class MaskShifted2DSequence(fastMRI2DSequence):
    def __init__(self, *args, inner_slices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices


    def get_item_train(self, filename):
        kspaces = from_file_to_kspace(filename)
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        img_batch = list()
        kspace_batch = list()
        mask_batch = list()
        for kspace in kspaces:
            masked_kspace = kspace * fourier_mask
            masked_kspace *= np.sqrt(np.prod(k_shape))
            shifted_masked_kspace = np.fft.ifftshift(masked_kspace)
            shifted_mask = np.fft.ifftshift(fourier_mask)
            image = np.abs(ifft(kspace))
            image_shifted = np.fft.fftshift(image)
            image_shifted = image_shifted[..., None]
            shifted_masked_kspace = shifted_masked_kspace[..., None]
            mask_batch.append(shifted_mask)
            kspace_batch.append(shifted_masked_kspace)
            img_batch.append(image_shifted)
        kspace_batch = np.array(kspace_batch)
        mask_batch = np.array(mask_batch)
        img_batch = np.array(img_batch)
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            kspace_batch = kspace_batch[slice_start:slice_start + self.inner_slices]
            img_batch = img_batch[slice_start:slice_start + self.inner_slices]
            mask_batch = mask_batch[slice_start:slice_start + self.inner_slices]
        return ([kspace_batch, mask_batch], img_batch)


    def get_item_test(self, filename):
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        k_shape = kspaces[0].shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        kspace_batch = list()
        mask_batch = list()
        for kspace in kspaces:
            kspace *= np.sqrt(np.prod(k_shape))
            shifted_kspace = np.fft.ifftshift(kspace)
            shifted_mask = np.fft.ifftshift(fourier_mask)
            shifted_kspace = shifted_kspace[..., None]
            mask_batch.append(shifted_mask)
            kspace_batch.append(shifted_kspace)
        kspace_batch = np.array(kspace_batch)
        mask_batch = np.array(mask_batch)
        return [kspace_batch, mask_batch]


class Untouched2DSequence(fastMRI2DSequence):
    def get_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        images = images[..., None]
        kspaces = kspaces[..., None]
        return images, kspaces

    def get_item_test(self, filename):
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        kspaces = kspaces[..., None]
        return mask, kspaces

class MaskedUntouched2DSequence(Untouched2DSequence):
    def __init__(self, *args, inner_slices=None, rand=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand

    def get_item_train(self, filename):
        images, kspaces = super(MaskedUntouched2DSequence, self).get_item_train(filename)
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
        kspaces *= mask_batch
        mask_batch = mask_batch[..., 0]
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            if self.rand:
                i_slice = random.randint(slice_start, slice_start + self.inner_slices)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
            images = images[selected_slices]
            mask_batch = mask_batch[selected_slices]
        return ([kspaces, mask_batch], images)

    def get_item_test(self, filename):
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        k_shape = kspaces[0].shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)
        return [kspaces, mask_batch]


class MaskedUntouchedSingleSlice2DSequence(SingleSliceSequence):
    def get_item_train(self, idx):
        filename, position = self.idx_to_filename_and_position[idx]
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        kspace = kspaces[position]
        k_shape = kspace.shape
        mask = gen_mask(kspace, accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        masked_kspace = kspace * fourier_mask
        image = images[position]
        kspace_batch = masked_kspace[None, ..., None].astype('complex64')
        mask_batch = fourier_mask[None, ...]
        img_batch = image[None, ..., None].astype('float32')
        return ([kspace_batch, mask_batch], img_batch)

    def get_item_test(self, idx):
        filename, position = self.idx_to_filename_and_position[idx]
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        kspace = kspaces[position]
        k_shape = kspace.shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = fourier_mask[None, ...]
        kspace_batch = kspace[None, ..., None].astype('complex64')
        return [kspace_batch, mask_batch]


class ZeroPadded2DSequence(fastMRI2DSequence):
    pad = 644
    def get_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        mask = gen_mask(kspaces[0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), kspaces[0].shape[0], axis=0)
        img_batch = list()
        z_kspace_batch = list()
        to_pad = type(self).pad - kspaces.shape[2]
        pad_seq = [(0, 0), (to_pad // 2, to_pad // 2)]
        if self.norm and self.mode == 'validation':
            means = list()
            stddevs = list()
        for kspace, image in zip(kspaces, images):
            zero_filled_rec = ifft(kspace * fourier_mask)
            if self.norm:
                zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                image = normalize(image, mean, std, eps=1e-11)
                if self.mode == 'validation':
                    means.append(mean)
                    stddevs.append(std)
            zero_filled_rec = np.pad(zero_filled_rec, pad_seq, mode='constant')
            z_kspace = fft(zero_filled_rec)
            z_kspace = z_kspace[:, :, None]
            z_kspace_batch.append(z_kspace)
            image = np.pad(image, pad_seq, mode='constant')
            image = image[..., None]
            img_batch.append(image)
        z_kspace_batch = np.array(z_kspace_batch)
        img_batch = np.array(img_batch)
        if self.norm and self.mode == 'validation':
            return z_kspace_batch, img_batch, means, stddevs
        else:
            return (z_kspace_batch, img_batch)


    def get_item_test(self, filename):
        _, kspaces = from_test_file_to_mask_and_kspace(filename)
        z_kspace_batch = list()
        means = list()
        stddevs = list()
        to_pad = type(self).pad - kspaces.shape[2]
        pad_seq = [(0, 0), (to_pad // 2, to_pad // 2)]
        for kspace in kspaces:
            zero_filled_rec = ifft(kspace)
            if self.norm:
                zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                means.append(mean)
                stddevs.append(std)
            zero_filled_rec = np.pad(zero_filled_rec, pad_seq, mode='constant')
            z_kspace = fft(zero_filled_rec)
            z_kspace = z_kspace[:, :, None]
            z_kspace_batch.append(z_kspace)
        z_kspace_batch = np.array(z_kspace_batch)
        if self.norm:
            return z_kspace_batch, means, stddevs
        else:
            return z_kspace_batch


class ZeroFilled2DSequence(fastMRI2DSequence):
    def get_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        mask = gen_mask(kspaces[0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), kspaces[0].shape[0], axis=0)
        img_batch = list()
        zero_img_batch = list()
        if self.norm and self.mode == 'validation':
            means = list()
            stddevs = list()
        for kspace, image in zip(kspaces, images):
            zero_filled_rec = zero_filled(kspace * fourier_mask)
            if self.norm:
                zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                image = normalize(image, mean, std, eps=1e-11)
                if self.mode == 'validation':
                    means.append(mean)
                    stddevs.append(std)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
            image = image[..., None]
            img_batch.append(image)
        zero_img_batch = np.array(zero_img_batch)
        img_batch = np.array(img_batch)
        if self.norm and self.mode == 'validation':
            return zero_img_batch, img_batch, means, stddevs
        else:
            return (zero_img_batch, img_batch)


    def get_item_test(self, filename):
        _, kspaces = from_test_file_to_mask_and_kspace(filename)
        zero_img_batch = list()
        means = list()
        stddevs = list()
        for kspace in kspaces:
            zero_filled_rec = zero_filled(kspace)
            if self.norm:
                zero_filled_rec, mean, std = normalize_instance(zero_filled_rec, eps=1e-11)
                means.append(mean)
                stddevs.append(std)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
        zero_img_batch = np.array(zero_img_batch)
        if self.norm:
            return zero_img_batch, means, stddevs
        else:
            return zero_img_batch


class ZeroFilled3DSequence(ZeroFilled2DSequence):
    slice_pad = 50


    def get_item_train(self, filename):
        if self.norm and self.mode == 'validation':
            z_kspace_batch, img_batch, means, stddevs = super(ZeroFilled3DSequence, self).get_item_train(filename)
        else:
            z_kspace_batch, img_batch = super(ZeroFilled3DSequence, self).get_item_train(filename)
        # to_pad = type(self).slice_pad - z_kspace_batch.shape[0]
        # pad_seq = [(to_pad // 2 + to_pad % 2, to_pad // 2), (0, 0), (0, 0), (0, 0)]
        # z_kspace_batch = np.pad(z_kspace_batch, pad_seq, mode='constant')
        # img_batch = np.pad(img_batch, pad_seq, mode='constant')
        z_kspace_batch = z_kspace_batch[None, ...]
        img_batch = img_batch[None, ...]
        if self.norm and self.mode == 'validation':
            return z_kspace_batch, img_batch, means, stddevs
        else:
            return z_kspace_batch, img_batch


    def get_item_test(self, filename):
        if self.norm:
            z_kspace_batch, means, stddevs = super(ZeroFilled3DSequence, self).get_item_test(filename)
        else:
            z_kspace_batch = super(ZeroFilled3DSequence, self).get_item_test(filename)
        to_pad = type(self).slice_pad - z_kspace_batch.shape[0]
        pad_seq = [(to_pad // 2 + to_pad % 2, to_pad // 2), (0, 0), (0, 0), (0, 0)]
        z_kspace_batch = np.pad(z_kspace_batch, pad_seq, mode='constant')
        z_kspace_batch = z_kspace_batch[None, ...]
        if self.norm:
            return z_kspace_batch, means, stddevs, pad_seq
        else:
            return z_kspace_batch, pad_seq


class fastMRI2DAllLoadedSequence(Sequence):
    train_modes = ('training', 'validation')

    def __init__(self, path, mode='training', af=4, norm=False, inner_slices=None):
        self.path = path
        self.mode = mode
        self.af = af
        self.norm = norm
        self.inner_slices = inner_slices

        filenames = glob.glob(path + '*.h5')
        if not filenames:
            raise ValueError('No h5 files at path {}'.format(path))
        filenames.sort()
        if mode == 'testing':
            af_filenames = list()
            for filename in filenames:
                mask, _ = from_test_file_to_mask_and_kspace(filename)
                mask_af = len(mask) / sum(mask)
                if af == 4 and mask_af < 5.5 or af == 8 and mask_af > 5.5:
                    af_filenames.append(filename)
            filenames = af_filenames
        self.data = []
        for filename in filenames:
            self.data.append(self.load_item(filename))

    def load_item(self, filename):
        if self.mode in type(self).train_modes:
            return self.load_item_train(filename)
        else:
            return self.load_item_test(filename)

    def load_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
            images = images[selected_slices]
        return images, kspaces

    def load_item_test(self, filename):
        mask, kspaces = from_test_file_to_mask_and_kspace(filename)
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
        return mask, kspaces

    def __len__(self):
        """From fastMRI paper"""
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode in type(self).train_modes:
            return self.get_item_train(idx)
        else:
            return self.get_item_test(idx)


    def get_item_train(self, idx):
        pass

    def get_item_test(self, idx):
        pass


class MaskedUntouched2DAllLoadedSequence(fastMRI2DAllLoadedSequence):
    def __init__(self, *args, rand=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand = rand

    def get_item_train(self, idx):
        images, kspaces = self.data[idx]
        images = images[..., None]
        kspaces = kspaces[..., None]
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
        kspaces *= mask_batch
        mask_batch = mask_batch[..., 0]
        if self.rand:
            i_slice = random.randint(0, self.inner_slices)
            kspaces = kspaces[i_slice:i_slice+1]
            images = images[i_slice:i_slice+1]
            mask_batch = mask_batch[i_slice:i_slice+1]
        return ([kspaces, mask_batch], images)

    def get_item_test(self, idx):
        mask, kspaces = self.data[idx]
        kspaces = kspaces[..., None]
        k_shape = kspaces[0].shape
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)
        return [kspaces, mask_batch]


class MaskedUntouched2DRealSequence(MaskedUntouched2DSequence):
    def get_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        real_images = np.empty_like(kspaces)
        for i, kspace in enumerate(kspaces):
            real_images[i] = ifft(kspace)
        images = images[..., None]
        real_images = real_images[..., None]
        kspaces = kspaces[..., None]
        k_shape = kspaces[0].shape
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
        kspaces *= mask_batch
        mask_batch = mask_batch[..., 0]
        if self.inner_slices is not None:
            n_slices = len(kspaces)
            slice_start = n_slices // 2 - self.inner_slices // 2
            if self.rand:
                i_slice = random.randint(slice_start, slice_start + self.inner_slices)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_start + self.inner_slices)
            kspaces = kspaces[selected_slices]
            images = images[selected_slices]
            mask_batch = mask_batch[selected_slices]
            real_images = real_images[selected_slices]
        return ([kspaces, mask_batch], images, real_images)

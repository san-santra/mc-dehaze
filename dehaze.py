import skimage.io as skimio
from skimage import img_as_float
import os
import sys
import time

# local
from unet import get_gradmodel
from model import get_gen_model
from contrast_stretch import contrast_stretch

# for CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python dehaze.py <hazy_images_path> <output_path>'
        print 'falling back to default directories'
        hazy_path = './hazy_images'
        out_path = './out'
    else:
        hazy_path = sys.argv[1]
        out_path = sys.argv[2]

    wt_path = './model_wt/gen_wt.h5'
    grad_wt_path = './model_wt/grad_wt.h5'

    hazy_files = sorted(os.listdir(hazy_path))
    enh_patch_size = [128, 256]

    # model
    model = get_gen_model()
    grad_model = get_gradmodel()
    
    model.load_weights(wt_path)
    grad_model.load_weights(grad_wt_path)

    for i in xrange(len(hazy_files)):
        sys.stdout.write('[{}/{}] - {}'.format(i+1, len(hazy_files),
                                               hazy_files[i]))
        sys.stdout.flush()

        hazy_im = img_as_float(skimio.imread(os.path.join(hazy_path,
                                                          hazy_files[i])))
            
        start = time.time()
        im2 = contrast_stretch(hazy_im, enh_patch_size[0])
        im3 = contrast_stretch(hazy_im, enh_patch_size[1])
            
        out_temp = model.predict([hazy_im[None, ...], im2[None, ...],
                                  im3[None, ...]])

        out = grad_model.predict([hazy_im[None, ...], out_temp])

        end = time.time()

        sys.stdout.write(' |time: {} s\n'.format(end - start))
        sys.stdout.flush()
        out_name = os.path.splitext(hazy_files[i])[0]
        skimio.imsave(os.path.join(out_path, out_name+'_out.png'), out[0])

from torchvision.transforms import Lambda
from PIL import Image
import numpy as np

def get_gaussian_blur_transform(std):
# std: 1 to 3 or 4
    return Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(std)))

def get_poisson_blur_transform(PEAK):
# PEAK: 0 to 1 defining intensity of noise
    def poisson_blur_f(image, PEAK):
        noisy = np.random.poisson(np.array(image) / 255.0 * PEAK) / PEAK * 255  # noisy image
        return Image.fromarray(noisy)
    return Lambda(lambda img: poisson_blur_f(image, PEAK))

def get_gaussian_additive_transform(std, percent):
# percent: 0 to 1
    gauss_tr = get_gaussian_blur_transform(std)
    
    def gaussian_noise_f(image, std, percent):
        noise = np.array(gauss_tr(image))
        return Image.fromarray(image + percent*noise)
    
    return Lambda(lambda img: gaussian_noise_f(img, std, percent))

def get_poisson_additive_transform(PEAK, percent):
# percent: 0 to 1
    poiss_tr = get_poisson_blur_transform(PEAK)

    def poisson_noise_f(image, PEAK, percent):
        noise = np.array(poiss_tr(image))
        return Image.fromarray(image + percent*noise)
    return Lambda(lambda img: poisson_noise_f(img, PEAK, percent))

def get_salt_and_pepper_additive_transform(percent):
# percent: 0 to 1
    
    def salt_and_pepper_f(image, amount, s_vs_p=0.5):
        image = np.array(image)
        row,col,ch = image.shape
        out = image.copy()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
        out[tuple(coords)] = 0 

        return Image.fromarray(out)

    return Lambda(lambda img: salt_and_pepper_f(img, percent))

if __name__=='__main__':
    
    tr = get_salt_and_pepper_additive_transform(0.01)
    img = Image.open('/home/hans/sample9.jpg')

    trimg = tr(img)
    print (trimg.size)

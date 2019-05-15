from torchvision.transforms import Lambda
from PIL import Image, ImageFilter
import numpy as np

def GaussianBlur(std):
# std: 1 to 2 or 3
    return Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(std)))

def PoissonBlur(peak):
# PEAK: 0 to 1 defining intensity of noise
    def poisson_blur_f(image, peak):
        noisy = np.random.poisson(np.array(image) / 255.0 * peak) / peak * 255  # noisy image
        return Image.fromarray(noisy.astype(np.uint8))
    return Lambda(lambda img: poisson_blur_f(img, peak))

def GaussianAdditiveNoise(std, percent):
# percent: 0 to 1
    
    def gaussian_noise_f(image, std, percent):
        image = np.array(image)
        noise = np.random.normal(loc=0, scale=std, size=image.shape)
        return Image.fromarray(np.array(image + percent*noise, dtype=np.uint8))
    
    return Lambda(lambda img: gaussian_noise_f(img, std, percent))

def PoissonAdditiveNoise(peak, percent):
# percent: 0 to 1 (1)
# PEAK: 0.1 to 0.9
    poiss_tr = PoissonBlur(peak)

    def poisson_noise_f(image, percent):
        noise = np.array(poiss_tr(image))
        image = image + percent*noise
        return Image.fromarray(image.astype(np.uint8))
    return Lambda(lambda img: poisson_noise_f(img, percent))

def SaltandPepperAdditiveNoise(percent):
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
    
    tr = GaussianBlur(0.01)
    img = Image.open('/home/hans/sample9.jpg')
    
    img.save('sample.jpg')
    trimg = tr(img)
    trimg.save('sample_noise.jpg')

import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from scipy.fftpack import dct, idct


def mse(img_ref, img):
    return np.mean((img_ref - img)**2)

def psnr(img, img_ref):
    num = np.linalg.norm(img_ref, ord = np.inf)
    den = np.linalg.norm(img_ref - img)/np.sqrt(img_ref.shape[0])
    return 20*np.log10(num/den)

def relative_change(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

def sampling_mask(W):
    return (W.T @ np.ones(W.shape[1])).reshape((256,256))

def objective_function(idx, x, m, lammbda, p, epsilon=1e-6):
    diff = x[idx] - m
    term1 = np.sum(diff * diff)  
    term2 = lammbda * np.sum(np.abs(epsilon + dct(x, norm = "ortho")**2)**p)
    return term1 + term2

def line_plot(list, titlex, titley):
    x = range(1,len(list)+1)
    plt.figure(figsize=(8,4))
    plt.plot(x, list, marker = '*')
    plt.title(titlex+"vs"+titley)
    plt.xlabel(titley)
    plt.ylabel(titlex)
    plt.grid(True)
    plt.show()


def sampling(img, r):
    img_vector = img.flatten()
    N = img.shape[0]*img.shape[1]
    M = int(np.round(r*N))
    idx = np.random.choice(N, size=M, replace=False)
    Wx_ = img_vector[idx]
    Wx_l2_2 = (np.linalg.norm(Wx_))**2
    sigma = np.sqrt(Wx_l2_2/(1000*M))
    noise = np.random.normal(0, sigma, (M,))

    return (Wx_ + noise), idx

def conjugate_gradient(Q_operator, b, x_0):
    g_0 = (Q_operator(x_0) - b)
    d_0 = -g_0
    iteration = 0
    while True:
        Q_operator_d0 = Q_operator(d_0)
        den = d_0.T @ (Q_operator_d0)
        alpha_k = -(g_0.T @ d_0) / (den)
        x_0 = x_0 + (alpha_k*d_0)
        g_0 = (Q_operator(x_0) - b)
        beta_k = (g_0.T @ (Q_operator_d0)) / (den)
        d_0 = -g_0 + (beta_k * d_0)
        iteration += 1
        if np.linalg.norm(g_0) < 1e-6:
            break
    return x_0, iteration



def mm_cg(img, m, idx, lammbda, p, epsilon = 1e-6):
    N = img.shape[0]*img.shape[1]
    x_0 = np.zeros((N,))
    x_0[idx] = m
    x_k = x_0
    mask = np.zeros((N,))
    mask[idx] = 1 
    list_iteration_cg = []
    list_objective_function = []
    list_relative_error = []
    while True:
        y_k = dct(x_k, norm= "ortho")
        w_k = p*((epsilon + y_k**2)**(p-1))
        def Q_operator(z):
            dct_z = dct(z, norm= "ortho")
            return mask*z + lammbda*idct(w_k*dct_z, norm= "ortho")
        x_cg, iteration_cg = conjugate_gradient(Q_operator, x_0, x_k)
        relative_error = relative_change(x_cg, x_k)
        list_objective_function.append(objective_function(idx,x_cg,m,lammbda,p))
        list_iteration_cg.append(iteration_cg)
        list_relative_error.append(relative_error)
        if  relative_error < 1e-4:
            line_plot(list_iteration_cg, "No. of MM Step", "No. of CG Iterations")
            line_plot(list_objective_function, "No. of Iteration", "Objective Function")
            line_plot(list_relative_error, "No. of Iteration", "Relative Error")
            break
        else: 
            x_k = x_cg
            print("Iteration done...")
            continue
    return x_cg


if __name__ == "__main__":
    image = img_as_float(skimage.io.imread("cameraman.tif"))
    image_sampled, idx = sampling(image, 0.2)
    N = image.shape[0]*image.shape[1]
    image_sampled_mapped = np.zeros((N,))
    image_sampled_mapped[idx] = image_sampled
    print("Sampling Done!")
    reconstructed_image = mm_cg(image,image_sampled, idx, 1e-3, 0.4)
    plt.figure(figsize=(12,36))
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(image, cmap = "gray")
    plt.subplot(1,3,2)
    plt.title("Sampled Noisy Image")
    plt.imshow(image_sampled_mapped.reshape((256,256)), cmap = "gray")
    plt.subplot(1,3,3)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image.reshape((256,256)), cmap = "gray")
    plt.show()



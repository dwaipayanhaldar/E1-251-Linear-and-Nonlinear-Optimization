import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from scipy.fftpack import dct, idct
import time 


def mse(img_ref, img):
    return np.mean((img_ref - img)**2)

def psnr(img, img_ref):
    num = np.linalg.norm(img_ref, ord = np.inf)
    den = np.linalg.norm(img_ref - img)/np.sqrt(img_ref.shape[0]*img_ref.shape[1])
    return 20*np.log10(num/den)

def relative_change(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

def sampling_mask(N,idx):
    sampling_mask = np.zeros((N,))
    sampling_mask[idx] = 1
    return sampling_mask

def objective_function(idx, x, m, lammbda, p, epsilon=1e-6):
    diff = x[idx] - m
    term1 = np.sum(diff * diff)  
    term2 = lammbda * np.sum(np.abs(epsilon + dct(x, norm = "ortho")**2)**p)
    return term1 + term2

def Wtranspose(N,x, idx):
    Wtransposex = np.zeros((N,))
    Wtransposex[idx] = x
    return Wtransposex

def line_plot(list, titlex, titley):
    x = range(1,len(list)+1)
    plt.figure(figsize=(8,4))
    plt.plot(x, list)
    plt.title(titlex+" vs "+titley)
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



def mm_cg(N, m, idx, lammbda, p, epsilon = 1e-6):
    x_0 = Wtranspose(N, m, idx)
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
            break
        else: 
            x_k = x_cg
            continue
    return x_cg, list_objective_function, list_iteration_cg, list_relative_error


def reconstruct(img, r):
    N = img.shape[0]*img.shape[1] #Total vector size
    m,idx = sampling(img, r) #Vector m
    sampling_mask_image = sampling_mask(N, idx).reshape((img.shape[0],img.shape[1])) #Sampling Mask to print
    noisy_image = Wtranspose(N,m,idx).reshape((img.shape[0],img.shape[1])) #Noisy Image to print

    reconstructed_image_p = []
    list_of_p = []
    list_iter_p = []
    list_re_p = []
    psnr_list_p = []
    start_time_p = time.time()
    for p in [0.3,0.4,0.5]:
        reconstructed_image_l = []
        runtime_list = []
        psnr_list = []
        list_of_l = []
        list_iter_l = []
        list_re_l = []
        for l in np.logspace(-4,0,num=5):
            start_time = time.time()
            reconstructed_image, list_of, list_iter,list_re = mm_cg(N,m,idx,l,p)
            reconstructed_image = reconstructed_image.reshape((256,256))
            end_time = time.time()
            psnr_list.append(psnr(reconstructed_image, img))
            runtime_list.append((end_time-start_time))
            reconstructed_image_l.append(reconstructed_image)
            list_of_l.append(list_of)
            list_iter_l.append(list_iter)
            list_re_l.append(list_re)
            print("One lambda completed...")
        
        index = np.argmax(psnr_list)
        lammbda = np.logspace(-4,0,num=5)[index]
        print(f"Best lambda for p = {p} is {lammbda}")
        print(f"PSNR for that best lambda = {lammbda} and p = {p} is {psnr_list[index]}")
        print(f"Runtime for that best lambda = {lammbda} and p = {p} is {runtime_list[index]}")
        reconstructed_image_p.append(reconstructed_image_l[index])
        list_of_p.append(list_of_l[index])
        list_iter_p.append(list_iter_l[index])
        list_re_p.append(list_re_l[index])
        psnr_list_p.append(psnr_list)


    end_time_p =  time.time()
    print("Total runtime for one r:", (end_time_p-start_time_p))

    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.title("Original Image")
    plt.imshow(img, cmap= "gray")
    plt.subplot(2,3,2)
    plt.title("Sampling Mask")
    plt.imshow(sampling_mask_image, cmap= "gray")
    plt.subplot(2,3,3)
    plt.title("Noisy Observation")
    plt.imshow(noisy_image, cmap= "gray")
    plt.subplot(2,3,4)
    plt.title("Reconstructed Image(p = 0.3)")
    plt.imshow(reconstructed_image_p[0], cmap= "gray")
    plt.subplot(2,3,5)
    plt.title("Reconstructed Image(p=0.4)")
    plt.imshow(reconstructed_image_p[1], cmap= "gray")
    plt.subplot(2,3,6)
    plt.title("Reconstructed Image(p=0.5)")
    plt.imshow(reconstructed_image_p[2], cmap= "gray")
    plt.show()

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1); plt.title(r"PSNR vs $\lambda$ for p=$0.3$");plt.semilogx(np.logspace(-4,0,num=5), psnr_list_p[0]);plt.xlabel(r"$\lambda$");plt.ylabel("PSNR")
    plt.subplot(1,3,2); plt.title(r"PSNR vs $\lambda$ for p=$0.4$");plt.semilogx(np.logspace(-4,0,num=5), psnr_list_p[1]);plt.xlabel(r"$\lambda$");plt.ylabel("PSNR")
    plt.subplot(1,3,3); plt.title(r"PSNR vs $\lambda$ for p=$0.5$");plt.semilogx(np.logspace(-4,0,num=5), psnr_list_p[2]);plt.xlabel(r"$\lambda$");plt.ylabel("PSNR")
    plt.show()

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1); plt.title(r"Objective Function vs No. of Iterations(k) for p=$0.3$");plt.plot(range(1,len(list_of_p[0])+1), list_of_p[0]);plt.xlabel("no. of iterations(k)");plt.ylabel("Objective Function")
    plt.subplot(1,3,2); plt.title(r"No. of CG iteration per MM vs No. of Iterations(k) for p=$0.3$");plt.plot(range(1,len(list_iter_p[0])+1), list_iter_p[0]);plt.xlabel("no. of iterations(k)");plt.ylabel("No. of CG iteration per MM")
    plt.subplot(1,3,3); plt.title(r"Relative Error vs No. of Iterations(k) for p=$0.3$");plt.plot(range(1,len(list_re_p[0])+1), list_re_p[0]);plt.xlabel("no. of iterations(k)");plt.ylabel("Relative Error")
    plt.show()

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1); plt.title(r"Objective Function vs No. of Iterations(k) for p=$0.4$");plt.plot(range(1,len(list_of_p[1])+1), list_of_p[1]);plt.xlabel("no. of iterations(k)");plt.ylabel("Objective Function")
    plt.subplot(1,3,2); plt.title(r"No. of CG iteration per MM vs No. of Iterations(k) for p=$0.4$");plt.plot(range(1,len(list_iter_p[1])+1), list_iter_p[1]);plt.xlabel("no. of iterations(k)");plt.ylabel("No. of CG iteration per MM")
    plt.subplot(1,3,3); plt.title(r"Relative Error vs No. of Iterations(k) for p=$0.4$");plt.plot(range(1,len(list_re_p[1])+1), list_re_p[1]);plt.xlabel("no. of iterations(k)");plt.ylabel("Relative Error")
    plt.show()

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1); plt.title(r"Objective Function vs No. of Iterations(k) for p=$0.5$");plt.plot(range(1,len(list_of_p[2])+1), list_of_p[2]);plt.xlabel("no. of iterations(k)");plt.ylabel("Objective Function")
    plt.subplot(1,3,2); plt.title(r"No. of CG iteration per MM vs No. of Iterations(k) for p=$0.5$");plt.plot(range(1,len(list_iter_p[2])+1), list_iter_p[2]);plt.xlabel("no. of iterations(k)");plt.ylabel("No. of CG iteration per MM")
    plt.subplot(1,3,3); plt.title(r"Relative Error vs No. of Iterations(k) for p=$0.5$");plt.plot(range(1,len(list_re_p[2])+1), list_re_p[2]);plt.xlabel("no. of iterations(k)");plt.ylabel("Relative Error")
    plt.show()


    return reconstructed_image_p


if __name__ == "__main__":
    image = img_as_float(skimage.io.imread("images/cameraman.tif"))
    # image_sampled, idx = sampling(image, 0.1)
    # N = image.shape[0]*image.shape[1]
    # image_sampled_mapped = np.zeros((N,))
    # image_sampled_mapped[idx] = image_sampled
    # print("Sampling Done!")
    # reconstructed_image = mm_cg(image,image_sampled, idx, 1e-3, 0.4)
    # plt.figure(figsize=(12,36))
    # plt.subplot(1,3,1)
    # plt.title("Original Image")
    # plt.imshow(image, cmap = "gray")
    # plt.subplot(1,3,2)
    # plt.title("Sampled Noisy Image")
    # plt.imshow(image_sampled_mapped.reshape((256,256)), cmap = "gray")
    # plt.subplot(1,3,3)
    # plt.title("Reconstructed Image")
    # plt.imshow(reconstructed_image.reshape((256,256)), cmap = "gray")
    # plt.show()
    reconstruct(image, 0.2)
    # plt.imshow(sampling_mask(image, idx), cmap="gray")
    # plt.show()


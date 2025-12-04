import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import time 

def dct2(x_flat, shape):
    """
    Performs 2D Orthonormal DCT on a flattened vector.
    Reshapes to 2D -> Applies DCT -> Flattens back.
    """
    x = x_flat.reshape(shape)
    return dct(dct(x.T, norm='ortho').T, norm='ortho').flatten()

def idct2(x_flat, shape):
    """
    Performs 2D Orthonormal IDCT on a flattened vector.
    """
    x = x_flat.reshape(shape)
    return idct(idct(x.T, norm='ortho').T, norm='ortho').flatten()

#  Metrics 
def mse(img_ref, img):
    return np.mean((img_ref - img)**2)

def psnr(img, img_ref):
    img = img.flatten()
    img_ref = img_ref.flatten()
    # Assuming pixels are float [0,1], peak is 1.0. 
    # If image is 0-255, change 1.0 to 255.0
    num = 1.0 
    if np.max(img_ref) > 1.1: num = 255.0
    
    den = np.linalg.norm(img_ref - img) / np.sqrt(img_ref.shape[0])
    if den == 0: return 100
    return 20 * np.log10(num / den)

def relative_change(x, y):
    x = x.flatten()
    y = y.flatten()
    if np.linalg.norm(y) == 0: return 0
    return np.linalg.norm(x - y) / np.linalg.norm(y)

#  Sampling 
def sampling_mask(N, idx):
    mask = np.zeros((N,))
    mask[idx] = 1
    return mask

def Wtranspose(N, x, idx):
    out = np.zeros((N,))
    out[idx] = x
    return out

def sampling(img, r):
    """
    Argument: img and sampling ratio r
    Output: Sampled Noisy Vector m and indices idx
    """
    img_vector = img.flatten() 
    N = img_vector.shape[0] 
    M = int(np.round(r * N)) 
    
    idx = np.random.choice(N, size=M, replace=False) 
    Wx_ = img_vector[idx] 

    # Noise addition
    Wx_l2_2 = (np.linalg.norm(Wx_))**2 
    sigma = np.sqrt(Wx_l2_2 / (1000 * M)) # Approx 30dB SNR
    noise = np.random.normal(0, sigma, (M,)) 
    return (Wx_ + noise), idx 

#  Optimization Functions 

def objective_function(idx, x, m, lammbda, p, img_shape, epsilon=1e-6):
    diff = x[idx] - m
    term1 = np.sum(diff**2)
    y = dct2(x, img_shape)
    term2 = lammbda * np.sum((epsilon + np.abs(y))**p)
    return term1 + term2

def soft_threshold(z, threshold):
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def admm_inner_solver(N, m, idx, w_k, x_init, y_prev, lammbda, rho, img_shape, max_admm_iter=1000, tol=1e-4):
    """
    Solves the Weighted L1 subproblem using ADMM.
    """
    x = x_init.copy()
    v = y_prev.copy()
    u = np.zeros_like(v)     
    
    mask = np.zeros(N)
    mask[idx] = 1.0       
    
    m_full = np.zeros(N)
    m_full[idx] = m       

    iteration = 0
    
    for t in range(max_admm_iter):
        v_prev = v.copy()
        
        #  x-update 
        s_t = idct2(v - u, img_shape)
        x = mask * ((m_full + rho * s_t) / (1 + rho)) + (1 - mask) * s_t
        
        #  v-update (DCT Domain)
        dct_x = dct2(x, img_shape)
        z_t = dct_x + u
        
        # Weighted Soft Thresholding
        threshold = (lammbda / rho) * w_k
        v = soft_threshold(z_t, threshold)
        
        #  u-update (Dual) 
        u = u + dct_x - v
        
        #  Convergence Check 
        r_norm = np.linalg.norm(dct_x - v)      # Primal residual
        s_norm = np.linalg.norm(rho * (v - v_prev)) # Dual residual
        
        iteration = t + 1
        if r_norm < tol and s_norm < tol:
            break
            
    return x, v, iteration

def mm_admm(N, m, idx, rho, lammbda, p, img_shape, max_mm_iter=100, tol=1e-3, epsilon=1e-6):
    # Initialize x with zero-filled observations
    x = np.zeros(N)
    x[idx] = m
    
    # Initial DCT coefficients
    v = dct2(x, img_shape)
    
    list_iteration_admm = [] 
    list_objective_function = [] 
    list_relative_error = [] 
    
    for k in range(max_mm_iter):
        x_prev = x.copy()
        y_prev = v
        
        #  MM Weight Update 
        w_k = p * (np.abs(y_prev) + epsilon)**(p - 1)
            
        #  ADMM Inner Solver 
        x, v, inner_iters = admm_inner_solver(
            N, m, idx, w_k, x_prev, y_prev, lammbda, rho, img_shape
        )
        
        #  Convergence Check 
        norm_x = np.linalg.norm(x)
        rel_error = np.linalg.norm(x - x_prev) / max(1, norm_x)

        # Logging
        obj_val = objective_function(idx, x, m, lammbda, p, img_shape)
        list_objective_function.append(obj_val)
        list_iteration_admm.append(inner_iters)
        list_relative_error.append(rel_error)
        
        if rel_error < tol:
            break
            
    return x, list_objective_function, list_iteration_admm, list_relative_error

def reconstruct(img, r):
    img_shape = img.shape
    N = img_shape[0] * img_shape[1] 
    
    # Generate Samples
    m, idx = sampling(img, r) 
    
    # For Visualization
    sampling_mask_image = sampling_mask(N, idx).reshape(img_shape)
    noisy_image = Wtranspose(N, m, idx).reshape(img_shape)

    # Storage for results across different p values
    results_by_p = [] # Will store dictionary of results for each p
    
    p_values = [0.5, 0.7, 0.9] # As per Algorithm 1
    
    print(f"Starting Reconstruction for r={r}...")
    start_time_total = time.time()
    
    for p in p_values:
        print(f"\n Processing p={p} ")
        
        # Initialize lists to store sensitivity data for this p
        psnrs_for_plot = []
        lambdas_for_plot = []
        
        best_metrics = {
            "psnr": -np.inf,
            "lambda": None,
            "image": None,
            "obj_hist": [],
            "iter_hist": [],
            "err_hist": []
        }
        
        # Grid search for Lambda
        lambda_grid = np.logspace(-4, 0, num=5) 
        
        for l in lambda_grid:
            start_time_lambda = time.time()
            
            # Run the algorithm
            x_rec_flat, obj_hist, iter_hist, err_hist = mm_admm(
                N, m, idx, rho=4.0, lammbda=l, p=p, img_shape=img_shape
            )
            
            end_time_lambda = time.time()
            runtime_lambda = end_time_lambda - start_time_lambda
            
            # Reshape and compute metrics
            x_rec = x_rec_flat.reshape(img_shape)
            curr_psnr = psnr(x_rec, img)
            curr_rel_error = relative_change(x_rec, img)
            
            # Store data for plotting
            psnrs_for_plot.append(curr_psnr)
            lambdas_for_plot.append(l)
            
            # Update best metrics if this is the best PSNR so far
            if curr_psnr > best_metrics["psnr"]:
                best_metrics["psnr"] = curr_psnr
                best_metrics["lambda"] = l
                best_metrics["image"] = x_rec
                best_metrics["obj_hist"] = obj_hist
                best_metrics["iter_hist"] = iter_hist
                best_metrics["err_hist"] = err_hist
                best_metrics["rel_error"] = curr_rel_error
                best_metrics["runtime"] = runtime_lambda

        # Save the sensitivity lists into the dictionary
        best_metrics["psnr_list_for_plot"] = psnrs_for_plot
        best_metrics["lambda_list_for_plot"] = lambdas_for_plot
        
        results_by_p.append(best_metrics)
        print(f"Best for p={p}: Lambda={best_metrics['lambda']:.1e}, PSNR={best_metrics['psnr']:.2f}")
        print(f"Runtime for best lambda: {best_metrics['runtime']:.4f}s, Relative error: {best_metrics['rel_error']:.4e}")


    end_time_total = time.time()
    print(f"\nTotal runtime: {end_time_total - start_time_total:.2f}s")

    #  Plotting 
    
    # 1. Visual Comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Sampling Mask")
    plt.imshow(sampling_mask_image, cmap="gray")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title(f"Noisy Obs (r={r})")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis('off')
    
    for i, p in enumerate(p_values):
        plt.subplot(2, 3, 4 + i)
        plt.title(f"Rec (p={p})\nPSNR: {results_by_p[i]['psnr']:.2f}dB")
        plt.imshow(results_by_p[i]['image'], cmap="gray")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 2. Convergence Metrics
    for i, p in enumerate(p_values):
        res = results_by_p[i]
        k_range = range(1, len(res['obj_hist']) + 1)
        
        plt.figure(figsize=(18, 4))
        plt.suptitle(f"Convergence Metrics for p={p} (Best Lambda={res['lambda']:.1e})")
        
        plt.subplot(1, 3, 1)
        plt.plot(k_range, res['obj_hist'])
        plt.title("Objective Function vs Outer Iter")
        plt.xlabel("Outer Iter (k)")
        plt.ylabel("J(x)")
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(k_range, res['iter_hist'])
        plt.title("ADMM Iterations per MM Step")
        plt.xlabel("Outer Iter (k)")
        plt.ylabel("Inner Iters")
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(k_range, res['err_hist'])
        plt.title("Relative Error vs Outer Iter")
        plt.xlabel("Outer Iter (k)")
        plt.yscale('log')
        plt.ylabel("||x_k+1 - x_k|| / ||x_k||")
        plt.grid(True)
        
        plt.show()

    # 3. Lambda Sensitivity Plot (New)
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(p_values):
        res = results_by_p[i]
        plt.semilogx(
            res["lambda_list_for_plot"], 
            res["psnr_list_for_plot"], 
            'o-', 
            linewidth=2, 
            label=f'p={p}'
        )
    
    plt.title("Parameter Sensitivity: PSNR vs $\lambda$")
    plt.xlabel("$\lambda$ (log scale)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.show()

    return [res['image'] for res in results_by_p]




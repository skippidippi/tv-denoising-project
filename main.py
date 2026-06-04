import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from skimage.transform import resize
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================
# Вспомогательные функции для TV
# ===============================

def gradient(img):
    """Вычисляет дискретный градиент изображения с периодическими граничными условиями"""
    grad_x = np.roll(img, -1, axis=1) - img
    grad_y = np.roll(img, -1, axis=0) - img
    return grad_x, grad_y

def divergence(grad_x, grad_y):
    """Вычисляет дивергенцию векторного поля (сопряженный оператор к градиенту)"""
    div_x = grad_x - np.roll(grad_x, 1, axis=1)
    div_y = grad_y - np.roll(grad_y, 1, axis=0)
    return div_x + div_y

def tv_norm(img):
    """Вычисляет анизотропную норму TV"""
    grad_x, grad_y = gradient(img)
    return np.sum(np.sqrt(grad_x**2 + grad_y**2))

def objective_function_tv(x, y, lambda_val):
    """Целевая функция для TV-регуляризации"""
    data_fidelity = 0.5 * np.sum((x - y)**2)
    regularization = lambda_val * tv_norm(x)
    return data_fidelity + regularization

def prox_tv(y, weight, num_iter_prox=25):
    """Проксимальный оператор для TV-нормы"""
    p_x = np.zeros_like(y)
    p_y = np.zeros_like(y)
    tau = 0.125
    for _ in range(num_iter_prox):
        div_p = divergence(p_x, p_y)
        grad_div_p_x, grad_div_p_y = gradient(div_p - y / weight)
        p_x_new = p_x + tau * grad_div_p_x
        p_y_new = p_y + tau * grad_div_p_y
        norm_p = np.sqrt(p_x_new**2 + p_y_new**2)
        norm_p[norm_p < 1.0] = 1.0
        p_x = p_x_new / norm_p
        p_y = p_y_new / norm_p
    return y - weight * divergence(p_x, p_y)

# ==========================
# Методы оптимизации для TV
# ==========================

def denoise_smoothed_gd_tv(y, original_img, lambda_val, num_iter, step_size, epsilon=1e-8):
    x = np.copy(y)
    obj_history = [objective_function_tv(x, y, lambda_val)]
    psnr_history = [psnr(original_img, np.clip(x, 0, 1))]
    for i in range(num_iter):
        grad_x, grad_y = gradient(x)
        norm_grad = np.sqrt(grad_x**2 + grad_y**2 + epsilon)
        grad_reg = -divergence(grad_x / norm_grad, grad_y / norm_grad)
        grad_f = (x - y) + lambda_val * grad_reg
        x = x - step_size * grad_f
        obj_history.append(objective_function_tv(x, y, lambda_val))
        psnr_history.append(psnr(original_img, np.clip(x, 0, 1)))
    return x, obj_history, psnr_history

def denoise_ista_tv(y, original_img, lambda_val, num_iter, step_size):
    x = np.copy(y)
    obj_history = [objective_function_tv(x, y, lambda_val)]
    psnr_history = [psnr(original_img, np.clip(x, 0, 1))]
    for i in range(num_iter):
        grad_f = x - y
        z = x - step_size * grad_f
        x = prox_tv(z, weight=step_size * lambda_val)
        obj_history.append(objective_function_tv(x, y, lambda_val))
        psnr_history.append(psnr(original_img, np.clip(x, 0, 1)))
    return x, obj_history, psnr_history

def denoise_fista_tv(y, original_img, lambda_val, num_iter, step_size):
    x = np.copy(y)
    t = 1.0
    x_k = np.copy(x)
    obj_history = [objective_function_tv(x, y, lambda_val)]
    psnr_history = [psnr(original_img, np.clip(x, 0, 1))]
    for i in range(num_iter):
        x_prev = np.copy(x)
        grad_f = x_k - y
        z = x_k - step_size * grad_f
        x = prox_tv(z, weight=step_size * lambda_val)
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        x_k = x + ((t_prev - 1) / t) * (x - x_prev)
        obj_history.append(objective_function_tv(x, y, lambda_val))
        psnr_history.append(psnr(original_img, np.clip(x, 0, 1)))
    return x, obj_history, psnr_history

# =======================================
# Основной блок выполнения и визуализации
# =======================================

if __name__ == "__main__":
    # --- Подготовка данных ---
    original_img = data.camera().astype(np.float64) / 255.0
    original_img = resize(original_img, (128, 128), anti_aliasing=True)

    noise_level = 0.1
    np.random.seed(42) # для воспроизводимости результатов
    noisy_img = random_noise(original_img, mode='gaussian', var=noise_level**2)

    # === ШАГ 1: Поиск оптимального параметра lambda ===
    print("--- Этап 1: Поиск оптимального lambda ---")
    lambdas = np.logspace(-2.5, -1, 20)
    psnr_values = []
    
    # Используем FISTA как самый быстрый и надежный метод для поиска
    for lmbd in lambdas:
        img_fista_lambda, _, _ = denoise_fista_tv(noisy_img, original_img, lmbd, 50, 1.0)
        current_psnr = psnr(original_img, np.clip(img_fista_lambda, 0, 1))
        psnr_values.append(current_psnr)
        print(f"Lambda: {lmbd:.4f}, PSNR: {current_psnr:.2f} dB")

    optimal_lambda_index = np.argmax(psnr_values)
    LAMBDA = lambdas[optimal_lambda_index]
    optimal_psnr = psnr_values[optimal_lambda_index]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, psnr_values, 'o-')
    plt.axvline(LAMBDA, color='r', linestyle='--', label=f'Оптимальная λ ≈ {LAMBDA:.4f}\n(Max PSNR={optimal_psnr:.2f} dB)')
    plt.xscale('log'); plt.xlabel('Коэффициент регуляризации λ (log scale)'); plt.ylabel('PSNR (dB)')
    plt.title('Поиск оптимального параметра регуляризации'); plt.legend(); plt.grid(True, which="both")
    plt.savefig('lambda_search.png'); plt.show()

    print(f"\nОптимальное значение lambda найдено: {LAMBDA:.4f}. Используем его для финального сравнения\n")

    # === ШАГ 2: Финальное сравнение методов с оптимальным lambda ===
    print("--- Этап 2: Финальное сравнение методов ---")
    ITERATIONS = 80
    STEP_SIZE_GD = 0.1
    STEP_SIZE_PROX = 1.0 # Теоретически оптимальный шаг для ISTA/FISTA

    results = {}
    
    # 1. Smoothed GD for TV
    start_time = time.time()
    img_gd, obj_hist_gd, psnr_hist_gd = denoise_smoothed_gd_tv(noisy_img, original_img, LAMBDA, ITERATIONS * 2, STEP_SIZE_GD)
    results['Smoothed GD (TV)'] = {'img': img_gd, 'time': time.time() - start_time, 'psnr': psnr(original_img, np.clip(img_gd, 0, 1)), 'obj_history': obj_hist_gd, 'psnr_history': psnr_hist_gd}

    # 2. ISTA for TV
    start_time = time.time()
    img_ista, obj_hist_ista, psnr_hist_ista = denoise_ista_tv(noisy_img, original_img, LAMBDA, ITERATIONS, STEP_SIZE_PROX)
    results['ISTA (TV)'] = {'img': img_ista, 'time': time.time() - start_time, 'psnr': psnr(original_img, np.clip(img_ista, 0, 1)), 'obj_history': obj_hist_ista, 'psnr_history': psnr_hist_ista}

    # 3. FISTA for TV
    start_time = time.time()
    img_fista, obj_hist_fista, psnr_hist_fista = denoise_fista_tv(noisy_img, original_img, LAMBDA, ITERATIONS, STEP_SIZE_PROX)
    results['FISTA (TV)'] = {'img': img_fista, 'time': time.time() - start_time, 'psnr': psnr(original_img, np.clip(img_fista, 0, 1)), 'obj_history': obj_hist_fista, 'psnr_history': psnr_hist_fista}



    # Визуализация результатов

    # Изображения
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    axes[0].imshow(original_img, cmap='gray'); axes[0].set_title('Оригинал')
    axes[1].imshow(noisy_img, cmap='gray'); axes[1].set_title(f'Шумное фото\nPSNR: {psnr(original_img, noisy_img):.2f} dB')
    method_names = ['Smoothed GD (TV)', 'ISTA (TV)', 'FISTA (TV)']
    for i, name in enumerate(method_names):
        res = results[name]
        axes[i+2].imshow(res['img'], cmap='gray')
        axes[i+2].set_title(f"{name}\nPSNR: {res['psnr']:.2f} dB")
    axes[5].axis('off')
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.savefig('TV_denoising_images_final.png'); plt.show()

    # График сходимости целевой функции
    plt.figure(figsize=(12, 8)); plt.yscale('log')
    for name, res in results.items(): plt.plot(res['obj_history'], label=f"{name} (PSNR: {res['psnr']:.2f})", lw=2)
    plt.xlabel('Итерация'); plt.ylabel('Значение целевой функции (log scale)')
    plt.title('Сравнение скорости сходимости методов (Целевая функция)'); plt.legend(); plt.grid(True)
    plt.savefig('convergence_objective_final.png'); plt.show()
    
    # 3. График роста PSNR
    plt.figure(figsize=(12, 8))
    for name, res in results.items(): plt.plot(res['psnr_history'], label=f"{name} (финальный PSNR: {res['psnr']:.2f})", lw=2)
    plt.xlabel('Итерация'); plt.ylabel('PSNR (dB)')
    plt.title('Сравнение скорости роста качества изображения (PSNR)'); plt.legend(); plt.grid(True)
    plt.savefig('convergence_psnr_final.png'); plt.show()



    # ========================================
    # СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ В ОТДЕЛЬНЫЕ ФАЙЛЫ
    # ========================================

    file_names = {
        'original': '01_original.png',
        'noisy': '02_noisy.png',
        'Smoothed GD (TV)': '03_result_gd.png',
        'ISTA (TV)': '04_result_ista.png',
        'FISTA (TV)': '05_result_fista.png'
    }


    # оригинал
    plt.imsave(file_names['original'], original_img, cmap='gray')
    print(f"Сохранено: {file_names['original']}")

    # зашумленное изображение
    noisy_clipped = np.clip(noisy_img, 0, 1)
    plt.imsave(file_names['noisy'], noisy_clipped, cmap='gray')
    print(f"Сохранено: {file_names['noisy']}")

    # результаты работы алгоритмов
    for name, res in results.items():
        if name in file_names:
            img_clipped = np.clip(res['img'], 0, 1)
            plt.imsave(file_names[name], img_clipped, cmap='gray')
            print(f"Сохранено: {file_names[name]}")

    # промежуточные результаты
    ITERS_TO_COMPARE = 10
    iters_gd_early = ITERS_TO_COMPARE * 2

    img_gd_early, _, _ = denoise_smoothed_gd_tv(noisy_img, original_img, LAMBDA, iters_gd_early, STEP_SIZE_GD)
    img_ista_early, _, _ = denoise_ista_tv(noisy_img, original_img, LAMBDA, ITERS_TO_COMPARE, STEP_SIZE_PROX)
    img_fista_early, _, _ = denoise_fista_tv(noisy_img, original_img, LAMBDA, ITERS_TO_COMPARE, STEP_SIZE_PROX)

    plt.imsave('06_early_gd.png', np.clip(img_gd_early, 0, 1), cmap='gray')
    print("Сохранено: 06_early_gd.png")
    plt.imsave('07_early_ista.png', np.clip(img_ista_early, 0, 1), cmap='gray')
    print("Сохранено: 07_early_ista.png")
    plt.imsave('08_early_fista.png', np.clip(img_fista_early, 0, 1), cmap='gray')
    print("Сохранено: 08_early_fista.png")


    # Итоговая таблица
    print("\n--- Итоговая таблица (с оптимальным lambda) ---")
    print(f"| {'Метод':<20} | {'PSNR (dB)':<10} | {'Время (с)':<12} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*14}|")
    for name, res in results.items():
        print(f"| {name:<20} | {res['psnr']:<10.2f} | {res['time']:<12.4f} |")





    # ====================================
    # Сравнение результатов за 10 итераций
    # ====================================
    ITERS_TO_COMPARE = 5

    fig_early, axes_early = plt.subplots(1, 4, figsize=(20, 5))
    fig_early.suptitle(f'Сравнение результатов после {ITERS_TO_COMPARE} итераций', fontsize=16)

    # Зашумленное изображение для сравнения
    axes_early[0].imshow(noisy_img, cmap='gray')
    axes_early[0].set_title(f'Шумное фото\nPSNR: {psnr(original_img, noisy_img):.2f} dB')

    # Smoothed GD
    iters_gd_early = ITERS_TO_COMPARE * 2
    img_gd_early, _, _ = denoise_smoothed_gd_tv(noisy_img, original_img, LAMBDA, iters_gd_early, STEP_SIZE_GD)
    psnr_gd_early = psnr(original_img, np.clip(img_gd_early, 0, 1))
    axes_early[1].imshow(img_gd_early, cmap='gray')
    axes_early[1].set_title(f'Smoothed GD ({iters_gd_early} итер.)\nPSNR: {psnr_gd_early:.2f} dB')

    # ISTA
    img_ista_early, _, _ = denoise_ista_tv(noisy_img, original_img, LAMBDA, ITERS_TO_COMPARE, STEP_SIZE_PROX)
    psnr_ista_early = psnr(original_img, np.clip(img_ista_early, 0, 1))
    axes_early[2].imshow(img_ista_early, cmap='gray')
    axes_early[2].set_title(f'ISTA ({ITERS_TO_COMPARE} итер.)\nPSNR: {psnr_ista_early:.2f} dB')

    # FISTA
    img_fista_early, _, _ = denoise_fista_tv(noisy_img, original_img, LAMBDA, ITERS_TO_COMPARE, STEP_SIZE_PROX)
    psnr_fista_early = psnr(original_img, np.clip(img_fista_early, 0, 1))
    axes_early[3].imshow(img_fista_early, cmap='gray')
    axes_early[3].set_title(f'FISTA ({ITERS_TO_COMPARE} итер.)\nPSNR: {psnr_fista_early:.2f} dB')

    for ax in axes_early:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('early_comparison.png')
    plt.show()

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Фіксація генератора псевдовипадкових чисел згідно з методичкою
random.seed(1800)
np.random.seed(1800)

# Створення папки для графіків
os.makedirs("graphics", exist_ok=True)


# =============================================================================
# БЛОК 1: Обробка незгрупованої вибірки та побудова ECDF
# =============================================================================
def run_block_1():
    print("=== БЛОК 1: Обробка незгрупованої вибірки (n=20) та ECDF ===")
    X_raw = [
        23.4, 18.2, 25.1, 21.0, 18.2, 29.5, 23.4, 21.0, 27.8, 18.2,
        23.4, 25.1, 31.0, 21.0, 23.4, 19.8, 27.8, 23.4, 21.0, 25.1
    ]
    n = len(X_raw)
    
    # 1. Ранжований варіаційний ряд
    X_sorted = sorted(X_raw)
    print("Ранжований ряд:", X_sorted)
    
    # 2. Статистичний розподіл (унікальні значення, частоти n_i, відносні w_i)
    unique_vals = sorted(list(set(X_sorted)))
    print("\nСтатистичний розподіл:")
    print(f"{'x_i':<8} | {'n_i':<5} | {'w_i':<8}")
    for val in unique_vals:
        ni = X_sorted.count(val)
        wi = ni / n
        print(f"{val:<8.1f} | {ni:<5} | {wi:<8.3f}")
        
    # 3. Емпіричні числові характеристики
    arr = np.array(X_sorted)
    x_mean = np.mean(arr)
    s2_biased = np.var(arr)                  # Вибіркова дисперсія s^2
    S2_unbiased = np.var(arr, ddof=1)        # Виправлена дисперсія S^2
    S_std = np.std(arr, ddof=1)              # Виправлене стандартне відхилення S
    x_median = np.median(arr)                # Медіана
    
    # Мода (елемент з найбільшою частотою)
    x_mode = max(unique_vals, key=lambda x: X_sorted.count(x))
    R_range = X_sorted[-1] - X_sorted[0]     # Розмах R
    V_coef = (S_std / x_mean) * 100          # Коефіцієнт варіації V (%)
    
    print(f"\nСереднє (x_bar):             {x_mean:.4f}")
    print(f"Вибіркова дисперсія (s^2):   {s2_biased:.4f}")
    print(f"Виправлена дисперсія (S^2):  {S2_unbiased:.4f}")
    print(f"Стандартне відхилення (S):   {S_std:.4f}")
    print(f"Медіана:                     {x_median:.4f}")
    print(f"Мода (Mo):                   {x_mode:.1f}")
    print(f"Розмах (R):                  {R_range:.1f}")
    print(f"Коефіцієнт варіації (V):     {V_coef:.2f}%")
    
    # 4. Ступінчастий графік ECDF F*(x)
    y_ecdf = np.arange(1, n + 1) / n
    plt.figure(figsize=(8, 5))
    plt.step(X_sorted, y_ecdf, where='post', color='blue', linewidth=2, label='ECDF F*(x)')
    plt.scatter(X_sorted, y_ecdf, color='red', zorder=3)
    plt.title("Блок 1: Емпірична функція розподілу F*(x)")
    plt.xlabel("Затримка виконання функції, мс")
    plt.ylabel("F*(x)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("graphics/block1_ecdf.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block1_ecdf.png\n")

if __name__ == "__main__":
    run_block_1()
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

# =============================================================================
# БЛОК 2: Інтервальне групування за варіантом (Варіант 2: RAM сервера)
# =============================================================================
def run_block_2():
    print("=== БЛОК 2: Варіант 2 (Використання RAM сервером, n=200) ===")
    n = 200
    mu, sigma = 1024.0, 128.0
    
    # 1. Генерація вибірки N(1024, 128^2)
    sample = np.random.normal(mu, sigma, n)
    
    # 2. Число інтервалів за Стерджесом (k = 8) та крок h
    k = 1 + int(round(3.322 * math.log10(n)))
    x_min, x_max = np.min(sample), np.max(sample)
    h = (x_max - x_min) / k
    
    print(f"Кількість інтервалів k = {k}, Крок h = {h:.2f} МБ")
    
    # 3. Побудова інтервального ряду
    counts, bin_edges = np.histogram(sample, bins=k)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    rel_freqs = counts / n
    densities = rel_freqs / h
    
    print("\nІнтервальний варіаційний ряд:")
    print(f"{'Інтервал [a_i, a_i+1)':<24} | {'x_i':<8} | {'n_i':<5} | {'w_i':<7} | {'p_i (щільність)':<15}")
    for i in range(k):
        interval_str = f"[{bin_edges[i]:.1f}; {bin_edges[i+1]:.1f})"
        print(f"{interval_str:<24} | {midpoints[i]:<8.1f} | {counts[i]:<5} | {rel_freqs[i]:<7.3f} | {densities[i]:<15.6f}")
        
    # 4. Згруповані оцінки vs Незгруповані
    x_raw_mean = np.mean(sample)
    S2_raw = np.var(sample, ddof=1)
    
    x_grp_mean = np.sum(counts * midpoints) / n
    S2_grp = np.sum(counts * ((midpoints - x_grp_mean) ** 2)) / (n - 1)
    
    print(f"\nНезгруповане середнє: {x_raw_mean:.2f} МБ | Згруповане середнє: {x_grp_mean:.2f} МБ (Похибка: {abs(x_raw_mean - x_grp_mean):.2f})")
    print(f"Незгрупована S^2:     {S2_raw:.2f}    | Згрупована S^2:     {S2_grp:.2f}    (Похибка: {abs(S2_raw - S2_grp):.2f})")
    
    # 5. Гістограма щільності та полігон частот на одному рисунку
    plt.figure(figsize=(9, 5))
    plt.hist(sample, bins=bin_edges, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Гістограма щільності')
    plt.plot(midpoints, densities, marker='o', color='red', linewidth=2, label='Полігон щільності')
    plt.title("Блок 2: Гістограма щільності та полігон частот (RAM сервера)")
    plt.xlabel("Використання RAM, МБ")
    plt.ylabel("Щільність p_i = w_i / h")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("graphics/block2_hist_polygon.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block2_hist_polygon.png\n")
    
    return sample

# =============================================================================
# БЛОК 3: Детекція та фільтрація аномальних вимірів (Outliers)
# =============================================================================
def run_block_3(base_sample):
    print("=== БЛОК 3: Детекція та фільтрація аномалій (Outliers) ===")
    x_base_mean = np.mean(base_sample)
    S_base = np.std(base_sample, ddof=1)
    
    # 1. Додаємо 10 аномальних значень, що перевищують x_bar + 5S
    threshold_outlier = x_base_mean + 5 * S_base
    outliers = np.random.uniform(threshold_outlier + 50, threshold_outlier + 400, 10)
    dirty_sample = np.concatenate([base_sample, outliers])
    
    # 2. Характеристики спотвореної вибірки (dirty)
    x_dirty = np.mean(dirty_sample)
    S_dirty = np.std(dirty_sample, ddof=1)
    med_dirty = np.median(dirty_sample)
    Q1_d, Q3_d = np.percentile(dirty_sample, [25, 75])
    IQR_dirty = Q3_d - Q1_d
    
    print(f"Спотворена вибірка (n={len(dirty_sample)}):")
    print(f"  x_dirty = {x_dirty:.2f} | S_dirty = {S_dirty:.2f} | Медіана = {med_dirty:.2f} | IQR = {IQR_dirty:.2f}")
    
    # 3. Межі Boxplot та фільтрація викидів
    lower_bound = Q1_d - 1.5 * IQR_dirty
    upper_bound = Q3_d + 1.5 * IQR_dirty
    print(f"Межі нормальних значень за IQR: [{lower_bound:.2f}; {upper_bound:.2f}]")
    
    clean_sample = dirty_sample[(dirty_sample >= lower_bound) & (dirty_sample <= upper_bound)]
    removed_count = len(dirty_sample) - len(clean_sample)
    print(f"Виявлено та відфільтровано викидів: {removed_count}")
    
    # 4. Характеристики очищеної вибірки (clean)
    x_clean = np.mean(clean_sample)
    S_clean = np.std(clean_sample, ddof=1)
    med_clean = np.median(clean_sample)
    
    print(f"Очищена вибірка (n={len(clean_sample)}):")
    print(f"  x_clean = {x_clean:.2f} | S_clean = {S_clean:.2f} | Медіана = {med_clean:.2f}")
    print(f"Зміна середнього: {abs(x_dirty - x_clean):.2f} МБ | Зміна медіани: {abs(med_dirty - med_clean):.2f} МБ")
    
    # Побудова Boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot([dirty_sample, clean_sample], tick_labels=['З викидами (dirty)', 'Очищена (clean)'], patch_artist=True)
    plt.title("Блок 3: Порівняння вибірок до та після фільтрації викидів (Boxplot)")
    plt.ylabel("Використання RAM, МБ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("graphics/block3_boxplot.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block3_boxplot.png\n")

# =============================================================================
# БЛОК 4: Професійна задача (Варіант 8: Лічильники контейнерів Docker)
# =============================================================================
def run_block_4():
    print("=== БЛОК 4: Варіант 8 (Лічильники контейнерів Docker, n=300) ===")
    n = 300
    
    # Моделюємо навантаження контейнерів Docker (наприклад, навантаження CPU у %: штатне ~45% + піки)
    normal_load = np.random.normal(45, 12, int(n * 0.85))
    peak_load = np.random.uniform(75, 98, n - len(normal_load))
    docker_metrics = np.clip(np.concatenate([normal_load, peak_load]), 5, 100)
    
    # Інтервальний розподіл за Стерджесом: k = 1 + [3.322 * log10(300)] = 9
    k = 1 + int(round(3.322 * math.log10(n)))
    counts, bin_edges = np.histogram(docker_metrics, bins=k)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    rel_freqs = counts / n
    
    print(f"Кількість інтервалів за Стерджесом: k = {k}")
    print(f"{'Інтервал навантаження (%)':<26} | {'Середина':<10} | {'Частота n_i':<12} | {'Частка w_i':<10}")
    
    # Рахуємо частку інтервалів із піковим навантаженням (де середина інтервалу > 75%)
    peak_intervals = 0
    for i in range(k):
        interval_str = f"[{bin_edges[i]:.1f}; {bin_edges[i+1]:.1f})"
        print(f"{interval_str:<26} | {midpoints[i]:<10.1f} | {counts[i]:<12} | {rel_freqs[i]:<10.3f}")
        if midpoints[i] > 75.0:
            peak_intervals += 1
            
    peak_ratio = (peak_intervals / k) * 100
    print(f"\nКількість інтервалів із піковим навантаженням (>75%): {peak_intervals} з {k} ({peak_ratio:.1f}%)")
    
    # Полігон частот
    plt.figure(figsize=(9, 5))
    plt.hist(docker_metrics, bins=bin_edges, alpha=0.4, color='teal', edgecolor='black', label='Гістограма частот')
    plt.plot(midpoints, counts, marker='s', color='darkorange', linewidth=2, label='Полігон частот')
    plt.axvline(75.0, color='red', linestyle='--', label='Поріг пікового навантаження (75%)')
    plt.title("Блок 4: Розподіл навантаження контейнерів Docker (n=300)")
    plt.xlabel("Навантаження контейнера, %")
    plt.ylabel("Абсолютна частота n_i")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("graphics/block4_docker_polygon.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block4_docker_polygon.png\n")

if __name__ == "__main__":
    run_block_1()
    run_block_2()
    ram_sample = run_block_2()
    run_block_3(ram_sample)
    run_block_4()
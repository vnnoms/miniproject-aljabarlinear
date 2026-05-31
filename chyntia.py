from callista import *
# ==========================================
# EVALUASI MODEL
# ==========================================
# Variabel 'residuals' dan 'y' sudah dideklarasikan di kode sebelumnya

# 1. Menghitung Sum of Squared Errors (SSE)
SSE = np.sum(residuals**2)

# 2. Menghitung Total Sum of Squares (TSS)
y_bar = np.mean(y)
TSS = np.sum((y - y_bar)**2)

# 3. Menghitung Koefisien Determinasi (R^2)
R_squared = 1 - (SSE / TSS)

print("\n" + "=" * 50)
print("HASIL EVALUASI MODEL (UNTUK LAPORAN 2.3)")
print("=" * 50)
print(f"Sum of Squared Errors (SSE) = {SSE:.4f}")
print(f"Total Sum of Squares (TSS)  = {TSS:.4f}")
print(f"Koefisien Determinasi (R^2) = {R_squared:.4f}")
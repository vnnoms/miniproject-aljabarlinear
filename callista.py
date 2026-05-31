import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 10, 20)
epsilon = np.random.normal(loc=0, scale=0.5, size=20)
y = 2 + 3 * x + epsilon

print("=" * 50)
print("DATA YANG DIBANGKITKAN")
print("=" * 50)
print(f"{'i':>3} | {'x_i':>8} | {'y_i':>10}")
print("-" * 28)

for i in range(20):
    print(f"{i+1:>3} | {x[i]:>8.4f} | {y[i]:>10.4f}")
A = np.column_stack([np.ones(20), x])
b = y

print("\n" + "=" * 50)
print("MATRIKS A (5 baris pertama dari 20)")
print("=" * 50)
print(A[:5])

print("\nVEKTOR b (5 elemen pertama dari 20):")
print(b[:5])

AtA = A.T @ A         
Atb = A.T @ b          

print("\n" + "=" * 50)
print("PERHITUNGAN PERSAMAAN NORMAL: A^T * A * x_hat = A^T * b")
print("=" * 50)

print("\nMatriks A^T * A:")
print(AtA)

print("\nVektor A^T * b:")
print(Atb)

AtA_inv = np.linalg.inv(AtA)
x_hat   = AtA_inv @ Atb

beta0_hat = x_hat[0]
beta1_hat = x_hat[1]

print("\n" + "=" * 50)
print("SOLUSI LEAST SQUARES")
print("=" * 50)
print(f"β̂₀ (intersep) = {beta0_hat:.4f}  (nilai asli: 2)")
print(f"β̂₁ (slope)    = {beta1_hat:.4f}  (nilai asli: 3)")
print(f"\nPersamaan garis regresi: ŷ = {beta0_hat:.4f} + {beta1_hat:.4f}x")

x_hat_verify, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(f"\n[Verifikasi numpy.linalg.lstsq]")
print(f"β̂₀ = {x_hat_verify[0]:.4f}, β̂₁ = {x_hat_verify[1]:.4f}")

# Visualisasi data dan garis regresi
y_hat = A @ x_hat          
residuals = y - y_hat      

x_line = np.linspace(0, 10, 200)
y_line = beta0_hat + beta1_hat * x_line

y_true_line = 2 + 3 * x_line

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.scatter(x, y, color='steelblue', s=60, zorder=5, label='Data (xᵢ, yᵢ)')

ax1.plot(x_line, y_line, color='crimson', linewidth=2,
         label=f'Regresi: ŷ = {beta0_hat:.4f} + {beta1_hat:.4f}x')

ax1.plot(x_line, y_true_line, color='green', linewidth=1.5,
         linestyle='--', label='True: y = 2 + 3x')

for i in range(len(x)):
    ax1.plot([x[i], x[i]], [y[i], y_hat[i]],
             color='gray', linewidth=0.8, linestyle='-', alpha=0.6)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Scatter Plot & Garis Regresi (Least Squares)', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]

ax2.scatter(x, residuals, color='darkorange', s=60, zorder=5)
ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='--')

for i in range(len(x)):
    ax2.plot([x[i], x[i]], [0, residuals[i]],
             color='darkorange', linewidth=0.8, alpha=0.6)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Residu (rᵢ = yᵢ − ŷᵢ)', fontsize=12)
ax2.set_title('Plot Residu', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisasi_regresi.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot berhasil disimpan sebagai 'visualisasi_regresi.png'")

# Print output 
print("\n" + "=" * 55)
print("MATRIKS A^T * A - ukuran 2x2")
print("=" * 55)
print(f"  [ {AtA[0,0]:>10.4f}   {AtA[0,1]:>10.4f} ]")
print(f"  [ {AtA[1,0]:>10.4f}   {AtA[1,1]:>10.4f} ]")

print("\n" + "=" * 55)
print("VEKTOR A^T * b - ukuran 2x1")
print("=" * 55)
print(f"  [ {Atb[0]:>12.4f} ]")
print(f"  [ {Atb[1]:>12.4f} ]")

print("\n" + "=" * 55)
print("SOLUSI x_hat = (A^T A)^-1 * A^T b")
print("=" * 55)
print(f"  x_hat = [ β̂₀ ]  =  [ {beta0_hat:.4f} ]")
print(f"          [ β̂₁ ]     [ {beta1_hat:.4f} ]")
print(f"\n  Persamaan regresi: ŷ = {beta0_hat:.4f} + {beta1_hat:.4f}x")
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros Iniciais

L = 250.0          # Comprimento (m)
B = 40.0           # Boca (m)
T = 12.0           # Calado (m)
Delta = 123000.0   # Deslocamento (ton)
rho = 1.025        # Massa específica da água (ton/m³)
g = 9.81           # Aceleração da gravidade (m/s²)
Ca = 1.3           # Coef. de massa adicional para seção retangular
A = 1.0            # Amplitude da onda (m)
phi = 0.05         # Fator de amortecimento (5%)

def calc_frequencia_periodo_natural():
    # Massa adicional (ton/m)
    Ma_2D = Ca * (rho * np.pi / 2) * (B / 2)**2

    # Massa adicional total (ton)
    Ma_total = Ma_2D * L

    # Rigidez hidrostática (kN/m)
    c = rho * g * L * B # (Awl = L*B)

    # Frequência natural (rad/s)
    wn = np.sqrt(c / (Delta + Ma_total))

    # Período natural (s)
    Tn = 2 * np.pi / wn

    return wn, Tn, Ma_2D

def calc_forca_froude_krylov(w):
    # Impede divisão por zero ao calcular o limite w -> 0
    w_safe = np.where(w == 0, 1e-10, w)

    F_FK = 2 * rho * g**2 * B * (A / w_safe**2) * np.exp(-w_safe**2 * T / g) * np.sin(w_safe**2 * L / (2 * g))

    # Limite matemático quando w = 0 (Força = rho * g * Volume deslocado da onda)
    F_FK = np.where(w == 0, rho * g * B * L * A, F_FK)

    return np.abs(F_FK)

def calc_forca_massa_adicional(w, Ma_2D):
    F_MA = 2 * Ma_2D * g * A * np.exp(-w**2 * T / g) * np.sin(w**2 * L / (2 * g))
    return np.abs(F_MA)

def calc_forca_total(w, Ma_2D):
    w_safe = np.where(w == 0, 1e-10, w)

    F_total = 2 * g * A * np.exp(-w_safe**2 * T / g) * np.sin(w_safe**2 * L / (2 * g)) * ((rho * g * B / w_safe**2) - Ma_2D)

    F_total = np.where(w == 0, rho * g * B * L * A, F_total)

    return np.abs(F_total)

def calc_forca_froude_krylov(w):
    # Impede divisao por zero ao calcular o limite w -> 0
    w_safe = np.where(w == 0, 1e-10, w)
    F_FK = 2 * rho * g**2 * B * (A / w_safe**2) * np.exp(-w_safe**2 * T / g) * np.sin(w_safe**2 * L / (2 * g))
    # Limite matematico quando w = 0 (Forca = rho * g * Volume deslocado da onda)
    F_FK = np.where(w == 0, rho * g * B * L * A, F_FK)
    return np.abs(F_FK)

def calc_forca_massa_adicional(w, Ma_2D):
    F_MA = 2 * Ma_2D * g * A * np.exp(-w**2 * T / g) * np.sin(w**2 * L / (2 * g))
    return np.abs(F_MA)

def calc_forca_total(w, Ma_2D):
    w_safe = np.where(w == 0, 1e-10, w)
    F_total = 2 * g * A * np.exp(-w_safe**2 * T / g) * np.sin(w_safe**2 * L / (2 * g)) * ((rho * g * B / w_safe**2) - Ma_2D)
    F_total = np.where(w == 0, rho * g * B * L * A, F_total)
    return np.abs(F_total)

def calc_rao_heave(w, wn, F_total):
    # Rigidez hidrostática (kN/m)
    c = rho * g * L * B

    # Fator de amplificação dinâmica
    Amp_din = np.sqrt((1 - (w / wn)**2)**2 + (2 * phi * w / wn)**2)

    # RAO = (|F_total| / c) / Fator

    # Obs: Precisa ser dividido por A para ficar adimensional (m/m)
    RAO = (F_total / (c * A)) / Amp_din

    return RAO

def plotar_graficos():
    # Calculando parâmetros naturais primeiro
    wn, Tn, Ma_2D = calc_frequencia_periodo_natural()
    print(f"--- Parâmetros Naturais ---")
    print(f"Frequência Natural (wn): {wn:.3f} rad/s")
    print(f"Período Natural (Tn):    {Tn:.2f} s")

    # Criação do vetor de frequências
    w = np.linspace(0.0, 1.5, 500) # 0 a 1.5 rad/s abrange períodos de ∞ até ~4s

    # Obtendo vetores de força (colocando em MN para melhor visualização)
    F_FK = calc_forca_froude_krylov(w) / 1000
    F_MA = calc_forca_massa_adicional(w, Ma_2D) / 1000
    F_Tot = calc_forca_total(w, Ma_2D) / 1000

    # Obtendo vetor do RAO
    # Obs: Passa o F_total original em kN para o RAO manter proporções corretas
    RAO = calc_rao_heave(w, wn, F_Tot * 1000)

    # Configuração
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Forças
    ax1.plot(w, F_FK, label='| F_FK | (Froude-Krylov)', color='blue')
    ax1.plot(w, F_MA, label='| F_MA | (Massa Adicional)', color='green')
    ax1.plot(w, F_Tot, label='| F_Total |', color='red', linewidth=2)
    ax1.set_title('Forças de Excitação de Heave (Amplitude = 1m)')
    ax1.set_xlabel('Frequência da onda $\omega$ [rad/s]')
    ax1.set_ylabel('Força [MN / m]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Linha vertical indicando período de 35s (w ~ 0.18 rad/s)
    w_35s = 2 * np.pi / 35
    ax1.axvline(x=w_35s, color='gray', linestyle=':', label='T = 35s')

    # Gráfico 2: RAO
    ax2.plot(w, RAO, color='purple', linewidth=2)
    ax2.set_title('RAO de Heave')
    ax2.set_xlabel('Frequência da onda $\omega$ [rad/s]')
    ax2.set_ylabel('RAO |Z|/A [m/m]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axvline(x=wn, color='orange', linestyle='--', label=f'Ressonância ($\omega_n$ = {wn:.2f})')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plotar_graficos()
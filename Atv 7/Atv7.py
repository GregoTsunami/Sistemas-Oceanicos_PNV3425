# Dados de entrada fornecidos
Lp = 100
bp = 15
hp = 15
bc = 15
hc = 45
T = 30
rho = 1.025

def calcular_hidrostatica(Lp, bp, hp, bc, hc, T, rho):
    """
    Calcula as propriedades hidrostáticas de uma plataforma semissubmersível.
    Dimensões em metros (m) e densidade em toneladas por metro cúbico (t/m³).
    """
    # 1. Volume Deslocado e Deslocamento
    vol_pont = 2 * (Lp * bp * hp)
    calado_colunas = T - hp
    vol_col = 4 * (bc**2 * calado_colunas)

    vol_desloc = vol_pont + vol_col
    deslocamento = vol_desloc * rho

    # 2. Posição Vertical do Centro de Carena (KB)
    kb_pont = hp / 2.0
    kb_col = hp + (calado_colunas / 2.0)

    kb = ((vol_pont * kb_pont) + (vol_col * kb_col)) / vol_desloc

    # 3. Raio Metacêntrico (BM)
    # Distância do centro geométrico da plataforma ao centro de cada coluna
    dist_centro_coluna = (Lp - bc) / 2.0

    # Teorema de Steiner para o momento de inércia do plano de flutuação
    inercia_propria_col = (bc**4) / 12.0
    area_col = bc**2

    inercia_total = 4 * (inercia_propria_col + area_col * (dist_centro_coluna**2))
    bm = inercia_total / vol_desloc

    # 4. Altura Metacêntrica (KM)
    km = kb + bm

    return {
        "Volume_Deslocado": vol_desloc,
        "Deslocamento": deslocamento,
        "KB": kb,
        "BM": bm,
        "KM": km
    }

if __name__ == "__main__":    
    resultados = calcular_hidrostatica(Lp, bp, hp, bc, hc, T, rho)

    print("--- Propriedades Hidrostáticas ---")
    for propriedade, valor in resultados.items():
        print(f"{propriedade}: {valor:.2f}")
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import dual_annealing
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================================
# 1. DADOS FIXOS DO CAMPO BM-S-52 E NORMAS
# =========================================================================
# Dados do Reservatório BM-S-52 (Pico de produção no Ano 6)
Q_prod = 230.0        # mil BOE/dia - Pico de produção no Ano 6 (230.000 BPD)
Hs_mar = 7.3          # m - Altura significativa do mar centenário (100 anos)
Tp_mar = 14.6         # s - Período de pico do mar centenário (100 anos)
gamma_jonswap = 3.3   # Fator de pico do espectro de JONSWAP

# Constantes Físicas e Critérios Normativos (DNV-OS-C301 / MODU 2009)
gamma_e  = 0.270      # t/m³ - Peso específico estrutural equivalente
rho_agua = 1.025      # t/m³ - Massa específica da água do mar
rho_ar   = 1.222e-3   # t/m³ - Massa específica do ar
g        = 9.81       # m/s² - Aceleração da gravidade
xi_heave = 0.05       # Adimensional - Fator de amortecimento em heave

V_vento_temp = 51.5   # m/s - Velocidade do vento de tempestade severa (100 nós)
V_vento_oper = 36.0   # m/s - Velocidade do vento de operação normal (70 nós)

# Dimensões do Topside (Estimadas por unidades semelhantes: P-51/P-52)
hd = 9.0              # m - Altura do deck box
hm = 16.0             # m - Altura média dos módulos de topside

def criar_entradas(Lp, bp, hp, bc, hc, T, Q=Q_prod, Hs=Hs_mar, Tp=Tp_mar):
    """
    Organiza as 6 variáveis de projeto e os parâmetros ambientais do cenário.
    """
    return {
        # Variáveis de decisão
        "Lp": float(Lp),   # Comprimento dos pontoons e lado da planta quadrada (m)
        "bp": float(bp),   # Boca dos pontoons (m)
        "hp": float(hp),   # Altura dos pontoons (m)
        "bc": float(bc),   # Boca/lado das colunas quadradas (m)
        "lc": float(bc),   # Largura das colunas (lc = bc para seção quadrada)
        "hc": float(hc),   # Altura das colunas (m)
        "T":  float(T),    # Calado de projeto (m)
        
        # Geometria derivada
        "ld": float(Lp - 2 * bc),  # Comprimento descoberto dos pontoons (m)
        "dp": float(Lp - bp),      # Distância entre os centros dos pontoons (m)
        
        # Parâmetros do cenário
        "Q":  float(Q),
        "Hs": float(Hs),
        "Tp": float(Tp)
    }

# Dados da P-51 / P-52 (Capacidade: 180 mil BPD | Elevação do convés: 46,0 m)
# Nota: Para avaliar a P-51/P-52 em sua própria condição, usamos Q=180.0
entrada_teste = criar_entradas(
    Lp=125.0, bp=18.0, hp=11.5, bc=18.0, hc=34.5, T=26.5,
    Q=180.0, Hs=Hs_mar, Tp=Tp_mar
)

print("=== ETAPA 1: DADOS DE ENTRADA - SEMELHANTE P-51 / P-52 ===")
for chave, valor in entrada_teste.items():
    print(f"{chave:<6}: {valor:.2f}")

def plotar_semisub_teste(dados, hd_val=9.0, hm_val=16.0, titulo="Modelo 3D - Semelhante P-51 / P-52"):
    Lp, bp, hp = dados["Lp"], dados["bp"], dados["hp"]
    bc, hc, T  = dados["bc"], dados["hc"], dados["T"]
    meio_Lp    = Lp / 2.0
    z_col_topo = hp + hc
    z_dk_topo  = z_col_topo + hd_val
    z_md_topo  = z_dk_topo + hm_val

    def _caixa_solida(x0, x1, y0, y1, z0, z1, cor, nome, leg=True):
        faces = [
            ([x0, x0, x1, x1], [y0, y1, y1, y0], [z0, z0, z0, z0]),
            ([x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]),
            ([x0, x0, x0, x0], [y0, y0, y1, y1], [z0, z1, z1, z0]),
            ([x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]),
            ([x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]),
            ([x0, x0, x1, x1], [y1, y1, y1, y1], [z0, z1, z1, z0]),
        ]
        trs = []
        for idx, (fx, fy, fz) in enumerate(faces):
            trs.append(go.Mesh3d(
                x=fx, y=fy, z=fz, i=[0, 0], j=[1, 2], k=[2, 3],
                color=cor, opacity=1.0, name=nome,
                showlegend=(leg and idx == 1), legendgroup=nome,
                flatshading=True, lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15)
            ))
        bx = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
        by = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None, y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
        bz = [z0, z0, z0, z0, z0, None, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1, None, z0, z1]
        trs.append(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line=dict(color="rgba(20,20,20,0.5)", width=2),
            name=nome, showlegend=False, legendgroup=nome
        ))
        return trs

    fig = go.Figure()

    # 1. Pontoons
    for sy in [-1, 1]:
        yc = sy * (Lp - bp) / 2.0
        for tr in _caixa_solida(-meio_Lp, meio_Lp, yc - bp/2.0, yc + bp/2.0, 0.0, hp, "#8B6914", "Pontoons", leg=(sy == -1)):
            fig.add_trace(tr)

    # 2. Colunas
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            cx, cy = sx * (Lp - bc) / 2.0, sy * (Lp - bc) / 2.0
            for tr in _caixa_solida(cx - bc/2.0, cx + bc/2.0, cy - bc/2.0, cy + bc/2.0, hp, z_col_topo, "#2E6B9E", "Colunas", leg=(sx == -1 and sy == -1)):
                fig.add_trace(tr)

    # 3. Deck Box
    for tr in _caixa_solida(-meio_Lp, meio_Lp, -meio_Lp, meio_Lp, z_col_topo, z_dk_topo, "#7F8C8D", "Deck Box (hd=9m)"):
        fig.add_trace(tr)

    # 4. Módulos de Topside
    recuo = 0.08 * Lp
    for tr in _caixa_solida(-meio_Lp + recuo, meio_Lp - recuo, -meio_Lp + recuo, meio_Lp - recuo, z_dk_topo, z_md_topo, "#D35400", "Módulos Topside (hm=16m)"):
        fig.add_trace(tr)

    # 5. Plano d'água no calado T
    w = meio_Lp + 20.0
    fig.add_trace(go.Mesh3d(
        x=[-w, w, w, -w], y=[-w, -w, w, w], z=[T, T, T, T],
        i=[0, 0], j=[1, 2], k=[2, 3], color="#0077CC", opacity=0.22,
        name=f"Linha d'água (T={T}m)", showlegend=True, legendgroup="LDA",
        flatshading=True, lighting=dict(ambient=1.0)
    ))
    fig.add_trace(go.Scatter3d(
        x=[-w, w, w, -w, -w], y=[-w, -w, w, w, -w], z=[T]*5,
        mode="lines", line=dict(color="#004C99", width=3),
        name=f"Linha d'água (T={T}m)", showlegend=False, legendgroup="LDA"
    ))

    axis_cfg = dict(showbackground=True, backgroundcolor="rgba(240, 248, 255, 0.5)")
    fig.update_layout(
        title=dict(text=titulo, x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis=dict(**axis_cfg, title="X (m)"),
            yaxis=dict(**axis_cfg, title="Y (m)"),
            zaxis=dict(**axis_cfg, title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.85))
        ),
        legend=dict(x=0.02, y=0.95, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=600
    )
    fig.show()

# Gera o desenho 3D da P-51
plotar_semisub_teste(entrada_teste, hd_val=9.0, hm_val=16.0, titulo="Semelhante P-51 / P-52 (Lp=125m, bp=bc=18m, T=26,5m)")

# =========================================================================
# 2. PROPRIEDADES HIDROSTÁTICAS, PESOS E RESTRIÇÕES ESTÁTICAS
# =========================================================================

def calcular_hidrostatica(dados):
    """
    Recebe o dicionário 'dados' da Etapa 1, calcula:
      1) Volume deslocado, deslocamento, KB, BM e KM
      2) Peso estrutural do casco, topside, lastro, KG e GM
      3) Área de convés e air gap em repouso
    E verifica as restrições estáticas/geométricas do modelo    
    """
    Lp = dados["Lp"]
    bp = dados["bp"]
    hp = dados["hp"]
    bc = dados["bc"]
    hc = dados["hc"]
    T  = dados["T"]
    Q  = dados["Q"]

    # 1) Volumes, Deslocamento e Carena
    vol_pontoons = 2.0 * Lp * bp * hp
    vol_colunas  = 4.0 * (bc**2) * hc

    vol_sub_pontoon = 2.0 * Lp * bp * min(hp, T)
    vol_sub_coluna  = 4.0 * (bc**2) * max(0.0, T - hp)
    nabla = vol_sub_pontoon + vol_sub_coluna # Volume total (m^3)
    Delta = nabla * rho_agua  # Deslocamento total (t)

    KB = (vol_sub_pontoon * (hp / 2.0) + vol_sub_coluna * (hp + (T - hp) / 2.0)) / nabla
    Ix = 4.0 * ((bc**4) / 12.0 + (bc**2) * (((Lp - bc) / 2.0)**2))
    BM = Ix / nabla
    KM = KB + BM

    # 2) Pesos, Centros e GM
    P_PL      = 83.0 * Q                     # Peso seco da planta de processo (t)
    P_topside = P_PL * (1.0 + 0.35 + 0.35)   # Topside total (planta + convés + operacional) (t)
    P_C       = (vol_pontoons + vol_colunas) * gamma_e  # Peso estrutural do casco (t)

    VCG_C = (vol_pontoons * (hp / 2.0) + vol_colunas * (hp + hc / 2.0)) / (vol_pontoons + vol_colunas)

    P_R = 0.04 * Delta  # Peso/tração vertical dos risers (4% de Delta)
    P_A = 0.03 * Delta  # Componente vertical da amarração (3% de Delta)
    P_L = Delta - (P_topside + P_C + P_R + P_A)  # Peso de lastro (t)
    V_lastro = P_L / rho_agua  # Volume de lastro (m³)

    VCG_topside = hp + hc + 10.0
    momento_total = (P_topside * VCG_topside + P_C * VCG_C +
                     (P_R + P_A) * (hp + hc) + P_L * (hp / 2.0))
    KG = momento_total / Delta
    GM = KM - KG

    # 3) Área de Convés e Air Gap
    area_disp = Lp**2
    area_req  = 42.246 * Q + 4850.1  # Correlação da aula de pesos e centros
    airgap    = hp + hc - T          # Distância vertical da linha d'água ao fundo do convés

    # 4) Verificação das Restrições Estáticas e Geométricas
    restricoes = {
        "Area_Conves_OK":   area_disp >= area_req,               # Lp² >= 42.246*Q + 4850.1
        "GM_Min_OK":        GM >= 1.0,                           # GM_T >= 1.0 m
        "GM_Max_OK":        GM <= 4.0,                           # GM_T <= 4.0 m
        "Lastro_Pos_OK":    P_L > 0.0,                           # Peso de lastro positivo
        "Lastro_Vol_OK":    V_lastro <= 0.9 * vol_pontoons,      # V_lastro <= 0.9 * V_pontoons
        "Coluna_Pont_OK":   bc <= bp,                            # bc <= bp
        "Pontoon_Subm_OK":  hp < T                               # Pontoons totalmente submersos
    }

    resultados = {
        "vol_pontoons": vol_pontoons, "vol_colunas": vol_colunas,
        "nabla": nabla, "Delta": Delta,
        "KB": KB, "BM": BM, "KM": KM, "KG": KG, "GM": GM,
        "P_PL": P_PL, "P_topside": P_topside, "P_C": P_C,
        "P_R": P_R, "P_A": P_A, "P_L": P_L, "V_lastro": V_lastro,
        "area_disp": area_disp, "area_req": area_req, "airgap": airgap,
        "restricoes": restricoes,
        "estatica_viavel": all(restricoes.values())
    }
    return resultados

# Teste rápido da Etapa 2 usando 'entrada_teste' da Etapa 1
res_etapa2 = calcular_hidrostatica(entrada_teste)

print("=== ETAPA 2: RESULTADOS HIDROSTÁTICOS E RESTRIÇÕES ESTÁTICAS ===")
print(f"Deslocamento (Δ):  {res_etapa2['Delta']:,.1f} t | Peso Casco (P_C): {res_etapa2['P_C']:,.1f} t")
print(f"Peso Lastro (P_L): {res_etapa2['P_L']:,.1f} t | Vol. Lastro: {res_etapa2['V_lastro']:,.1f} / {0.9*res_etapa2['vol_pontoons']:,.1f} m³")
print(f"KB = {res_etapa2['KB']:.2f} m | BM = {res_etapa2['BM']:.2f} m | KG = {res_etapa2['KG']:.2f} m | GM = {res_etapa2['GM']:.2f} m")
print(f"Área Convés: {res_etapa2['area_disp']:,.1f} / {res_etapa2['area_req']:,.1f} m² | Air gap: {res_etapa2['airgap']:.2f} m")
print("\nStatus das Restrições Estáticas:")
for nome_r, status_r in res_etapa2["restricoes"].items():
    print(f"  - {nome_r:<18}: {'ATENDE' if status_r else 'FALHA'}")

# =========================================================================
# 3. CURVA GZ (PROJEÇÕES 2D) E CRITÉRIO DE VENTO (DNV-OS-C301 / MODU)
# =========================================================================

def _recortar_poligono_2d(vertices, z0, phi_rad):
    """
    Aplica o algoritmo de Sutherland-Hodgman para encontrar a área e o centroide
    da porção submersa de um retângulo 2D inclinado de um ângulo phi_rad.
    """
    poly_sub = []
    tan_phi = np.tan(phi_rad)
    # Distância vertical de cada vértice até a linha d'água inclinada (d >= 0 está submerso)
    d = [z0 + u * tan_phi - z for u, z in vertices]
    n = len(vertices)

    for i in range(n):
        p1, d1 = vertices[i], d[i]
        p2, d2 = vertices[(i + 1) % n], d[(i + 1) % n]
        if d1 >= 0:
            poly_sub.append(p1)
        if d1 * d2 < 0:  # Aresta cruza a linha d'água
            s = d1 / (d1 - d2)
            poly_sub.append((p1[0] + s * (p2[0] - p1[0]), p1[1] + s * (p2[1] - p1[1])))

    if len(poly_sub) < 3:
        return 0.0, 0.0, 0.0

    area_2x, sum_u, sum_z = 0.0, 0.0, 0.0
    m = len(poly_sub)
    for i in range(m):
        u1, z1 = poly_sub[i]
        u2, z2 = poly_sub[(i + 1) % m]
        cross = u1 * z2 - u2 * z1
        area_2x += cross
        sum_u += (u1 + u2) * cross
        sum_z += (z1 + z2) * cross

    area = area_2x / 2.0
    if area <= 1e-9:
        return 0.0, 0.0, 0.0
    return area, sum_u / (6.0 * area), sum_z / (6.0 * area)

def _coeficiente_altura_Ch(h):
    """Coeficiente de altura Ch conforme tabela da DNV-OS-C301 / Código MODU."""
    if h <= 15.3: return 1.00
    if h <= 30.5: return 1.10
    if h <= 46.0: return 1.20
    if h <= 61.0: return 1.30
    return 1.37

def _avaliar_plano_vento(dados, res_estatica, plano, passo_deg=1.0):
    """
    Calcula a curva GZ (0 a 50 deg), ângulo de alagamento, braço emborcador HA(phi),
    interceptos phi1 e phi2, e áreas A e B para um plano ('Transversal' ou 'Longitudinal').
    """
    Lp, bp, hp = dados["Lp"], dados["bp"], dados["hp"]
    bc, hc, T  = dados["bc"], dados["hc"], dados["T"]
    dp         = dados["dp"]
    xc         = Lp / 2.0 - bc / 2.0

    nabla    = res_estatica["nabla"]
    KG       = res_estatica["KG"]
    airgap   = res_estatica["airgap"]
    Delta_MN = res_estatica["Delta"] * g / 1000.0  # Peso deslocado em MN

    # 1) Definição das seções 2D e comprimentos extrudados no plano escolhido
    if plano == "Transversal":
        secoes_2d = [
            ([(-dp/2 - bp/2, 0), (-dp/2 + bp/2, 0), (-dp/2 + bp/2, hp), (-dp/2 - bp/2, hp)], Lp),
            ([( dp/2 - bp/2, 0), ( dp/2 + bp/2, 0), ( dp/2 + bp/2, hp), ( dp/2 - bp/2, hp)], Lp),
            ([(-dp/2 - bc/2, hp), (-dp/2 + bc/2, hp), (-dp/2 + bc/2, hp+hc), (-dp/2 - bc/2, hp+hc)], 2*bc),
            ([( dp/2 - bc/2, hp), ( dp/2 + bc/2, hp), ( dp/2 + bc/2, hp+hc), ( dp/2 - bc/2, hp+hc)], 2*bc),
        ]
        A_pont_sub = Lp * hp
        A_col_sub  = 2 * bc * (T - hp)
    else:  # == Longitudinal
        secoes_2d = [
            ([(-Lp/2, 0), (Lp/2, 0), (Lp/2, hp), (-Lp/2, hp)], 2*bp),
            ([(-xc - bc/2, hp), (-xc + bc/2, hp), (-xc + bc/2, hp+hc), (-xc - bc/2, hp+hc)], 2*bc),
            ([( xc - bc/2, hp), ( xc + bc/2, hp), ( xc + bc/2, hp+hc), ( xc - bc/2, hp+hc)], 2*bc),
        ]
        A_pont_sub = 2 * bp * hp
        A_col_sub  = 2 * bc * (T - hp)

    # Centro de resistência lateral submerso (z_CLR) em repouso
    z_CLR = (A_pont_sub * (hp / 2.0) + A_col_sub * (hp + (T - hp) / 2.0)) / (A_pont_sub + A_col_sub)

    # 2) Ângulo de alagamento (fundo do convés toca a linha d'água na borda Lp/2)
    phi_al_deg = np.degrees(np.arctan(airgap / (Lp / 2.0)))

    # 3) Curva GZ de 0° a 50° mantendo o volume deslocado constante (nabla)
    phi_deg = np.arange(0.0, 50.0 + passo_deg, passo_deg)
    GZ = np.zeros_like(phi_deg)

    for idx, phi in enumerate(phi_deg):
        if phi == 0.0:
            GZ[idx] = 0.0
            continue
        phi_rad = np.radians(phi)

        def balanco_volume(z0_teste):
            vol = sum(_recortar_poligono_2d(v, z0_teste, phi_rad)[0] * comp for v, comp in secoes_2d)
            return vol - nabla

        try:
            z0_eq = opt.brentq(balanco_volume, -Lp, 2.0 * (hp + hc))
        except ValueError:
            GZ[idx] = -1.0
            continue

        vol_total, mom_u, mom_z = 0.0, 0.0, 0.0
        for verts, comp in secoes_2d:
            area, uc, zc = _recortar_poligono_2d(verts, z0_eq, phi_rad)
            if area > 0:
                vol_total += area * comp
                mom_u += area * comp * uc
                mom_z += area * comp * zc

        if vol_total > 0:
            u_B = mom_u / vol_total
            z_B = mom_z / vol_total
            GZ[idx] = u_B * np.cos(phi_rad) + (z_B - KG) * np.sin(phi_rad)

    # Integração da Área A sob a curva GZ de 0 até phi_al (em radianos)
    idx_al = np.argmin(np.abs(phi_deg - phi_al_deg))
    phi_int_rad = np.radians(phi_deg[:idx_al + 1])
    Area_A = simpson(y=GZ[:idx_al + 1], x=phi_int_rad) if idx_al >= 2 else 0.0

    # 4) Cálculo do Momento e Braço de Vento (HA) para as duas velocidades normativas
    resultados_velocidades = {}
    for chave_v, V_vento in [("100_nos", V_vento_temp), ("70_nos", V_vento_oper)]:
        q = 0.5 * rho_ar * (V_vento**2)  # Pressão dinâmica do vento (kPa)

        # Parcelas verticais expostas sem inclinação (HM0)
        parcelas = [
            {"Componente": "Colunas",  "Area_m2": 4 * bc * airgap, "h_m": airgap / 2.0,           "Cs": 1.0},
            {"Componente": "Deck box", "Area_m2": Lp * hd,         "h_m": airgap + hd / 2.0,      "Cs": 1.0},
            {"Componente": "Módulos",  "Area_m2": Lp * hm,         "h_m": airgap + hd + hm / 2.0, "Cs": 1.1},
        ]

        HM0 = 0.0
        tabela_forcas = []
        for p in parcelas:
            Ch = _coeficiente_altura_Ch(p["h_m"])
            F_MN = q * p["Cs"] * Ch * p["Area_m2"] / 1000.0
            braco = p["h_m"] + T - z_CLR
            M_MNm = F_MN * braco
            HM0 += M_MNm
            tabela_forcas.append({
                "Componente": p["Componente"], "Area (m²)": p["Area_m2"],
                "h (m)": p["h_m"], "Ch": Ch, "Cs": p["Cs"],
                "Força (MN)": F_MN, "Braço (m)": braco, "Momento (MN·m)": M_MNm
            })

        # Parcela da face inferior do convés (HM_ud)
        Ch_ud = _coeficiente_altura_Ch(airgap)
        F_ud = q * 1.0 * Ch_ud * (Lp**2) / 1000.0
        braco_ud = airgap + T - z_CLR
        HM_ud = F_ud * braco_ud
        tabela_forcas.append({
            "Componente": "Face inf. convés (ud)", "Area (m²)": Lp**2,
            "h (m)": airgap, "Ch": Ch_ud, "Cs": 1.0,
            "Força (MN)": F_ud, "Braço (m)": braco_ud, "Momento (MN·m)": HM_ud
        })

        # Curva HA(phi) e Área B até phi_al
        phi_all_rad = np.radians(phi_deg)
        HA = (HM0 * (np.cos(phi_all_rad)**2) + HM_ud * (np.sin(phi_all_rad)**2)) / Delta_MN
        Area_B = simpson(y=HA[:idx_al + 1], x=phi_int_rad) if idx_al >= 2 else 1e9
        razao_AB = Area_A / Area_B if Area_B > 0 else 0.0

        # Interceptos phi1 (1º cruzamento GZ = HA) e phi2 (2º cruzamento GZ = HA)
        diff = GZ - HA
        phi1, phi2 = None, None
        for i in range(1, len(phi_deg)):
            if diff[i-1] <= 0 and diff[i] > 0 and phi1 is None:
                phi1 = phi_deg[i]
            elif diff[i-1] >= 0 and diff[i] < 0 and phi1 is not None and phi2 is None:
                phi2 = phi_deg[i]

        # Verificação GZ > 0 entre 0° e min(phi2, phi_al)
        limite_gz = phi2 if (phi2 is not None and phi2 < phi_al_deg) else phi_al_deg
        idx_lim = np.argmin(np.abs(phi_deg - limite_gz))
        gz_positivo = bool(np.all(GZ[1:idx_lim + 1] > 0)) if idx_lim >= 1 else False

        atende_criterio = (razao_AB >= 1.3) and gz_positivo and (phi1 is not None and phi1 < phi_al_deg)

        resultados_velocidades[chave_v] = {
            "V_ms": V_vento, "z_CLR": z_CLR, "HM0": HM0, "HM_ud": HM_ud,
            "phi_al": phi_al_deg, "phi1": phi1, "phi2": phi2,
            "Area_A": Area_A, "Area_B": Area_B, "Razao_AB": razao_AB,
            "GZ_positivo_OK": gz_positivo, "Atende": atende_criterio,
            "HA_curve": HA, "tabela_forcas": tabela_forcas
        }

    return {
        "plano": plano, "phi_deg": phi_deg, "GZ_curve": GZ,
        "phi_al": phi_al_deg, "velocidades": resultados_velocidades
    }

def calcular_estabilidade_vento(dados, res_estatica, passo_deg=1.0):
    """Executa a verificação nos 4 casos obrigatórios (2 planos x 2 velocidades)."""
    trans = _avaliar_plano_vento(dados, res_estatica, "Transversal", passo_deg=passo_deg)
    long  = _avaliar_plano_vento(dados, res_estatica, "Longitudinal", passo_deg=passo_deg)

    todos_atendem = (
        trans["velocidades"]["100_nos"]["Atende"] and trans["velocidades"]["70_nos"]["Atende"] and
        long["velocidades"]["100_nos"]["Atende"]  and long["velocidades"]["70_nos"]["Atende"]
    )
    menor_AB = min(
        trans["velocidades"]["100_nos"]["Razao_AB"], trans["velocidades"]["70_nos"]["Razao_AB"],
        long["velocidades"]["100_nos"]["Razao_AB"],  long["velocidades"]["70_nos"]["Razao_AB"]
    )

    return {
        "Transversal": trans,
        "Longitudinal": long,
        "vento_viavel": todos_atendem,
        "menor_AB": menor_AB
    }

# Teste rápido da Etapa 3
res_etapa3 = calcular_estabilidade_vento(entrada_teste, res_etapa2, passo_deg=0.5)

print("=== ETAPA 3: VERIFICAÇÃO DO CRITÉRIO DE VENTO (4 CASOS) ===")
resumo_v = []
for pl in ["Transversal", "Longitudinal"]:
    for vel_k, vel_lbl in [("100_nos", "51,5 m/s (Tempestade)"), ("70_nos", "36,0 m/s (Operação)")]:
        d = res_etapa3[pl]["velocidades"][vel_k]
        resumo_v.append({
            "Plano": pl, "Cenário Vento": vel_lbl,
            "φ1 (°)": f"{d['phi1']:.1f}" if d["phi1"] else "-",
            "φ2 (°)": f"{d['phi2']:.1f}" if d["phi2"] else "> φ_al",
            "φ_al (°)": f"{d['phi_al']:.1f}",
            "Área A": f"{d['Area_A']:.3f}", "Área B": f"{d['Area_B']:.3f}",
            "A/B (>=1.30)": f"{d['Razao_AB']:.2f}",
            "Status": "ATENDE" if d["Atende"] else "FALHA"
        })
print(pd.DataFrame(resumo_v).to_string(index=False))

# =========================================================================
# 4. COMPORTAMENTO EM ONDAS (HEAVE) E AIR GAP MÍNIMO DINÂMICO
# =========================================================================

def calcular_heave_e_airgap(dados, res_estatica, num_freq=1000):
    """
    Calcula:
      6) Forças de heave, massa adicional, frequência e período natural e RAO
      7) Espectro de mar JONSWAP, espectro de resposta e Hs de heave
      8) Air gap mínimo (eta_max + |z_min|) e verificação das restrições dinâmicas
    """
    bp = dados["bp"]
    hp = dados["hp"]
    bc = dados["bc"]
    lc = dados["lc"]
    ld = dados["ld"]
    dp = dados["dp"]
    T  = dados["T"]
    Hs = dados["Hs"]
    Tp = dados["Tp"]

    M      = res_estatica["Delta"]   # Massa total deslocada (t)
    airgap = res_estatica["airgap"]  # Air gap em repouso: a = hp + hc - T (m)

    #  1) Massa Adicional, Restauração e Frequência Natural 
    Ma2D   = 1.51 * rho_agua * np.pi * ((bp / 2.0)**2)  # t/m (DNV)
    Ma     = 2.0 * Ma2D * ld                            # t (trechos descobertos dos pontoons)
    c_rest = 4.0 * rho_agua * g * bc * lc               # Restauração hidrostática (kN/m)

    wn = np.sqrt(c_rest / (M + Ma))                     # Frequência natural de heave (rad/s)
    Tn = 2.0 * np.pi / wn                               # Período natural de heave (s)

    #  2) Forças de Excitação por Unidade de Amplitude (F / A) e RAO 
    omega = np.linspace(0.01, 1.50, num_freq)
    termo_FK = rho_agua * (g**2) * np.exp(-(omega**2) * T / g) * np.cos((omega**2) * dp / (2.0 * g))

    F_FKD_A = (4.0 * ld * termo_FK
               * (np.sin((omega**2) * bp / (2.0 * g)) / (omega**2))
               * (1.0 - np.exp((omega**2) * hp / g)))

    F_FKC_A = (8.0 * lc * termo_FK
               * (np.sin((omega**2) * bc / (2.0 * g)) / (omega**2)))

    F_FK_A  = F_FKD_A + F_FKC_A

    F_MA_A  = (-2.0 * Ma2D * ld * (omega**2)
               * np.exp(-(omega**2) * (T - hp / 2.0) / g)
               * np.cos((omega**2) * dp / (2.0 * g)))

    F_tot_A = F_FK_A + F_MA_A

    razao_freq = omega / wn
    fator_din  = np.sqrt((1.0 - razao_freq**2)**2 + (2.0 * xi_heave * razao_freq)**2)
    RAO_heave  = np.abs(F_tot_A / c_rest) / fator_din

    #  3) Espectro do Mar (JONSWAP) e Espectro de Resposta em Heave 
    wp = 2.0 * np.pi / Tp
    alpha_star = 0.0624 / (0.230 + 0.0336 * gamma_jonswap - 0.185 * ((1.9 + gamma_jonswap)**-1))
    tau = np.where(omega < wp, 0.07, 0.09)

    expoente_gamma = np.exp(-((omega - wp)**2) / (2.0 * (tau**2) * (wp**2)))
    S_zeta = (alpha_star * (Hs**2) * (omega**-5) / (wp**-4)
              * np.exp(-1.25 * ((omega / wp)**-4)) * (gamma_jonswap**expoente_gamma))

    S_heave = (RAO_heave**2) * S_zeta 

    m0_mar   = simpson(y=S_zeta, x=omega)
    Hs_check = 4.0 * np.sqrt(m0_mar)

    m0_z     = simpson(y=S_heave, x=omega) 
    Hs_heave = 4.0 * np.sqrt(m0_z)          # Altura significativa de heave H_{1/3,z} (m)
    Amp_sig  = 2.0 * np.sqrt(m0_z)          # Amplitude significativa de heave (m)

    #  4) Air Gap Mínimo Dinâmico (Seção 4 do Enunciado)
    eta_max    = (1.86 * Hs) / 2.0          # Crista máxima da onda (m)
    z_min_abs  = (1.86 * Hs_heave) / 2.0    # Maior excursão para baixo em heave |z_min| (m)
    airgap_min = eta_max + z_min_abs        # a_min = eta_max + |z_min| (m)
    folga_ag   = airgap - airgap_min        # Folga entre a crista máxima e o fundo do convés

    #  5) Restrições Dinâmicas e de Air Gap
    restricoes_din = {
        "Periodo_Tn_OK": Tn > 20.0,               # Tn > 20 s
        "AirGap_Min_OK": airgap >= airgap_min     # a >= a_min
    }

    return {
        "Ma2D": Ma2D, "Ma": Ma, "c_rest": c_rest,
        "wn": wn, "Tn": Tn,
        "omega": omega, "F_FKD_A": F_FKD_A, "F_FKC_A": F_FKC_A,
        "F_MA_A": F_MA_A, "F_tot_A": F_tot_A, "RAO_heave": RAO_heave,
        "S_zeta": S_zeta, "S_heave": S_heave, "Hs_check": Hs_check,
        "Hs_heave": Hs_heave, "Amp_sig": Amp_sig,
        "eta_max": eta_max, "z_min_abs": z_min_abs,
        "airgap": airgap, "airgap_min": airgap_min, "folga_airgap": folga_ag,
        "restricoes": restricoes_din,
        "dinamica_viavel": all(restricoes_din.values())
    }

# Teste rápido da Etapa 4
res_etapa4 = calcular_heave_e_airgap(entrada_teste, res_etapa2)

print("=== ETAPA 4: COMPORTAMENTO EM ONDAS E AIR GAP DINÂMICO ===")
print(f"Massa Adicional (Ma):         {res_etapa4['Ma']:,.1f} t")
print(f"Frequência Natural (wn):      {res_etapa4['wn']:.4f} rad/s | Período Natural (Tn): {res_etapa4['Tn']:.2f} s")
print(f"RAO Máximo (Ressonância):     {np.max(res_etapa4['RAO_heave']):.2f} m/m")
print(f"Altura Signif. Heave (H1/3,z):{res_etapa4['Hs_heave']:.3f} m (Amp. Sig.: {res_etapa4['Amp_sig']:.3f} m)")
print(f"Crista Máxima (η_max):        {res_etapa4['eta_max']:.2f} m")
print(f"Excursão Mínima (|z_min|):    {res_etapa4['z_min_abs']:.2f} m")
print(f"Air Gap Mínimo (a_min):       {res_etapa4['airgap_min']:.2f} m | Air Gap Atual (a): {res_etapa4['airgap']:.2f} m (Folga: {res_etapa4['folga_airgap']:+.2f} m)")
print("\nStatus das Restrições Dinâmicas:")
for nome_r, status_r in res_etapa4["restricoes"].items():
    print(f"  - {nome_r:<18}: {'ATENDE' if status_r else 'FALHA'}")

# =========================================================================
# 5. OTIMIZAÇÃO MULTIOBJETIVO COM DUAL ANNEALING E FRONTEIRA DE PARETO
# =========================================================================

# Limites de busca das 6 variáveis de projeto: [Lp, bp, hp, bc, hc, T]
# Para BM-S-52 (Q = 230 mil BOE/dia), Lp_min = sqrt(42.246 * 230 + 4850.1) = 120.69 m
Lp_min = np.sqrt(42.246 * Q_prod + 4850.1)
bounds_sa = [
    (Lp_min, 185.0),  # Lp: comprimento/boca total (m)
    (12.0,   35.0),   # bp: boca dos pontoons (m)
    (8.0,    25.0),   # hp: altura dos pontoons (m)
    (10.0,   30.0),   # bc: largura das colunas quadradas (m)
    (25.0,   65.0),   # hc: altura das colunas (m)
    (20.0,   45.0),   # T:  calado de projeto (m)
]

solucoes_viaveis_sa = []

def avaliar_candidata_sa(x, w1):
    """
    Integra as Etapas 1 a 4 e retorna o valor da função objetivo ponderada
    (com penalidades contínuas para guiar o Simulated Annealing à região viável).
    """
    Lp, bp, hp, bc, hc, T = x
    w2 = 1.0 - w1

    # Etapa 1: Empacotar entradas
    dados = criar_entradas(Lp, bp, hp, bc, hc, T)

    # Etapa 2: Hidrostática, Pesos e Restrições Estáticas
    res_est = calcular_hidrostatica(dados)

    # Etapa 4: Hidrodinâmica (Heave) e Air Gap Mínimo Dinâmico
    res_din = calcular_heave_e_airgap(dados, res_est, num_freq=350)

    # Cálculo das violações (penalidades contínuas)
    penalidade = 0.0
    if res_est["area_disp"] < res_est["area_req"]:
        penalidade += 100.0 * (res_est["area_req"] - res_est["area_disp"])
    if res_est["GM"] < 1.0:
        penalidade += 500.0 * (1.0 - res_est["GM"])
    elif res_est["GM"] > 4.0:
        penalidade += 500.0 * (res_est["GM"] - 4.0)
    if res_est["P_L"] <= 0.0:
        penalidade += 10.0 * (1.0 - res_est["P_L"])
    if res_est["V_lastro"] > 0.9 * res_est["vol_pontoons"]:
        penalidade += 10.0 * (res_est["V_lastro"] - 0.9 * res_est["vol_pontoons"])
    if bc > bp:
        penalidade += 500.0 * (bc - bp)
    if hp > T - 2.0:
        penalidade += 500.0 * (hp - (T - 2.0))
    if res_din["Tn"] <= 20.0:
        penalidade += 200.0 * (20.05 - res_din["Tn"])
    if res_din["airgap"] < res_din["airgap_min"]:
        penalidade += 300.0 * (res_din["airgap_min"] - res_din["airgap"])

    # Se violou alguma restrição rápida, retorna penalidade sem gastar tempo no recorte 2D
    if penalidade > 0.0:
        return 1e5 + penalidade

    # Etapa 3: Curva GZ e Critério de Vento (4 casos)
    res_ven = calcular_estabilidade_vento(dados, res_est, passo_deg=1.0)
    if not res_ven["vento_viavel"]:
        penalidade_vento = 0.0
        for pl in ["Transversal", "Longitudinal"]:
            for vk in ["100_nos", "70_nos"]:
                dv = res_ven[pl]["velocidades"][vk]
                if dv["Razao_AB"] < 1.30:
                    penalidade_vento += 2000.0 * (1.30 - dv["Razao_AB"])
                if dv["phi1"] is None or dv["phi1"] >= dv["phi_al"]:
                    penalidade_vento += 1500.0
                if not dv["GZ_positivo_OK"]:
                    penalidade_vento += 1500.0
        return 1e5 + max(50.0, penalidade_vento)

    # Solução 100% viável! Registra no histórico para montar o espaço de soluções
    P_C = res_est["P_C"]
    Hs_heave = res_din["Hs_heave"]

    solucoes_viaveis_sa.append({
        "Lp": Lp, "bp": bp, "hp": hp, "bc": bc, "hc": hc, "T": T,
        "P_C": P_C, "Hs_heave": Hs_heave, "GM": res_est["GM"],
        "Tn": res_din["Tn"], "airgap": res_din["airgap"],
        "airgap_min": res_din["airgap_min"], "menor_AB": res_ven["menor_AB"],
        "P_L": res_est["P_L"], "V_lastro": res_est["V_lastro"],
        "vol_pontoons": res_est["vol_pontoons"], "Delta": res_est["Delta"]
    })

    # Função Objetivo Ponderada
    return w1 * Hs_heave + w2 * (P_C / 10000.0)

print("Iniciando otimização com Dual Annealing (Simulated Annealing)...")
pesos_w1 = np.linspace(0.0, 1.0, 10)  # 15 combinações de pesos entre 0 e 1
resultados_pareto_sa = []

for idx, w1 in enumerate(pesos_w1):
    res_opt = opt.dual_annealing(
        func=avaliar_candidata_sa,
        bounds=bounds_sa,
        args=(w1,),
        maxiter=120,           # Número de ciclos de resfriamento
        initial_temp=3000.0,   # Temperatura inicial padrão do Generalized SA
        visit=2.62,            # Parâmetro de distribuição de salto
        no_local_search=False, # Mantém o refinamento local contínuo
        seed=100 + idx,
        # accept = -5            # Taxa de aceite do SA
    )

    viavel_rodada = res_opt.fun < 1e5
    if viavel_rodada:
        Lp, bp, hp, bc, hc, T = res_opt.x
        d_opt = criar_entradas(Lp, bp, hp, bc, hc, T)
        e_opt = calcular_hidrostatica(d_opt)
        h_opt = calcular_heave_e_airgap(d_opt, e_opt, num_freq=500)
        v_opt = calcular_estabilidade_vento(d_opt, e_opt, passo_deg=1.0)
        print(f"[{idx+1:02d}/{len(pesos_w1)}] w1={w1:.2f} -> VIÁVEL | P_C={e_opt['P_C']:,.0f} t | Hs_heave={h_opt['Hs_heave']:.3f} m | GM={e_opt['GM']:.2f}m | Tn={h_opt['Tn']:.1f}s | A/B={v_opt['menor_AB']:.2f}")
    else:
        print(f"[{idx+1:02d}/{len(pesos_w1)}] w1={w1:.2f} -> Não convergiu para região viável (Penalidade: {res_opt.fun - 1e5:.1f})")

df_viaveis = pd.DataFrame(solucoes_viaveis_sa)

if len(df_viaveis) > 0:
    # Ordena pelo menor Peso Estrutural (P_C) e desempata pelo menor Hs_heave
    df_viaveis = df_viaveis.sort_values(by=["P_C", "Hs_heave"]).reset_index(drop=True)

    indices_pareto = []
    menor_hs_atual = np.inf
    for i, row in df_viaveis.iterrows():
        if row["Hs_heave"] < menor_hs_atual - 1e-4:
            indices_pareto.append(i)
            menor_hs_atual = row["Hs_heave"]

    df_pareto = df_viaveis.loc[indices_pareto].reset_index(drop=True)
    print(f"\nResumo da Otimização:")
    print(f"  - Total de plataformas viáveis avaliadas: {len(df_viaveis)}")
    print(f"  - Pontos na Fronteira de Pareto:       {len(df_pareto)}")
else:
    print("\nNenhuma solução viável registrada. Experimente aumentar 'maxiter' para 200.")

# =========================================================================
# 6. EXTRAÇÃO DOS 3 CENÁRIOS, TABELAS E GRÁFICOS FINAIS (BM-S-52)
# =========================================================================

if len(df_pareto) == 0:
    raise RuntimeError("O DataFrame 'df_pareto' está vazio. Rode a Etapa 5 primeiro!")

# 1. Seleção dos 3 Cenários na Fronteira de Pareto
# Cenário 1: Menor Peso Estrutural (primeiro ponto ordenado por P_C)
row_c1 = df_pareto.iloc[0]

# Cenário 2: Menor Movimento de Heave (último ponto ordenado por P_C)
row_c2 = df_pareto.iloc[-1]

# Cenário 3: Solução Intermediária ("Joelho" da curva - menor distância normalizada ao ponto utopia)
pc_min, pc_max = df_pareto["P_C"].min(), df_pareto["P_C"].max()
hs_min, hs_max = df_pareto["Hs_heave"].min(), df_pareto["Hs_heave"].max()

pc_norm = (df_pareto["P_C"] - pc_min) / (pc_max - pc_min + 1e-9)
hs_norm = (df_pareto["Hs_heave"] - hs_min) / (hs_max - hs_min + 1e-9)
idx_joelho = np.sqrt(pc_norm**2 + hs_norm**2).idxmin()
row_c3 = df_pareto.iloc[idx_joelho]

def reavaliar_detalhado(row):
    """Recalcula um cenário com alta resolução (passo 0.5 deg e 1000 frequências)."""
    d = criar_entradas(row["Lp"], row["bp"], row["hp"], row["bc"], row["hc"], row["T"])
    e = calcular_hidrostatica(d)
    v = calcular_estabilidade_vento(d, e, passo_deg=0.5)
    h = calcular_heave_e_airgap(d, e, num_freq=1000)
    return {"dados": d, "est": e, "ven": v, "din": h}

c1 = reavaliar_detalhado(row_c1)
c2 = reavaliar_detalhado(row_c2)
c3 = reavaliar_detalhado(row_c3)


fig_pareto = go.Figure()

# Nuvem de pontos viáveis
df_nuvem = df_viaveis.sample(min(3000, len(df_viaveis)), random_state=42)

fig_pareto.add_trace(go.Scatter(
    x=df_nuvem["P_C"], y=df_nuvem["Hs_heave"],
    mode="markers", name="Soluções Viáveis",
    marker=dict(color="#E8A0BF", size=5, opacity=0.7)
))

# Fronteira de Pareto — curva azul suave
if len(df_pareto) >= 4:
    _x = df_pareto["P_C"].values
    _y = df_pareto["Hs_heave"].values
    _order = np.argsort(_x)
    _x, _y = _x[_order], _y[_order]
    _t = np.linspace(0, 1, len(_x))
    _t_fine = np.linspace(0, 1, 300)
    _k = min(3, len(_x) - 1)
    _spline_x = make_interp_spline(_t, _x, k=_k)(_t_fine)
    _spline_y = make_interp_spline(_t, _y, k=_k)(_t_fine)
    fig_pareto.add_trace(go.Scatter(
        x=_spline_x, y=_spline_y,
        mode="lines", name="Fronteira de Pareto",
        line=dict(color="royalblue", width=2.5, shape="spline")
    ))
elif len(df_pareto) >= 2:
    fig_pareto.add_trace(go.Scatter(
        x=df_pareto["P_C"], y=df_pareto["Hs_heave"],
        mode="lines", name="Fronteira de Pareto",
        line=dict(color="royalblue", width=2.5)
    ))

# Marcadores dos 3 Cenários
marcadores = [
    (c1, "Cenário 1: Mínimo Peso", "green", "diamond"),
    (c2, "Cenário 2: Mínimo Heave", "crimson", "square"),
    (c3, "Cenário 3: Solução Escolhida", "darkorange", "star")
]

for cen, rotulo, cor, simb in marcadores:
    fig_pareto.add_trace(go.Scatter(
        x=[cen["est"]["P_C"]], y=[cen["din"]["Hs_heave"]],
        mode="markers+text", name=rotulo,
        text=[rotulo.split(":")[0]], textposition="top right",
        marker=dict(
            color=cor, size=14, symbol=simb,
            line=dict(width=1.5, color="black")
        )
    ))

# Layout do gráfico
fig_pareto.update_layout(
    title="Conjunto de Soluções Viáveis (Campo BM-S-52)",
    xaxis_title="Peso Estrutural do Casco - P_C (t)",
    yaxis_title="Altura Significativa de Heave - H1/3,z (m)",
    width=700,
    height=700
)

fig_pareto.show()


def _col_ficha(c):
    d, e, v, h = c["dados"], c["est"], c["ven"], c["din"]
    return [
        f"{d['Lp']:.2f} m", f"{d['bp']:.2f} m", f"{d['hp']:.2f} m",
        f"{d['bc']:.2f} m", f"{d['hc']:.2f} m", f"{d['T']:.2f} m",
        f"{e['Delta']:,.0f} t", f"{e['P_C']:,.0f} t", f"{e['P_topside']:,.0f} t", f"{e['P_L']:,.0f} t",
        f"{e['KB']:.2f} m", f"{e['BM']:.2f} m", f"{e['KM']:.2f} m", f"{e['KG']:.2f} m", f"{e['GM']:.2f} m",
        f"{e['area_disp']:,.0f} m²",
        f"{e['airgap']:.2f} m (mín: {h['airgap_min']:.2f} m)",
        f"{v['menor_AB']:.2f}",
        f"{h['Ma']:,.0f} t", f"{h['Tn']:.2f} s", f"{h['Hs_heave']:.3f} m"
    ]

df_ficha = pd.DataFrame({
    "Parâmetro": [
        "Comprimento/Boca Total (Lp)", "Boca do Pontoon (bp)", "Altura do Pontoon (hp)",
        "Lado da Coluna (bc)", "Altura da Coluna (hc)", "Calado (T)",
        "Deslocamento (Δ)", "Peso Estrutural Casco (P_C)", "Peso Topside (P_topside)", "Peso de Lastro (P_L)",
        "Centro de Carena (KB)", "Raio Metacêntrico (BM)", "Metacentro (KM)", "Centro de Gravidade (KG)", "Altura Metacêntrica (GM)",
        "Área de Convés Disponível",
        "Air Gap em Repouso (a)",
        "Menor Razão de Vento (A/B)",
        "Massa Adicional Heave (Ma)", "Período Natural Heave (Tn)", "Altura Significativa Heave (H1/3,z)"
    ],
    "Cenário 1 (Mín. Peso)": _col_ficha(c1),
    "Cenário 2 (Mín. Heave)": _col_ficha(c2),
    "Cenário 3 (Escolhida)": _col_ficha(c3),
})
print("\n=== FICHA TÉCNICA COMPARATIVA DOS 3 CENÁRIOS (CAMPO BM-S-52) ===")
print(df_ficha.to_string(index=False))

d3, e3, v3, h3 = c3["dados"], c3["est"], c3["ven"], c3["din"]
df_folgas = pd.DataFrame([
    {"Restrição": "Área de Convés >= Área Requerida", "Valor Obtido": f"{e3['area_disp']:,.1f} m²", "Limite": f">= {e3['area_req']:,.1f} m²", "Folga": f"+{e3['area_disp'] - e3['area_req']:,.1f} m²"},
    {"Restrição": "GM Mínimo (>= 1,0 m)", "Valor Obtido": f"{e3['GM']:.2f} m", "Limite": ">= 1,00 m", "Folga": f"+{e3['GM'] - 1.0:.2f} m"},
    {"Restrição": "GM Máximo (<= 4,0 m)", "Valor Obtido": f"{e3['GM']:.2f} m", "Limite": "<= 4,00 m", "Folga": f"+{4.0 - e3['GM']:.2f} m"},
    {"Restrição": "Período Natural Heave (Tn > 20 s)", "Valor Obtido": f"{h3['Tn']:.2f} s", "Limite": "> 20,00 s", "Folga": f"+{h3['Tn'] - 20.0:.2f} s"},
    {"Restrição": "Peso de Lastro Positivo (P_L > 0)", "Valor Obtido": f"{e3['P_L']:,.0f} t", "Limite": "> 0 t", "Folga": f"+{e3['P_L']:,.0f} t"},
    {"Restrição": "Vol. Lastro <= 90% Vol. Pontoons", "Valor Obtido": f"{e3['V_lastro']:,.0f} m³", "Limite": f"<= {0.9*e3['vol_pontoons']:,.0f} m³", "Folga": f"+{0.9*e3['vol_pontoons'] - e3['V_lastro']:,.0f} m³"},
    {"Restrição": "Coluna <= Pontoon (bc <= bp)", "Valor Obtido": f"{d3['bc']:.2f} m", "Limite": f"<= {d3['bp']:.2f} m", "Folga": f"+{d3['bp'] - d3['bc']:.2f} m"},
    {"Restrição": "Critério de Vento (Menor A/B >= 1,30)", "Valor Obtido": f"{v3['menor_AB']:.2f}", "Limite": ">= 1,30", "Folga": f"+{v3['menor_AB'] - 1.30:.2f}"},
    {"Restrição": "Air Gap Dinâmico (a >= η_max + |z_min|)", "Valor Obtido": f"{e3['airgap']:.2f} m", "Limite": f">= {h3['airgap_min']:.2f} m", "Folga": f"+{e3['airgap'] - h3['airgap_min']:.2f} m"},
])
print("\n=== TABELA-RESUMO DE RESTRIÇÕES E FOLGAS (CENÁRIO 3 - ESCOLHIDA) ===")
print(df_folgas.to_string(index=False))

print("\n=== FORÇAS, BRAÇOS E MOMENTOS DE VENTO POR PARCELA (CENÁRIO 3 - 51,5 m/s) ===")
for pl in ["Transversal", "Longitudinal"]:
    z_clr = v3[pl]["velocidades"]["100_nos"]["z_CLR"]
    print(f"\nPlano {pl} (z_CLR = {z_clr:.2f} m):")
    print(pd.DataFrame(v3[pl]["velocidades"]["100_nos"]["tabela_forcas"]).to_string(index=False))

print("\n=== RESUMO DO CRITÉRIO DE ÁREAS DE VENTO (CENÁRIO 3 - 4 CASOS) ===")
linhas_v = []
for pl in ["Transversal", "Longitudinal"]:
    for vk, vlbl in [("100_nos", "51,5 m/s (Tempestade)"), ("70_nos", "36,0 m/s (Operação)")]:
        dv = v3[pl]["velocidades"][vk]
        linhas_v.append({
            "Plano": pl, "Vento": vlbl,
            "φ1 (°)": f"{dv['phi1']:.1f}" if dv["phi1"] else "-",
            "φ2 (°)": f"{dv['phi2']:.1f}" if dv["phi2"] else "> φ_al",
            "φ_al (°)": f"{dv['phi_al']:.1f}",
            "Área A (m·rad)": f"{dv['Area_A']:.3f}",
            "Área B (m·rad)": f"{dv['Area_B']:.3f}",
            "A/B": f"{dv['Razao_AB']:.2f}",
            "Status": "ATENDE" if dv["Atende"] else "FALHA"
        })
print(pd.DataFrame(linhas_v).to_string(index=False))

fig_painel = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Estabilidade Transversal (GZ x HA)",
        "Estabilidade Longitudinal (GZ x HA)",
        "RAO de Heave vs. Frequência ω",
        "Espectro de Mar (BM-S-52) e Resposta em Heave"
    )
)

for idx_col, pl in enumerate(["Transversal", "Longitudinal"], start=1):
    phi_r = v3[pl]["phi_deg"]
    fig_painel.add_trace(go.Scatter(x=phi_r, y=v3[pl]["GZ_curve"], name="GZ", line=dict(color="teal", width=2.5), showlegend=(idx_col==1)), row=1, col=idx_col)
    fig_painel.add_trace(go.Scatter(x=phi_r, y=v3[pl]["velocidades"]["100_nos"]["HA_curve"], name="HA (51,5 m/s)", line=dict(color="crimson"), showlegend=(idx_col==1)), row=1, col=idx_col)
    fig_painel.add_trace(go.Scatter(x=phi_r, y=v3[pl]["velocidades"]["70_nos"]["HA_curve"], name="HA (36,0 m/s)", line=dict(color="orange"), showlegend=(idx_col==1)), row=1, col=idx_col)
    fig_painel.add_vline(x=v3[pl]["phi_al"], line_dash="dash", line_color="gray", annotation_text=f"φ_al={v3[pl]['phi_al']:.1f}°", row=1, col=idx_col)

fig_painel.add_trace(go.Scatter(x=h3["omega"], y=h3["RAO_heave"], name="RAO Heave", line=dict(color="royalblue", width=2.5)), row=2, col=1)
fig_painel.add_vline(x=h3["wn"], line_dash="dash", line_color="red", annotation_text=f"ωn={h3['wn']:.2f}", row=2, col=1)

fig_painel.add_trace(go.Scatter(x=h3["omega"], y=h3["S_zeta"], name="S_ζ(ω) Mar BM-S-52", line=dict(color="navy")), row=2, col=2)
fig_painel.add_trace(go.Scatter(x=h3["omega"], y=h3["S_heave"], name="S_z(ω) Heave", fill="tozeroy", line=dict(color="darkorange")), row=2, col=2)

fig_painel.update_xaxes(title_text="Ângulo φ (°)", range=[0, 50], row=1, col=1)
fig_painel.update_xaxes(title_text="Ângulo φ (°)", range=[0, 50], row=1, col=2)
fig_painel.update_xaxes(title_text="Frequência ω (rad/s)", range=[0, 1.5], row=2, col=1)
fig_painel.update_xaxes(title_text="Frequência ω (rad/s)", range=[0, 1.5], row=2, col=2)
fig_painel.update_layout(height=750, title_text="Análise Detalhada da Solução Escolhida (Cenário 3 - BM-S-52)")
fig_painel.show()

def _criar_caixa_3d(x0, x1, y0, y1, z0, z1, cor, nome, scene="scene", mostrar_legenda=True):
    """
    Constrói um paralelepípedo 3D 100% opaco com 6 faces independentes e
    arestas contornadas para eliminar qualquer transparência ou vazamento de faces.
    """
    # Vértices ordenados com normais apontando para fora (regra da mão direita)
    faces = [
        ([x0, x0, x1, x1], [y0, y1, y1, y0], [z0, z0, z0, z0]),  # Base (z0, normal -Z)
        ([x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]),  # Topo (z1, normal +Z)
        ([x0, x0, x0, x0], [y0, y0, y1, y1], [z0, z1, z1, z0]),  # Face X- (normal -X)
        ([x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]),  # Face X+ (normal +X)
        ([x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]),  # Face Y- (normal -Y)
        ([x0, x0, x1, x1], [y1, y1, y1, y1], [z0, z1, z1, z0]),  # Face Y+ (normal +Y)
    ]
    traces = []
    for idx, (fx, fy, fz) in enumerate(faces):
        traces.append(go.Mesh3d(
            x=fx, y=fy, z=fz,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=cor,
            opacity=1.0, 
            name=nome,
            scene=scene,
            showlegend=(mostrar_legenda and idx == 1),
            legendgroup=nome,
            flatshading=True,
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15, roughness=0.5)
        ))

    # Arestas da caixa para dar nitidez visual CAD
    bx = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
    by = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None, y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
    bz = [z0, z0, z0, z0, z0, None, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1, None, z0, z1]
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="rgba(20, 20, 20, 0.45)", width=2),
        name=nome,
        scene=scene,
        showlegend=False,
        legendgroup=nome
    ))
    return traces

def _gerar_traces_semisub_completa(cenario_dict, scene="scene", mostrar_legenda=True):
    d = cenario_dict["dados"]
    Lp, bp, hp = d["Lp"], d["bp"], d["hp"]
    bc, hc, T  = d["bc"], d["hc"], d["T"]

    traces = []
    meio_Lp = Lp / 2.0
    z_topo_col = hp + hc
    z_topo_deck = z_topo_col + hd
    z_topo_mod = z_topo_deck + hm

    # 1. Pontoons (sólidos)
    for sinal_y in [-1, 1]:
        yc = sinal_y * (Lp - bp) / 2.0
        sl = mostrar_legenda and (sinal_y == -1)
        traces += _criar_caixa_3d(
            -meio_Lp, meio_Lp, yc - bp/2.0, yc + bp/2.0, 0.0, hp,
            cor="#8B6914", nome="Pontoons", scene=scene, mostrar_legenda=sl
        )

    # 2. Colunas (sólidas)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            cx = sx * (Lp - bc) / 2.0
            cy = sy * (Lp - bc) / 2.0
            sl = mostrar_legenda and (sx == -1 and sy == -1)
            traces += _criar_caixa_3d(
                cx - bc/2.0, cx + bc/2.0, cy - bc/2.0, cy + bc/2.0, hp, z_topo_col,
                cor="#3B7EA1", nome="Colunas", scene=scene, mostrar_legenda=sl
            )

    # 3. Deck Box (sólido)
    traces += _criar_caixa_3d(
        -meio_Lp, meio_Lp, -meio_Lp, meio_Lp, z_topo_col, z_topo_deck,
        cor="#7F8C8D", nome="Deck Box", scene=scene, mostrar_legenda=mostrar_legenda
    )

    # 4. Módulos de Topside (sólidos)
    recuo = 0.08 * Lp
    traces += _criar_caixa_3d(
        -meio_Lp + recuo, meio_Lp - recuo, -meio_Lp + recuo, meio_Lp - recuo,
        z_topo_deck, z_topo_mod,
        cor="#D35400", nome="Módulos Topside", scene=scene, mostrar_legenda=mostrar_legenda
    )

    # 5. Plano d'água no calado T
    w = meio_Lp + 20.0
    traces.append(go.Mesh3d(
        x=[-w, w, w, -w], y=[-w, -w, w, w], z=[T, T, T, T],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#0077CC", opacity=0.20,
        name="Linha d'água (T)", scene=scene,
        showlegend=mostrar_legenda, legendgroup="Linha d'água (T)",
        flatshading=True, lighting=dict(ambient=1.0)
    ))
    traces.append(go.Scatter3d(
        x=[-w, w, w, -w, -w], y=[-w, -w, w, w, -w], z=[T]*5,
        mode="lines", line=dict(color="#004C99", width=3),
        name="Linha d'água (T)", scene=scene,
        showlegend=False, legendgroup="Linha d'água (T)"
    ))

    return traces

fig_3d = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=[
        f"Cenário 1: Mín. Peso<br>P_C={c1['est']['P_C']:,.0f}t | Hs_z={c1['din']['Hs_heave']:.2f}m",
        f"Cenário 3: Escolhida<br>P_C={c3['est']['P_C']:,.0f}t | Hs_z={c3['din']['Hs_heave']:.2f}m",
        f"Cenário 2: Mín. Heave<br>P_C={c2['est']['P_C']:,.0f}t | Hs_z={c2['din']['Hs_heave']:.2f}m"
    ]
)

cenarios_3d = [(c1, "scene", True), (c3, "scene2", False), (c2, "scene3", False)]
for cen_obj, sc_name, leg in cenarios_3d:
    col_idx = 1 if sc_name == "scene" else (2 if sc_name == "scene2" else 3)
    for tr in _gerar_traces_semisub_completa(cen_obj, scene=sc_name, mostrar_legenda=leg):
        fig_3d.add_trace(tr, row=1, col=col_idx)

cam_view = dict(eye=dict(x=1.65, y=1.65, z=0.85))
axis_cfg = dict(showbackground=True, backgroundcolor="rgba(240, 248, 255, 0.5)")

for sc_name in ("scene", "scene2", "scene3"):
    fig_3d.update_layout(**{
        sc_name: dict(
            xaxis=dict(**axis_cfg, title="X (m)"),
            yaxis=dict(**axis_cfg, title="Y (m)"),
            zaxis=dict(**axis_cfg, title="Z (m)"),
            aspectmode="data",
            camera=cam_view
        )
    })

fig_3d.update_layout(
    title=dict(text="Comparação Geométrica 3D — Soluções da Fronteira de Pareto (BM-S-52)", x=0.5, font=dict(size=16)),
    legend=dict(x=0.01, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(l=0, r=0, t=90, b=0),
    height=620
)
fig_3d.show()
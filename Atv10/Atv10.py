import numpy as np
import gurobipy as gp
from gurobipy import GRB
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constantes globais
BOE_dia   = 120000  # Produção diária da plataforma em barris de óleo equivalente por dia
gamma_e   = 0.270   # t/m3 - Peso específico estrutural equivalente para estimativa de peso do casco
rho_agua  = 1.025   # t/m3 - Massa específica da água do mar, usada no cálculo do deslocamento (\Delta = \rho * \nabla)
P_PL      = 83 * (BOE_dia / 1000) # Peso seco da planta de processo, estimado em 83 kg/BOE para semissubmersíveis
# Peso total do topside (P_PL + P_CO + P_OP), considerando estrutura do convés (35% de P_PL) e peso operacional (35% de P_PL)
P_topside = P_PL * (1 + 0.35 + 0.35)
area_req  = 42.246 * (BOE_dia / 1000) + 4850.1 # Área mínima da planta de processo no convés baseada na produção

def avaliar_plataforma(Lp, bp, hp, bc, hc, T):
    # Lp: Comprimento dos pontoons e dimensão global da plataforma quadrada (Lp x Lp)
    # bp: Largura (boca) dos pontoons
    # hp: Altura dos pontoons
    # bc: Largura (lado) das colunas quadradas
    # hc: Altura das colunas
    # T: Calado de projeto

    vol_pontoons = 2 * Lp * bp * hp # Volume total dos 2 pontoons
    vol_colunas  = 4 * bc**2 * hc   # Volume total das 4 colunas
    P_C = (vol_pontoons + vol_colunas) * gamma_e # Peso da estrutura do casco (P_C)

    # VCG do casco estimado pelo centroide ponderado dos volumes estruturais
    VCG_C = (vol_pontoons * (hp / 2) + vol_colunas * (hp + hc / 2)) / (vol_pontoons + vol_colunas)

    vol_sub_pontoon = 2 * Lp * bp * min(hp, T) # Volume submerso dos pontoons.
    vol_sub_coluna  = 4 * bc**2 * max(0, T - hp) # Volume submerso das colunas.
    nabla = vol_sub_pontoon + vol_sub_coluna # Volume deslocado total
    Delta = nabla * rho_agua # Deslocamento total da plataforma em toneladas

    P_R = 0.04 * Delta # Peso/tração vertical dos risers estimado como 4% do deslocamento (peso da plataforma)
    P_A = 0.03 * Delta # Componente vertical da amarração estimada como 3% do deslocamento
    P_L = Delta - (P_topside + P_C + P_R + P_A) # Peso do lastro calculado pelo balanço de pesos e empuxo

    VCG_topside = hp + hc + 10 # VCG do topside estimado a 10 metros acima do topo das colunas
    # Somatória dos momentos estáticos verticais (Peso x VCG de cada parcela)
    momento_total = (P_topside * VCG_topside + P_C * VCG_C
                     + (P_R + P_A) * (hp + hc) + P_L * (hp / 2))
    KG = momento_total / Delta # Centro de gravidade vertical (KG) da plataforma

    # Altura do centro de carena (KB) baseada nos centroides dos volumes submersos
    KB = (vol_sub_pontoon * (hp / 2) + vol_sub_coluna * (hp + (T - hp) / 2)) / nabla

    # Inércia transversal considera o afastamento das colunas baseada na largura total Lp
    I_x = 4 * (bc**4 / 12 + bc**2 * ((Lp - bc) / 2)**2)
    GM = KB + I_x / nabla - KG # Altura metacêntrica (GM) transversal/longitudinal

    airgap   = hp + hc - T # Airgap da plataforma (distância do convés à linha d'água)
    area_disp = Lp**2 # Área de convés disponível (Lp x Lp)

    return {
        'P_topside': P_topside, 'VCG_topside': VCG_topside,
        'P_C': P_C, 'VCG_C': VCG_C,
        'P_L': P_L, 'VCG_L': hp / 2,
        'P_P': Delta, 'KG': KG,
        'V_deslocado': nabla, 'Delta': Delta,
        'GM': GM, 'airgap': airgap,
        'Area_disp': area_disp, 'Area_req': area_req,
        # Restrições do modelo: GM entre 1.0m e 4.0m, airgap >= 12m, área compatível e lastro positivo
        'Viavel': (1.0 <= GM <= 4.0) and (airgap >= 12.0) and (area_disp >= area_req) and (P_L > 0)
    }

# A. Avaliação do Caso-Base
print("=== AVALIAÇÃO DO CASO BASE ===")
base = avaliar_plataforma(Lp=100, bp=15, hp=15, bc=15, hc=45, T=30)
for k, v in base.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

# --- Modelo de Otimização ---
m = gp.Model("Otimizacao_Semissubmersivel")
m.params.NonConvex = 2

T = 30.0

# 1. Variáveis de Decisão
Lp = m.addVar(lb=80,  ub=150, name="Lp")
bp = m.addVar(lb=10,  ub=30,  name="bp")
hp = m.addVar(lb=5,   ub=25,  name="hp")
bc = m.addVar(lb=10,  ub=30,  name="bc")
hc = m.addVar(lb=20,  ub=60,  name="hc")

# =====================================================================
# RESTRIÇÕES GEOMÉTRICAS
# =====================================================================
m.addConstr(hp == bp, "Pontoon_Secao_Quadrada")
m.addConstr(bc == bp, "Coluna_Igual_Largura_Pontoon")
# A restrição do topside quadrado de lado Lp é garantida por Lp^2 >= area_req.
# =====================================================================

# 2. Produtos auxiliares (decomposição bilinear para linearizar equações complexas de inércia e volume)
Lp2    = m.addVar(name="Lp2")
bc2    = m.addVar(name="bc2")
bc3    = m.addVar(name="bc3")
bc4    = m.addVar(name="bc4")
Lp_bp  = m.addVar(name="Lp_bp")
bc2_hp = m.addVar(name="bc2_hp")
Lp_bc3 = m.addVar(name="Lp_bc3")
bc2_Lp2 = m.addVar(name="bc2_Lp2")

m.addConstr(Lp2    == Lp * Lp)
m.addConstr(bc2    == bc * bc)
m.addConstr(bc3    == bc2 * bc)
m.addConstr(bc4    == bc3 * bc)
m.addConstr(Lp_bp  == Lp * bp)
m.addConstr(bc2_hp == bc2 * hp)
m.addConstr(Lp_bc3 == Lp * bc3)
m.addConstr(bc2_Lp2 == bc2 * Lp2)

# 3. Volumes (calculados de forma bilinear para o Gurobi)
Vol_p = m.addVar(name="Vol_p")
Vol_c = m.addVar(name="Vol_c")
m.addConstr(Vol_p == 2 * Lp_bp * hp)   # Volume dos pontoons
m.addConstr(Vol_c == 4 * bc2 * hc)     # Volume das colunas

Vol_sub_c = m.addVar(name="Vol_sub_c")
m.addConstr(Vol_sub_c == 4 * T * bc2 - 4 * bc2_hp) # Volume submerso das colunas algebraicamente expandido.

nabla = m.addVar(name="nabla")
Delta = m.addVar(name="Delta")
m.addConstr(nabla == Vol_p + Vol_sub_c)
m.addConstr(Delta == rho_agua * nabla)

# 4. Peso do casco (objetivo da otimização) e lastro
P_C = m.addVar(name="P_C")
m.addConstr(P_C == gamma_e * (Vol_p + Vol_c))
m.setObjective(P_C, GRB.MINIMIZE)

P_L = m.addVar(lb=0, name="P_L")
m.addConstr(P_L == 0.93 * Delta - (P_topside + P_C))  # 1 - 0.04 (risers) - 0.03 (amarração) = 0.93

# 5. Momentos estáticos — produtos auxiliares adicionais para o cálculo linearizado do KG
Vol_p_hp  = m.addVar(name="Vol_p_hp")
Vol_c_hp  = m.addVar(name="Vol_c_hp")
Vol_c_hc  = m.addVar(name="Vol_c_hc")
P_L_hp    = m.addVar(name="P_L_hp")
Delta_hphc = m.addVar(name="Delta_hphc")  # Delta * (hp + hc) para o momento de risers e amarração

m.addConstr(Vol_p_hp   == Vol_p * hp)
m.addConstr(Vol_c_hp   == Vol_c * hp)
m.addConstr(Vol_c_hc   == Vol_c * hc)
m.addConstr(P_L_hp     == P_L * hp)
m.addConstr(Delta_hphc == Delta * (hp + hc))

Momento_Total = m.addVar(name="Momento_Total")
m.addConstr(Momento_Total == P_topside * (hp + hc + 10) # Momento do topside
            + gamma_e * (0.5 * Vol_p_hp + Vol_c_hp + 0.5 * Vol_c_hc) # Momento do casco
            + 0.07 * Delta_hphc # Momento de risers e amarração
            + 0.5 * P_L_hp) # Momento do lastro

# 6. Estabilidade (Formulação algébrica contornando divisões de variáveis)
Vol_sub_c_hp = m.addVar(name="Vol_sub_c_hp")
m.addConstr(Vol_sub_c_hp == Vol_sub_c * hp)

KB_nabla = m.addVar(name="KB_nabla") # Centro de carena multiplicado pelo volume
m.addConstr(KB_nabla == 0.5 * Vol_p_hp + 0.5 * Vol_sub_c_hp + 0.5 * T * Vol_sub_c)

Ix = m.addVar(name="Ix") # Inércia do plano de flutuação expandida para evitar (Lp-bc)^2.
m.addConstr(Ix == (4.0/3.0) * bc4 + bc2_Lp2 - 2 * Lp_bc3)

# Restrições de GM: multiplicadas pelo deslocamento/volume para manter bilinearidade
m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total >= 1.0 * Delta, "GM_Minimo") # GM >= 1.0m
m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total <= 4.0 * Delta, "GM_Maximo") # GM <= 4.0m

# 7. Restrições operacionais
m.addConstr(hp <= T - 1,          "Pontoons_Submersos")
m.addConstr(hp + hc - T >= 12.0,  "Airgap_Minimo") # Airgap deve ser superior a 12m
m.addConstr(Lp2 >= area_req,      "Area_Conves_Minima") # Área disponível na plataforma quadrada deve suprir a área da planta

m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"\nSolução Ótima Encontrada: Peso do Casco = {P_C.X:.2f} t")
    print(f"Dimensões [m]: Lp={Lp.X:.2f}, bp={bp.X:.2f}, hp={hp.X:.2f}, bc={bc.X:.2f}, hc={hc.X:.2f}")
    KG_val = Momento_Total.X / Delta.X
    KB_val = KB_nabla.X / nabla.X
    BM_val = Ix.X / nabla.X
    print(f"GM = {KB_val + BM_val - KG_val:.2f} m | Airgap = {hp.X + hc.X - T:.2f} m")
else:
    print("Nenhuma solução viável no espaço de busca com as restrições impostas.")


# === VISUALIZAÇÃO 3D COMPARATIVA ===

def _box_traces(x0, x1, y0, y1, z0, z1, color, name, opacity=1.0, scene="scene", showlegend=True):
    """
    Desenha um paralelepípedo com 6 faces independentes (Mesh3d por face),
    eliminando artefatos de triangulação visível.
    """
    faces = [
        ([x0,x1,x1,x0], [y0,y0,y1,y1], [z0,z0,z0,z0]),
        ([x0,x1,x1,x0], [y0,y0,y1,y1], [z1,z1,z1,z1]),
        ([x0,x0,x0,x0], [y0,y1,y1,y0], [z0,z0,z1,z1]),
        ([x1,x1,x1,x1], [y0,y1,y1,y0], [z0,z0,z1,z1]),
        ([x0,x1,x1,x0], [y0,y0,y0,y0], [z0,z0,z1,z1]),
        ([x0,x1,x1,x0], [y1,y1,y1,y1], [z0,z0,z1,z1]),
    ]
    traces = []
    for idx, (fx, fy, fz) in enumerate(faces):
        traces.append(go.Mesh3d(
            x=fx, y=fy, z=fz,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=opacity,
            name=name, scene=scene,
            showlegend=(showlegend and idx == 1),
            legendgroup=name, flatshading=True,
            lighting=dict(diffuse=0.9, specular=0.1, ambient=0.4)
        ))
    return traces

def _water_plane(half, T, scene, show_legend):
    """Plano d'água como polígono preenchido + borda destacada representativa do calado (T)."""
    w = half + 20
    xs = [-w,  w,  w, -w, -w]
    ys = [-w, -w,  w,  w, -w]
    traces = [
        go.Mesh3d(
            x=[-w, w,  w, -w], y=[-w, -w, w,  w], z=[T, T, T, T],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#0077CC", opacity=0.25,
            name="Linha d'água", scene=scene,
            showlegend=show_legend, legendgroup="Linha d'água",
            flatshading=True, lighting=dict(ambient=1.0)
        ),
        go.Scatter3d(
            x=xs, y=ys, z=[T]*5,
            mode="lines",
            line=dict(color="#0055AA", width=3),
            name="Linha d'água", scene=scene,
            showlegend=False, legendgroup="Linha d'água"
        )
    ]
    return traces

def _semisub_traces(Lp, bp, hp, bc, hc, T, scene, show_legend):
    """Retorna lista de traces Plotly reconstruindo a geometria (2 pontoons, 4 colunas)."""
    traces = []
    half_Lp = Lp / 2
    deck_h = 3

    # Pontoons (distanciados transversalmente para que a largura total externa seja Lp)
    for sign in [-1, 1]:
        yc = sign * (Lp - bp) / 2
        sl = show_legend and (sign == -1)
        traces += _box_traces(-half_Lp, half_Lp, yc - bp/2, yc + bp/2, 0, hp,
                               "#8B6914", "Pontoon", scene=scene, showlegend=sl)

    # Colunas localizadas nas quatro extremidades, alinhadas às bordas dos pontoons
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            cx = sign_x * (Lp - bc) / 2
            cy = sign_y * (Lp - bc) / 2
            sl = show_legend and (sign_x == -1 and sign_y == -1)
            traces += _box_traces(cx - bc/2, cx + bc/2, cy - bc/2, cy + bc/2, hp, hp + hc,
                                   "#4A90D9", "Coluna", scene=scene, showlegend=sl)

    # Deck (estrutura do convés exatamente quadrada: Lp x Lp)
    traces += _box_traces(-half_Lp, half_Lp, -half_Lp, half_Lp, hp + hc, hp + hc + deck_h,
                           "#A0A0A0", "Deck", opacity=0.9, scene=scene, showlegend=show_legend)

    traces += _water_plane(half_Lp, T, scene, show_legend)
    return traces

def comparar_plataformas(base_dims, otim_dims, T, base_viavel, otim_res):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"Caso Base — INVIÁVEL  |  GM={base_viavel['GM']:.2f}m, Airgap={base_viavel['airgap']:.1f}m",
            f"Caso Ótimo — VIÁVEL  |  GM={otim_res['GM']:.2f}m, Airgap={otim_res['airgap']:.1f}m, P_C={otim_res['P_C']:.0f}t"
        ]
    )

    for trace in _semisub_traces(*base_dims, T, scene="scene",  show_legend=True):
        fig.add_trace(trace, row=1, col=1)
    for trace in _semisub_traces(*otim_dims,  T, scene="scene2", show_legend=False):
        fig.add_trace(trace, row=1, col=2)

    cam = dict(eye=dict(x=1.6, y=1.6, z=0.8))
    axis = dict(showbackground=True, backgroundcolor="rgba(240,248,255,0.5)")
    for scene_id in ("scene", "scene2"):
        fig.update_layout(**{
            scene_id: dict(xaxis=dict(**axis, title="X (m)"),
                           yaxis=dict(**axis, title="Y (m)"),
                           zaxis=dict(**axis, title="Z (m)"),
                           aspectmode="data", camera=cam)
        })

    fig.update_layout(
        title=dict(text="Comparação: Caso Base vs. Solução Ótima", x=0.5, font=dict(size=16)),
        legend=dict(x=0.01, y=0.95),
        margin=dict(l=0, r=0, t=80, b=0),
        height=650
    )
    fig.show()

# Dimensões baseadas no enunciado: (Lp, bp, hp, bc, hc)
base_dims = (100, 15, 15, 15, 45)

if m.status == GRB.OPTIMAL:
    otim_dims = (Lp.X, bp.X, hp.X, bc.X, hc.X)
    otim_res  = avaliar_plataforma(*otim_dims, T=30)
else:
    otim_dims = base_dims
    otim_res  = base

comparar_plataformas(base_dims, otim_dims, T=30, # Calado fixo
                     base_viavel=base, otim_res=otim_res)

from scipy.integrate import simpson

def analisar_heave_semisub(lp, bp, hp, bc, lc, hc, T, xi=0.05, nome_caso="Caso-Base"):
    print(f"\n{'='*78}")
    print(f"EXERCÍCIO 10: COMPORTAMENTO EM ONDAS (HEAVE) - {nome_caso.upper()}")
    print(f"{'='*78}")

    # Constantes Físicas e Ambientais (Mar centenário - PNV3425)
    rho   = rho_agua     # 1.025 t/m^3
    g     = 9.81         # m/s^2
    Tp    = 14.6         # Período de pico (s)
    Hs    = 12.2         # Altura significativa (m)
    gamma = 3.3          # Fator de pico JONSWAP

    # Relações geométricas derivadas (Planta quadrada)
    ld = lp - 2 * lc     # Comprimento dos pontoons descoberto de colunas (m)
    dp = lp - bp         # Distância entre os centros dos pontoons (m)

    # Volume deslocado e Massa da plataforma
    vol_pontoons = 2 * lp * bp * hp
    vol_colunas  = 4 * bc * lc * (T - hp)
    nabla = vol_pontoons + vol_colunas
    M = rho * nabla      # Massa total (t)

    # 5) Massa Adicional da Plataforma (DNV - trechos descobertos dos pontoons)
    Ma2D = 1.51 * rho * np.pi * ((bp / 2.0)**2)  # t/m
    Ma   = 2 * Ma2D * ld                         # t

    # 6) Frequência Natural e Período Natural de Heave
    c_rest = 4 * rho * g * bc * lc               # Restauração hidrostática (kN/m)
    wn = np.sqrt(c_rest / (M + Ma))              # rad/s
    Tn = 2 * np.pi / wn                          # s

    # Vetor de frequências (rad/s)
    omega = np.linspace(0.01, 1.50, 1000)

    # 1) Força de Heave de Froude-Krylov - Parte Descoberta (F_FKD / A) [kN/m]
    termo_comum_FK = rho * (g**2) * np.exp(-(omega**2) * T / g) * np.cos((omega**2) * dp / (2 * g))
    F_FKD_A = (4 * ld * termo_comum_FK
               * (np.sin((omega**2) * bp / (2 * g)) / (omega**2))
               * (1 - np.exp((omega**2) * hp / g)))

    # 2) Força de Heave de Froude-Krylov - Parte Coberta (F_FKC / A) [kN/m]
    F_FKC_A = (8 * lc * termo_comum_FK
               * (np.sin((omega**2) * bc / (2 * g)) / (omega**2)))

    # Força Total de Froude-Krylov (F_FK / A)
    F_FK_A = F_FKD_A + F_FKC_A

    # 3) Força de Heave da Massa Adicional (F_MA / A) [kN/m]
    F_MA_A = (-2 * Ma2D * ld * (omega**2)
              * np.exp(-(omega**2) * (T - hp / 2.0) / g)
              * np.cos((omega**2) * dp / (2 * g)))

    # 4) Força Total de Heave (F_h / A) [kN/m]
    F_total_A = F_FK_A + F_MA_A

    # 7) RAO de Heave da Plataforma (|Z(w)| / A)
    razao_freq = omega / wn
    fator_dinamico = np.sqrt((1 - razao_freq**2)**2 + (2 * xi * razao_freq)**2)
    RAO_heave = np.abs(F_total_A / (4 * rho * g * bc * lc)) / fator_dinamico

    # 8) Espectro do Mar de JONSWAP
    wp = 2 * np.pi / Tp
    alpha_star = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 * ((1.9 + gamma)**-1))
    tau = np.where(omega < wp, 0.07, 0.09)

    expoente_gamma = np.exp(-((omega - wp)**2) / (2 * (tau**2) * (wp**2)))
    S_zeta = (alpha_star * (Hs**2) * (omega**-5) / (wp**-4)
              * np.exp(-1.25 * ((omega / wp)**-4)) * (gamma**expoente_gamma))

    # 9) Espectro de Resposta, Amplitude Significativa e Altura Significativa de Heave
    S_heave = (RAO_heave**2) * S_zeta

    m0_mar   = simpson(y=S_zeta, x=omega)
    Hs_check = 4.0 * np.sqrt(m0_mar)

    m0_heave = simpson(y=S_heave, x=omega)
    Amp_sig  = 2.0 * np.sqrt(m0_heave) # Amplitude significativa (m)
    Alt_sig  = 4.0 * np.sqrt(m0_heave) # Altura significativa (m)

    # Relatório no Console
    print(f"Geometria: lp={lp:.2f}m | bp={bp:.2f}m | hp={hp:.2f}m | bc={bc:.2f}m | ld={ld:.2f}m | dp={dp:.2f}m | T={T:.1f}m")
    print(f"5) Massa Adicional (Ma):          {Ma:,.2f} t  (Massa Plataforma M = {M:,.2f} t)")
    print(f"6) Frequência Natural (wn):       {wn:.4f} rad/s")
    print(f"   Período Natural (Tn):          {Tn:.2f} s  (Restrição de projeto: Tn > 20 s -> {'ATENDE' if Tn > 20 else 'FALHA'})")
    print(f"7) RAO Máximo (Ressonância):      {np.max(RAO_heave):.2f} m/m (em w = {omega[np.argmax(RAO_heave)]:.3f} rad/s)")
    print(f"   RAO no Pico da Onda (wp):      {np.interp(wp, omega, RAO_heave):.3f} m/m (em wp = {wp:.3f} rad/s)")
    print(f"8) Hs Mar (Input / Integrado):    {Hs:.2f} m / {Hs_check:.2f} m")
    print(f"9) Amplitude Significativa Heave: {Amp_sig:.2f} m")
    print(f"   Altura Significativa Heave:    {Alt_sig:.2f} m")

    # Plotagem dos Resultados
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1-4) Forças de Excitação / A (kN/m)",
            "7) RAO de Heave vs. Frequência ω (rad/s)",
            "8) Espectro de Mar JONSWAP S_ζ(ω)",
            "9) Espectro de Resposta de Heave S_heave(ω)"
        )
    )

    # Gráfico 1: Parcelas de Força
    fig.add_trace(go.Scatter(x=omega, y=F_FKD_A, name="F_FK Descoberta", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=omega, y=F_FKC_A, name="F_FK Coberta", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=omega, y=F_MA_A, name="F_MA (Massa Adic.)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=omega, y=F_total_A, name="F_Total", line=dict(color="black", width=2.5)), row=1, col=1)

    # Gráfico 2: RAO
    fig.add_trace(go.Scatter(x=omega, y=RAO_heave, name="RAO Heave", line=dict(color="royalblue", width=2.5)), row=1, col=2)
    fig.add_vline(x=wn, line_dash="dash", line_color="red", annotation_text=f"ωn={wn:.2f}", row=1, col=2)
    fig.add_vline(x=wp, line_dash="dot", line_color="green", annotation_text=f"ωp={wp:.2f}", row=1, col=2)

    # Gráfico 3: Espectro de Mar
    fig.add_trace(go.Scatter(x=omega, y=S_zeta, name="S_ζ(ω)", fill="tozeroy", line=dict(color="teal")), row=2, col=1)

    # Gráfico 4: Espectro de Heave
    fig.add_trace(go.Scatter(x=omega, y=S_heave, name="S_heave(ω)", fill="tozeroy", line=dict(color="darkorange")), row=2, col=2)

    fig.update_xaxes(title_text="ω (rad/s)", range=[0, 1.5], row=1, col=1)
    fig.update_yaxes(title_text="F / A (kN/m)", row=1, col=1)

    fig.update_xaxes(title_text="ω (rad/s)", range=[0, 1.5], row=1, col=2)
    fig.update_yaxes(title_text="RAO (m/m)", row=1, col=2)

    fig.update_xaxes(title_text="ω (rad/s)", range=[0, 1.5], row=2, col=1)
    fig.update_yaxes(title_text="S_ζ (m²·s)", row=2, col=1)

    fig.update_xaxes(title_text="ω (rad/s)", range=[0, 1.5], row=2, col=2)
    fig.update_yaxes(title_text="S_heave (m²·s)", row=2, col=2)

    fig.update_layout(height=750, title_text=f"Comportamento em Ondas (Heave) - {nome_caso}", showlegend=True)
    fig.show()


# 1. Caso-Base
analisar_heave_semisub(
    lp=100, bp=15, hp=15, bc=15, lc=15, hc=45, T=30, xi=0.05,
    nome_caso="Caso-Base (Ex. 10)"
)

# 2. Solução Ótima do Gurobi
if m.status == GRB.OPTIMAL:
    analisar_heave_semisub(
        lp=Lp.X, bp=bp.X, hp=hp.X, bc=bc.X, lc=bc.X, hc=hc.X, T=T, xi=0.05,
        nome_caso="Solução Ótima"
    )
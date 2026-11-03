import numpy as np
import gurobipy as gp
from gurobipy import GRB
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constantes globais
BOE_dia   = 120000
gamma_e   = 0.270   # t/m3
rho_agua  = 1.025   # t/m3
P_PL      = 83 * (BOE_dia / 1000)
P_topside = P_PL * (1 + 0.35 + 0.35)
area_req  = 42.246 * (BOE_dia / 1000) + 4850.1

def avaliar_plataforma(Lp, bp, hp, bc, hc, T):
    vol_pontoons = 2 * Lp * bp * hp
    vol_colunas  = 4 * bc**2 * hc
    P_C = (vol_pontoons + vol_colunas) * gamma_e

    VCG_C = (vol_pontoons * (hp / 2) + vol_colunas * (hp + hc / 2)) / (vol_pontoons + vol_colunas)

    vol_sub_pontoon = 2 * Lp * bp * min(hp, T)
    vol_sub_coluna  = 4 * bc**2 * max(0, T - hp)
    nabla = vol_sub_pontoon + vol_sub_coluna
    Delta = nabla * rho_agua

    P_R = 0.04 * Delta
    P_A = 0.03 * Delta
    P_L = Delta - (P_topside + P_C + P_R + P_A)

    VCG_topside = hp + hc + 10
    momento_total = (P_topside * VCG_topside + P_C * VCG_C
                     + (P_R + P_A) * (hp + hc) + P_L * (hp / 2))
    KG = momento_total / Delta

    KB = (vol_sub_pontoon * (hp / 2) + vol_sub_coluna * (hp + (T - hp) / 2)) / nabla
    I_x = 4 * (bc**4 / 12 + bc**2 * ((Lp - bc) / 2)**2)
    GM = KB + I_x / nabla - KG

    airgap   = hp + hc - T
    area_disp = Lp**2

    return {
        'P_topside': P_topside, 'VCG_topside': VCG_topside,
        'P_C': P_C, 'VCG_C': VCG_C,
        'P_L': P_L, 'VCG_L': hp / 2,
        'P_P': Delta, 'KG': KG,
        'V_deslocado': nabla, 'Delta': Delta,
        'GM': GM, 'airgap': airgap,
        'Area_disp': area_disp, 'Area_req': area_req,
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

# 2. Produtos auxiliares (decomposição bilinear)
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

# 3. Volumes
Vol_p = m.addVar(name="Vol_p")
Vol_c = m.addVar(name="Vol_c")
m.addConstr(Vol_p == 2 * Lp_bp * hp)   # bilinear: Lp_bp * hp
m.addConstr(Vol_c == 4 * bc2 * hc)     # bilinear: bc2 * hc

Vol_sub_c = m.addVar(name="Vol_sub_c")
m.addConstr(Vol_sub_c == 4 * T * bc2 - 4 * bc2_hp)

nabla = m.addVar(name="nabla")
Delta = m.addVar(name="Delta")
m.addConstr(nabla == Vol_p + Vol_sub_c)
m.addConstr(Delta == rho_agua * nabla)

# 4. Peso do casco (objetivo) e lastro
P_C = m.addVar(name="P_C")
m.addConstr(P_C == gamma_e * (Vol_p + Vol_c))
m.setObjective(P_C, GRB.MINIMIZE)

P_L = m.addVar(lb=0, name="P_L")
m.addConstr(P_L == 0.93 * Delta - (P_topside + P_C))  # 1 - 0.04 - 0.03 = 0.93

# 5. Momentos estáticos — produtos auxiliares adicionais
Vol_p_hp  = m.addVar(name="Vol_p_hp")
Vol_c_hp  = m.addVar(name="Vol_c_hp")
Vol_c_hc  = m.addVar(name="Vol_c_hc")
P_L_hp    = m.addVar(name="P_L_hp")
Delta_hphc = m.addVar(name="Delta_hphc")  # Delta * (hp + hc)

m.addConstr(Vol_p_hp   == Vol_p * hp)
m.addConstr(Vol_c_hp   == Vol_c * hp)
m.addConstr(Vol_c_hc   == Vol_c * hc)
m.addConstr(P_L_hp     == P_L * hp)
m.addConstr(Delta_hphc == Delta * (hp + hc))  # bilinear: Delta*(hp+hc)

# Momento_Total = P_topside*(hp+hc+10) + gamma_e*(0.5*Vol_p_hp + Vol_c_hp + 0.5*Vol_c_hc)
#               + 0.07*Delta*(hp+hc) + 0.5*P_L*hp
Momento_Total = m.addVar(name="Momento_Total")
m.addConstr(Momento_Total == P_topside * (hp + hc + 10)
            + gamma_e * (0.5 * Vol_p_hp + Vol_c_hp + 0.5 * Vol_c_hc)
            + 0.07 * Delta_hphc
            + 0.5 * P_L_hp)

# 6. Estabilidade
Vol_sub_c_hp = m.addVar(name="Vol_sub_c_hp")
m.addConstr(Vol_sub_c_hp == Vol_sub_c * hp)

KB_nabla = m.addVar(name="KB_nabla")
m.addConstr(KB_nabla == 0.5 * Vol_p_hp + 0.5 * Vol_sub_c_hp + 0.5 * T * Vol_sub_c)

Ix = m.addVar(name="Ix")
m.addConstr(Ix == (4.0/3.0) * bc4 + bc2_Lp2 - 2 * Lp_bc3)

# GM constraints (multiplied by Delta to avoid division)
m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total >= 1.0 * Delta, "GM_Minimo")
m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total <= 4.0 * Delta, "GM_Maximo")

# 7. Restrições operacionais
m.addConstr(hp <= T - 1,          "Pontoons_Submersos")
m.addConstr(hp + hc - T >= 12.0,  "Airgap_Minimo")
m.addConstr(Lp2 >= area_req,      "Area_Conves_Minima")

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
    # Cada face: 4 vértices → 2 triângulos com normais consistentes
    faces = [
        # (xs, ys, zs)  — sentido anti-horário visto de fora
        ([x0,x1,x1,x0], [y0,y0,y1,y1], [z0,z0,z0,z0]),  # fundo
        ([x0,x1,x1,x0], [y0,y0,y1,y1], [z1,z1,z1,z1]),  # topo
        ([x0,x0,x0,x0], [y0,y1,y1,y0], [z0,z0,z1,z1]),  # esquerda
        ([x1,x1,x1,x1], [y0,y1,y1,y0], [z0,z0,z1,z1]),  # direita
        ([x0,x1,x1,x0], [y0,y0,y0,y0], [z0,z0,z1,z1]),  # frente
        ([x0,x1,x1,x0], [y1,y1,y1,y1], [z0,z0,z1,z1]),  # trás
    ]
    traces = []
    for idx, (fx, fy, fz) in enumerate(faces):
        traces.append(go.Mesh3d(
            x=fx, y=fy, z=fz,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=opacity,
            name=name, scene=scene,
            showlegend=(showlegend and idx == 1),  # legenda só no topo
            legendgroup=name, flatshading=True,
            lighting=dict(diffuse=0.9, specular=0.1, ambient=0.4)
        ))
    return traces

def _water_plane(half, T, scene, show_legend):
    """Plano d'água como polígono preenchido + borda destacada."""
    w = half + 20
    xs = [-w,  w,  w, -w, -w]
    ys = [-w, -w,  w,  w, -w]
    traces = [
        # Superfície preenchida
        go.Mesh3d(
            x=[-w, w,  w, -w], y=[-w, -w, w,  w], z=[T, T, T, T],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#0077CC", opacity=0.25,
            name="Linha d'água", scene=scene,
            showlegend=show_legend, legendgroup="Linha d'água",
            flatshading=True, lighting=dict(ambient=1.0)
        ),
        # Borda do plano
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
    """Retorna lista de traces Plotly para uma configuração de semisub."""
    traces = []
    half = Lp / 2
    deck_overhang = 5
    deck_h = 3

    # Pontoons
    for sign, label in [(-1, "Pontoon"), (1, "Pontoon")]:
        yc = sign * (bp / 2 + bc / 2)
        sl = show_legend and (sign == -1)
        traces += _box_traces(-half, half, yc - bp/2, yc + bp/2, 0, hp,
                               "#8B6914", "Pontoon", scene=scene, showlegend=sl)

    # Colunas
    col_cx = [-half + bc/2,  half - bc/2,  half - bc/2, -half + bc/2]
    col_cy = [-bp/2 - bc/2, -bp/2 - bc/2,  bp/2 + bc/2,  bp/2 + bc/2]
    for idx, (cx, cy) in enumerate(zip(col_cx, col_cy)):
        sl = show_legend and (idx == 0)
        traces += _box_traces(cx - bc/2, cx + bc/2, cy - bc/2, cy + bc/2, hp, hp + hc,
                               "#4A90D9", "Coluna", scene=scene, showlegend=sl)

    # Deck
    dh = half + deck_overhang
    dy = bp / 2 + bc / 2 + deck_overhang
    traces += _box_traces(-dh, dh, -dy, dy, hp + hc, hp + hc + deck_h,
                           "#A0A0A0", "Deck", opacity=0.9, scene=scene, showlegend=show_legend)

    # Plano d'água
    traces += _water_plane(half, T, scene, show_legend)
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


# Dimensões: (Lp, bp, hp, bc, hc)
base_dims = (100, 15, 15, 15, 45)

if m.status == GRB.OPTIMAL:
    otim_dims = (Lp.X, bp.X, hp.X, bc.X, hc.X)
    otim_res  = avaliar_plataforma(*otim_dims, T=30)
else:
    otim_dims = base_dims  # fallback
    otim_res  = base

comparar_plataformas(base_dims, otim_dims, T=30,
                     base_viavel=base, otim_res=otim_res)

import numpy as np
import scipy.optimize as opt
from scipy.integrate import simpson
import gurobipy as gp
from gurobipy import GRB
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================================
# CONSTANTES GLOBAIS
# =========================================================================
BOE_dia   = 120000  # Produção diária (barris de óleo equivalente/dia)
gamma_e   = 0.270   # Peso específico estrutural (t/m3)
rho_agua  = 1.025   # Massa específica da água do mar (t/m3)
rho_ar    = 1.222 / 1000 # Massa específica do ar (t/m3)
g         = 9.81    # Gravidade (m/s2)

P_PL      = 83 * (BOE_dia / 1000) # Peso da planta de processo
P_topside = P_PL * (1 + 0.35 + 0.35) # Topside total
area_req  = 42.246 * (BOE_dia / 1000) + 4850.1 # Área mínima

# =========================================================================
# ATIVIDADE 8 - AVALIAÇÃO E OTIMIZAÇÃO (PESOS E CENTROS)
# =========================================================================

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

# =========================================================================
# FUNÇÕES DE VISUALIZAÇÃO 3D (PLOTLY)
# =========================================================================

def _box_traces(x0, x1, y0, y1, z0, z1, color, name, opacity=1.0, scene="scene", showlegend=True):
    faces = [
        ([x0,x1,x1,x0], [y0,y0,y1,y1], [z0,z0,z0,z0]), ([x0,x1,x1,x0], [y0,y0,y1,y1], [z1,z1,z1,z1]),
        ([x0,x0,x0,x0], [y0,y1,y1,y0], [z0,z0,z1,z1]), ([x1,x1,x1,x1], [y0,y1,y1,y0], [z0,z0,z1,z1]),
        ([x0,x1,x1,x0], [y0,y0,y0,y0], [z0,z0,z1,z1]), ([x0,x1,x1,x0], [y1,y1,y1,y1], [z0,z0,z1,z1]),
    ]
    traces = []
    for idx, (fx, fy, fz) in enumerate(faces):
        traces.append(go.Mesh3d(
            x=fx, y=fy, z=fz, i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=opacity, name=name, scene=scene,
            showlegend=(showlegend and idx == 1), legendgroup=name, flatshading=True,
            lighting=dict(diffuse=0.9, specular=0.1, ambient=0.4)
        ))
    return traces

def _water_plane(half, T, scene, show_legend):
    w = half + 20
    xs, ys = [-w, w, w, -w, -w], [-w, -w, w, w, -w]
    return [
        go.Mesh3d(
            x=[-w, w, w, -w], y=[-w, -w, w, w], z=[T, T, T, T],
            i=[0, 0], j=[1, 2], k=[2, 3], color="#0077CC", opacity=0.25,
            name="Linha d'água", scene=scene, showlegend=show_legend, legendgroup="Linha d'água",
            flatshading=True, lighting=dict(ambient=1.0)
        ),
        go.Scatter3d(
            x=xs, y=ys, z=[T]*5, mode="lines", line=dict(color="#0055AA", width=3),
            name="Linha d'água", scene=scene, showlegend=False, legendgroup="Linha d'água"
        )
    ]

def _semisub_traces(Lp, bp, hp, bc, hc, T, scene, show_legend):
    traces = []
    half_Lp, deck_h = Lp / 2, 3
    # Pontoons
    for sign in [-1, 1]:
        yc = sign * (Lp - bp) / 2
        sl = show_legend and (sign == -1)
        traces += _box_traces(-half_Lp, half_Lp, yc - bp/2, yc + bp/2, 0, hp, "#8B6914", "Pontoon", scene=scene, showlegend=sl)
    # Colunas
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            cx, cy = sign_x * (Lp - bc) / 2, sign_y * (Lp - bc) / 2
            sl = show_legend and (sign_x == -1 and sign_y == -1)
            traces += _box_traces(cx - bc/2, cx + bc/2, cy - bc/2, cy + bc/2, hp, hp + hc, "#4A90D9", "Coluna", scene=scene, showlegend=sl)
    # Deck
    traces += _box_traces(-half_Lp, half_Lp, -half_Lp, half_Lp, hp + hc, hp + hc + deck_h, "#A0A0A0", "Deck", opacity=0.9, scene=scene, showlegend=show_legend)
    traces += _water_plane(half_Lp, T, scene, show_legend)
    return traces

def comparar_plataformas(base_dims, otim_dims, T, base_viavel, otim_res):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"Caso Base — INVIÁVEL  |  GM={base_viavel['GM']:.2f}m, Airgap={base_viavel['airgap']:.1f}m",
            f"Caso Ótimo — VIÁVEL  |  GM={otim_res['GM']:.2f}m, Airgap={otim_res['airgap']:.1f}m, P_C={otim_res['P_C']:.0f}t"
        ]
    )
    for trace in _semisub_traces(*base_dims, T, scene="scene",  show_legend=True):
        fig.add_trace(trace, row=1, col=1)
    for trace in _semisub_traces(*otim_dims,  T, scene="scene2", show_legend=False):
        fig.add_trace(trace, row=1, col=2)

    axis = dict(showbackground=True, backgroundcolor="rgba(240,248,255,0.5)")
    for scene_id in ("scene", "scene2"):
        fig.update_layout(**{
            scene_id: dict(xaxis=dict(**axis, title="X (m)"), yaxis=dict(**axis, title="Y (m)"), zaxis=dict(**axis, title="Z (m)"),
                           aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)))
        })
    fig.update_layout(title=dict(text="Comparação: Caso Base vs. Solução Ótima", x=0.5, font=dict(size=16)),
                      legend=dict(x=0.01, y=0.95), margin=dict(l=0, r=0, t=80, b=0), height=650)
    fig.show()

# =========================================================================
# ATIVIDADE 9 - ESTABILIDADE AO VENTO (DNV-OS-C301)
# =========================================================================

def verificar_atividade_9(Lp, bp, hp, bc, hc, T, KG, nome_caso):
    print(f"\n{'='*80}\n=== ATIVIDADE 9: AVALIAÇÃO DE ESTABILIDADE AO VENTO - {nome_caso.upper()} ===\n{'='*80}")
    
    hd, hm = 8.0, 15.0
    dp, xc = Lp - bp, Lp/2 - bc/2
    a_airgap = hp + hc - T
    
    vol_pontoons = 2 * Lp * bp * hp
    vol_colunas = 4 * (bc**2) * (T - hp)
    nabla = vol_pontoons + vol_colunas 
    Delta_N = (nabla * rho_agua) * g  

    def poligono_submerso(P_verts, z0, phi_rad):
        Q_verts = []
        d = [z0 + p[0] * np.tan(phi_rad) - p[1] for p in P_verts]
        
        for i in range(len(P_verts)):
            p1, d1 = P_verts[i], d[i]
            p2, d2 = P_verts[(i+1) % len(P_verts)], d[(i+1) % len(P_verts)]
            if d1 >= 0: Q_verts.append(p1)
            if d1 * d2 < 0:
                s = d1 / (d1 - d2)
                Q_verts.append((p1[0] + s*(p2[0]-p1[0]), p1[1] + s*(p2[1]-p1[1])))
                
        if len(Q_verts) < 3: return 0.0, 0.0, 0.0
        
        A_poly, uc_sum, zc_sum = 0.0, 0.0, 0.0
        for i in range(len(Q_verts)):
            u_k, z_k = Q_verts[i]
            u_k1, z_k1 = Q_verts[(i+1) % len(Q_verts)]
            c_k = u_k * z_k1 - u_k1 * z_k
            A_poly += c_k
            uc_sum += (u_k + u_k1) * c_k
            zc_sum += (z_k + z_k1) * c_k
            
        A = A_poly / 2.0
        return A, uc_sum / (6.0 * A), zc_sum / (6.0 * A)

    def calcular_GZ(phi_deg, plano):
        phi_rad = np.radians(phi_deg)
        retangulos = []
        if plano == "Transversal":
            retangulos.append(( [(-dp/2 - bp/2, 0), (-dp/2 + bp/2, 0), (-dp/2 + bp/2, hp), (-dp/2 - bp/2, hp)], Lp ))
            retangulos.append(( [(dp/2 - bp/2, 0), (dp/2 + bp/2, 0), (dp/2 + bp/2, hp), (dp/2 - bp/2, hp)], Lp ))
            retangulos.append(( [(-dp/2 - bc/2, hp), (-dp/2 + bc/2, hp), (-dp/2 + bc/2, hp+hc), (-dp/2 - bc/2, hp+hc)], 2*bc ))
            retangulos.append(( [(dp/2 - bc/2, hp), (dp/2 + bc/2, hp), (dp/2 + bc/2, hp+hc), (dp/2 - bc/2, hp+hc)], 2*bc ))
        else: # Longitudinal
            retangulos.append(( [(-Lp/2, 0), (Lp/2, 0), (Lp/2, hp), (-Lp/2, hp)], 2*bp ))
            retangulos.append(( [(-xc - bc/2, hp), (-xc + bc/2, hp), (-xc + bc/2, hp+hc), (-xc - bc/2, hp+hc)], 2*bc ))
            retangulos.append(( [(xc - bc/2, hp), (xc + bc/2, hp), (xc + bc/2, hp+hc), (xc - bc/2, hp+hc)], 2*bc ))
            
        def erro_volume(z0_test):
            return sum(poligono_submerso(v, z0_test, phi_rad)[0] * w for v, w in retangulos) - nabla
            
        if phi_deg == 0:
            z0 = T
        else:
            try: z0 = opt.brentq(erro_volume, -Lp, 2*(hp+hc))
            except: return 0, 0
                
        vol_tot, u_mom, z_mom = 0.0, 0.0, 0.0
        for verts, w in retangulos:
            A, uc, zc = poligono_submerso(verts, z0, phi_rad)
            if A > 0:
                vol_tot += A * w
                u_mom += A * w * uc
                z_mom += A * w * zc
                
        if vol_tot == 0: return 0, z0
        return (u_mom/vol_tot) * np.cos(phi_rad) + ((z_mom/vol_tot) - KG) * np.sin(phi_rad), z0

    def coef_Ch(h):
        if h <= 15.3: return 1.00
        if h <= 30.5: return 1.10
        if h <= 46.0: return 1.20
        if h <= 61.0: return 1.30
        return 1.37

    def analisar_plano(plano):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{plano} - V = 51.5 m/s", f"{plano} - V = 36.0 m/s"))
        
        for idx, V_vento in enumerate([51.5, 36.0]):
            q = 0.5 * rho_ar * (V_vento**2) 
            
            parcelas = [
                {"nome": "Colunas", "A": 4 * bc * a_airgap, "h": a_airgap / 2, "Cs": 1.0},
                {"nome": "Deck box", "A": Lp * hd, "h": a_airgap + hd / 2, "Cs": 1.0},
                {"nome": "Modulos", "A": Lp * hm, "h": a_airgap + hd + hm / 2, "Cs": 1.1}
            ]
            
            if plano == "Transversal":
                A_p_sub, A_c_sub = Lp * hp, 2 * bc * (T - hp)
            else:
                A_p_sub, A_c_sub = 2 * bp * hp, 2 * bc * (T - hp)
            z_CLR = (A_p_sub * (hp/2) + A_c_sub * (hp + (T-hp)/2)) / (A_p_sub + A_c_sub)
            
            HM0 = 0.0
            print(f"\n--- {plano} | Vento: {V_vento} m/s ---")
            print(f"z_CLR = {z_CLR:.2f} m")
            print(f"{'Parcela':<15} | {'Força (MN)':<10} | {'Braço (m)':<10} | {'Momento (MN-m)':<15}")
            print("-" * 60)
            
            for p in parcelas:
                Ch = coef_Ch(p["h"])
                F_MN = q * p["Cs"] * Ch * p["A"] / 1000 
                braco = p["h"] + T - z_CLR
                M_MNm = F_MN * braco
                HM0 += M_MNm
                print(f"{p['nome']:<15} | {F_MN:<10.2f} | {braco:<10.2f} | {M_MNm:<15.1f}")
                
            F_ud = q * 1.0 * coef_Ch(a_airgap) * (Lp**2) / 1000 
            b_ud = a_airgap + T - z_CLR
            HM_ud = F_ud * b_ud
            print("-" * 60)
            print(f"TOTAL HM0 = {HM0:.1f} MN-m | Face Inferior HM_ud = {HM_ud:.1f} MN-m\n")
            
            u_col_max = (Lp/2)
            phi_al_deg = np.degrees(np.arctan((hp + hc - T) / u_col_max))
            
            phi_range = np.arange(0, 50.5, 0.5)
            GZ_curve, HA_curve = [], []
            phi1, phi2 = None, None
            
            for phi in phi_range:
                gz, _ = calcular_GZ(phi, plano)
                GZ_curve.append(gz)
                
                phi_r = np.radians(phi)
                ha = (HM0 * np.cos(phi_r) + HM_ud * np.sin(phi_r)) / (Delta_N / 1000)
                HA_curve.append(ha)
                
                if phi > 0 and gz > ha and phi1 is None: phi1 = phi
                if phi > phi_al_deg and gz < ha and phi1 is not None and phi2 is None: phi2 = phi
                    
            idx_al = np.argmin(np.abs(phi_range - phi_al_deg))
            x_rads = np.radians(phi_range[:idx_al+1])
            Area_A = simpson(y=GZ_curve[:idx_al+1], x=x_rads)
            Area_B = simpson(y=HA_curve[:idx_al+1], x=x_rads)
            razao = Area_A / Area_B
            
            atende = (razao >= 1.3) and all(gz > 0 for gz in GZ_curve[1:idx_al+1]) and (phi1 is not None and phi1 < phi_al_deg)
            
            print(f"Resultados do Critério de Áreas:")
            print(f"phi_al = {phi_al_deg:.1f}° | phi_1 = {phi1}° | phi_2 = {phi2}°")
            print(f"Área A = {Area_A:.3f} m-rad | Área B = {Area_B:.3f} m-rad | Razão A/B = {razao:.2f}")
            print(f"Critério (A/B >= 1.3 e GZ > 0): {'ATENDE ✅' if atende else 'FALHA ❌'}\n")
            
            col = idx + 1
            fig.add_trace(go.Scatter(x=phi_range, y=GZ_curve, mode='lines', name='GZ', line=dict(color='teal')), row=1, col=col)
            fig.add_trace(go.Scatter(x=phi_range, y=HA_curve, mode='lines', name='HA', line=dict(color='navy')), row=1, col=col)
            fig.add_vline(x=phi_al_deg, line_dash="dash", line_color="gray", annotation_text="φ_al", row=1, col=col)
            
            fig.update_xaxes(title_text="Ângulo (graus)", range=[0, 50], row=1, col=col)
            fig.update_yaxes(title_text="Braço (m)", row=1, col=col)
            
        fig.update_layout(title_text=f"Curvas GZ e HA - {plano} ({nome_caso})", height=500, showlegend=False)
        fig.show()

    analisar_plano("Transversal")
    analisar_plano("Longitudinal")

# =========================================================================
# EXECUÇÃO PRINCIPAL DO SCRIPT
# =========================================================================
if __name__ == "__main__":
    print("=== AVALIAÇÃO DO CASO BASE (ATIVIDADE 8) ===")
    base_dims = (100, 15, 15, 15, 45)
    T_projeto = 30.0
    base = avaliar_plataforma(*base_dims, T=T_projeto)
    for k, v in base.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== OTIMIZAÇÃO COM GUROBI (ATIVIDADE 8) ===")
    m = gp.Model("Otimizacao_Semissubmersivel")
    m.params.NonConvex = 2
    m.params.LogToConsole = 0 # Ocultar log longo do gurobi

    Lp = m.addVar(lb=80, ub=150, name="Lp")
    bp = m.addVar(lb=10, ub=30,  name="bp")
    hp = m.addVar(lb=5,  ub=25,  name="hp")
    bc = m.addVar(lb=10, ub=30,  name="bc")
    hc = m.addVar(lb=20, ub=60,  name="hc")

    m.addConstr(hp == bp, "Pontoon_Secao_Quadrada") 
    m.addConstr(bc == bp, "Coluna_Igual_Largura_Pontoon")

    Lp2 = m.addVar(name="Lp2")
    bc2 = m.addVar(name="bc2")
    bc3 = m.addVar(name="bc3")
    bc4 = m.addVar(name="bc4")
    Lp_bp = m.addVar(name="Lp_bp")
    bc2_hp = m.addVar(name="bc2_hp")
    Lp_bc3 = m.addVar(name="Lp_bc3")
    bc2_Lp2 = m.addVar(name="bc2_Lp2")

    m.addConstr(Lp2 == Lp * Lp)
    m.addConstr(bc2 == bc * bc)
    m.addConstr(bc3 == bc2 * bc)
    m.addConstr(bc4 == bc3 * bc)
    m.addConstr(Lp_bp == Lp * bp)
    m.addConstr(bc2_hp == bc2 * hp)
    m.addConstr(Lp_bc3 == Lp * bc3)
    m.addConstr(bc2_Lp2 == bc2 * Lp2)

    Vol_p = m.addVar(name="Vol_p")
    Vol_c = m.addVar(name="Vol_c")
    m.addConstr(Vol_p == 2 * Lp_bp * hp)
    m.addConstr(Vol_c == 4 * bc2 * hc)

    Vol_sub_c = m.addVar(name="Vol_sub_c")
    m.addConstr(Vol_sub_c == 4 * T_projeto * bc2 - 4 * bc2_hp)

    nabla = m.addVar(name="nabla")
    Delta = m.addVar(name="Delta")
    m.addConstr(nabla == Vol_p + Vol_sub_c)
    m.addConstr(Delta == rho_agua * nabla)

    P_C = m.addVar(name="P_C")
    m.addConstr(P_C == gamma_e * (Vol_p + Vol_c))
    m.setObjective(P_C, GRB.MINIMIZE)

    P_L = m.addVar(lb=0, name="P_L")
    m.addConstr(P_L == 0.93 * Delta - (P_topside + P_C))

    Vol_p_hp = m.addVar(name="Vol_p_hp")
    Vol_c_hp = m.addVar(name="Vol_c_hp")
    Vol_c_hc = m.addVar(name="Vol_c_hc")
    P_L_hp = m.addVar(name="P_L_hp")
    Delta_hphc = m.addVar(name="Delta_hphc")

    m.addConstr(Vol_p_hp == Vol_p * hp)
    m.addConstr(Vol_c_hp == Vol_c * hp)
    m.addConstr(Vol_c_hc == Vol_c * hc)
    m.addConstr(P_L_hp == P_L * hp)
    m.addConstr(Delta_hphc == Delta * (hp + hc))

    Momento_Total = m.addVar(name="Momento_Total")
    m.addConstr(Momento_Total == P_topside * (hp + hc + 10) + gamma_e * (0.5 * Vol_p_hp + Vol_c_hp + 0.5 * Vol_c_hc) + 0.07 * Delta_hphc + 0.5 * P_L_hp)

    Vol_sub_c_hp = m.addVar(name="Vol_sub_c_hp")
    m.addConstr(Vol_sub_c_hp == Vol_sub_c * hp)
    KB_nabla = m.addVar(name="KB_nabla")
    m.addConstr(KB_nabla == 0.5 * Vol_p_hp + 0.5 * Vol_sub_c_hp + 0.5 * T_projeto * Vol_sub_c)

    Ix = m.addVar(name="Ix")
    m.addConstr(Ix == (4.0/3.0) * bc4 + bc2_Lp2 - 2 * Lp_bc3)

    m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total >= 1.0 * Delta, "GM_Minimo")
    m.addConstr(rho_agua * KB_nabla + rho_agua * Ix - Momento_Total <= 4.0 * Delta, "GM_Maximo")

    m.addConstr(hp <= T_projeto - 1, "Pontoons_Submersos")
    m.addConstr(hp + hc - T_projeto >= 12.0, "Airgap_Minimo")
    m.addConstr(Lp2 >= area_req, "Area_Conves_Minima")

    m.optimize()

    if m.status == GRB.OPTIMAL:
        print(f"Solução Ótima: Peso do Casco = {P_C.X:.2f} t")
        print(f"Dimensões [m]: Lp={Lp.X:.2f}, bp={bp.X:.2f}, hp={hp.X:.2f}, bc={bc.X:.2f}, hc={hc.X:.2f}")
        KG_val = Momento_Total.X / Delta.X
        otim_dims = (Lp.X, bp.X, hp.X, bc.X, hc.X)
        otim_res  = avaliar_plataforma(*otim_dims, T=T_projeto)
    else:
        print("Nenhuma solução ótima viável encontrada.")
        otim_dims, otim_res = base_dims, base
        KG_val = base['KG']

    # Visualização
    comparar_plataformas(base_dims, otim_dims, T_projeto, base, otim_res)

    # Atividade 9
    verificar_atividade_9(*base_dims, T=T_projeto, KG=base['KG'], nome_caso="Caso Base")
    
    if m.status == GRB.OPTIMAL:
        verificar_atividade_9(*otim_dims, T=T_projeto, KG=KG_val, nome_caso="Solução Otimizada")
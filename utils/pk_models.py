'''
Schnider model for PK/PD of propofol
'''
def schnider_pk(t, dose, V1=4.27, V2=18.9, k10=0.119, k12=0.112, k21=0.055, ke0=0.456):
    from scipy.integrate import odeint
    import numpy as np

    def deriv(y, t):
        Cp, Ce = y
        dCp_dt = - (k10 + k12) * Cp + k21 * (dose / V2)  # 2-comp model simplification
        dCe_dt = ke0 * (Cp - Ce)
        return [dCp_dt, dCe_dt]

    y0 = [dose / V1, 0.0]
    result = odeint(deriv, y0, t)
    Cp, Ce = result[:, 0], result[:, 1]
    return Cp, Ce

'''
Minto model for PK/PD of remifentanyl
'''
def minto_pk(t, dose, V1=5.1, V2=9.8, k10=0.3, k12=0.15, k21=0.12, ke0=0.5):
    from scipy.integrate import odeint
    import numpy as np

    def deriv(y, t):
        Cp, Ce = y
        dCp_dt = - (k10 + k12) * Cp + k21 * (dose / V2)
        dCe_dt = ke0 * (Cp - Ce)
        return [dCp_dt, dCe_dt]

    y0 = [dose / V1, 0.0]
    result = odeint(deriv, y0, t)
    Cp, Ce = result[:, 0], result[:, 1]
    return Cp, Ce
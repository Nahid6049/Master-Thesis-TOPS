from tops.dyn_models.blocks import *
from .pll import PLL1, PLL2
import numpy as np


# --------------------------------------------------
# ORIGINAL VSC (KEEP THIS)
# --------------------------------------------------
class VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units),
                                dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units),
                                    dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.pi_p = PIRegulator(K_p=p['P_K_p'], K_i=p['P_K_i'])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p['Q_K_p'], K_i=p['Q_K_i'])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p['T_i'])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_q(x, v)

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))

        self.pi_p.initialize(
            x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0)
        )
        self.pi_q.initialize(
            x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0)
        )

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]


# --------------------------------------------------
# YOUR VSC_V (CORRECTLY ADDED)
# --------------------------------------------------
class VSC_V(DAEModel):
    mdl = 'vsc'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units),
                                dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units),
                                    dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par

        self.pll = PLL2(K_p=p['pll_K_p'], K_i=p['pll_K_i'], bus=p['bus'])

        self.i_p = Integrator(n_units=self.n_units)
        self.i_p.input = lambda x, v: (self.P_setp(x, v) - self.P(x, v)) * p['P_K_i']

        self.i_q = Integrator(n_units=self.n_units)
        self.i_q.input = lambda x, v: ((self.V_setp(x, v) - self.V(x, v)) * p['K_v']) * p['Q_K_i']

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_q = TimeConstant(T=p['T_i'])

        self.lag_p.input = self.i_p.output
        self.lag_q.input = self.i_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def input_list(self):
        return ['P_setp', 'V_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']]) * v_n
        return np.sqrt(3) * V * self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']]) * v_n
        return np.sqrt(3) * V * self.I_q(x, v)

    def V(self, x, v):
        return abs(v[self.bus_idx_red['terminal']])

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        return self.bus_idx_red['terminal'], (self.I_d(x, v) - 1j*self.I_q(x, v))/i_n[self.bus_idx_red['terminal']]
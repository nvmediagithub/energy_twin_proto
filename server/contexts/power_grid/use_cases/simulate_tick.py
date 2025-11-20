from __future__ import annotations
from dataclasses import dataclass
import random, math
from ..domain.services.power_flow_solver import RadialPQDistFlowSolver
from ..domain.services.load_profiles import LoadProfiles

@dataclass
class SimParams:
    dt: float = 1.0          # seconds per tick
    noise_sigma: float = 0.10

class SimulateTick:
    def __init__(self, params: SimParams = SimParams()):
        self.params = params
        self.solver = RadialPQDistFlowSolver()

    def execute(self, g, t: float):
        # hour of day (t in seconds)
        hour = (t / 3600.0) % 24.0

        for n in g.consumers():
            base=float(n.props.get("base_kw",1.5))
            profile=str(n.props.get("profile","residential"))
            cos_phi=float(n.props.get("cos_phi",0.95))
            cos_phi=max(0.5, min(1.0, cos_phi))

            mult = LoadProfiles.multiplier(profile, hour)
            p = max(0.1, random.gauss(base*mult, base*mult*self.params.noise_sigma))

            # Q from cosφ: Q = P * tan(arccos(cosφ))
            phi = math.acos(cos_phi)
            q = p * math.tan(phi)

            n.state["p_kw"]=p
            n.state["q_kvar"]=q
            n.state["s_kva"]=math.sqrt(p*p + q*q)

        self.solver.solve(g)
        g.meta["sim_time"]=t
        g.meta["hour"]=hour
        return g

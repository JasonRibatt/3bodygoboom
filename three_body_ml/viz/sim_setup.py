"""Builds a REBOUND simulation exactly as data_generator.simulate_encounter does."""
import rebound

def build(a_out, e_out, phase_out, r_star=0.005, a_in=0.5):
    sim = rebound.Simulation(); sim.G = 1.0; sim.integrator = "ias15"
    sim.collision = "direct"; sim.collision_resolve = "halt"
    sim.add(m=1.0, r=r_star)
    sim.add(m=1.0, a=a_in, e=0.0, f=0.0, r=r_star)
    sim.move_to_com()
    sim.add(m=1.0, a=a_out, e=e_out, M=phase_out, r=r_star)
    sim.move_to_com()
    return sim

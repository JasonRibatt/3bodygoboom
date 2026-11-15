import rebound
import numpy as np
import csv
import itertools
from collections import Counter

# compute  orbital elements between two bodies
def get_orbital_elements(mu, dx, dy, dz, dvx, dvy, dvz):
    r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-15)
    v2 = dvx**2 + dvy**2 + dvz**2
    rv = dx*dvx + dy*dvy + dz*dvz
    E = 0.5*v2 - mu/r
    
    # bound systems only
    a = -mu / (2.0*E) if E < 0.0 else np.inf
        
    hx = dy*dvz - dz*dvy
    hy = dz*dvx - dx*dvz
    hz = dx*dvy - dy*dvx
    h2 = hx**2 + hy**2 + hz**2
    
    # prevent negative sqrt from float precision issues
    e = np.sqrt(max(0.0, 1.0 + (2.0*E*h2)/(mu**2)))
    
    return a, e, E, rv, r

# compute total energy and z angular momentum in COM frame
def get_conserved_quantities(sim):
    p = sim.particles
    K = 0.5 * sum(pt.m * (pt.vx**2 + pt.vy**2 + pt.vz**2) for pt in p)
    U = 0.0
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        rij = np.sqrt((p[i].x - p[j].x)**2 + (p[i].y - p[j].y)**2 + (p[i].z - p[j].z)**2)
        U -= sim.G * p[i].m * p[j].m / max(rij, 1e-15)
    Lz = sum(pt.m * (pt.x * pt.vy - pt.y * pt.vx) for pt in p)
    return K + U, Lz

# identify most tightly bound pair (E_ij < 0) and outer 3rd body elements
def find_bound_hierarchy(sim):
    p = sim.particles
    pairs = [(0, 1), (1, 2), (0, 2)]
    pair_stats = {}
    
    for i, j in pairs:
        mu_ij = sim.G * (p[i].m + p[j].m)
        a_ij, e_ij, E_ij, rv_ij, r_ij = get_orbital_elements(
            mu_ij,
            p[i].x - p[j].x, p[i].y - p[j].y, p[i].z - p[j].z,
            p[i].vx - p[j].vx, p[i].vy - p[j].vy, p[i].vz - p[j].vz
        )
        pair_stats[(i, j)] = (a_ij, e_ij, E_ij, rv_ij, r_ij)
        
    # filter for gravitationally bound pairs
    bound_pairs = {pair: stats for pair, stats in pair_stats.items() if stats[2] < 0.0}
    if not bound_pairs:
        return None
        
    # select pair with most negative specific orbital energy
    best_pair = min(bound_pairs, key=lambda k: bound_pairs[k][2])
    i, j = best_pair
    k = [x for x in (0, 1, 2) if x not in (i, j)][0]
    a_in, e_in, E_in, _, r_in = bound_pairs[best_pair]
    
    # inner binary center of mass
    m_in = p[i].m + p[j].m
    x_in = (p[i].x*p[i].m + p[j].x*p[j].m) / m_in
    y_in = (p[i].y*p[i].m + p[j].y*p[j].m) / m_in
    z_in = (p[i].z*p[i].m + p[j].z*p[j].m) / m_in
    vx_in = (p[i].vx*p[i].m + p[j].vx*p[j].m) / m_in
    vy_in = (p[i].vy*p[i].m + p[j].vy*p[j].m) / m_in
    vz_in = (p[i].vz*p[i].m + p[j].vz*p[j].m) / m_in
    
    # outer body relative to inner binary COM
    mu_out = sim.G * (m_in + p[k].m)
    a_out, e_out, E_out, rv_out, r_out = get_orbital_elements(
        mu_out,
        p[k].x - x_in, p[k].y - y_in, p[k].z - z_in,
        p[k].vx - vx_in, p[k].vy - vy_in, p[k].vz - vz_in
    )
    
    return {
        "pair": best_pair, "k": k,
        "a_in": a_in, "e_in": e_in, "E_in": E_in, "r_in": r_in,
        "a_out": a_out, "e_out": e_out, "E_out": E_out, "rv_out": rv_out, "r_out": r_out
    }

# check if hierarchy state is bound and compute Mardling-Aarseth quantities
def compute_ma_quantities(h):
    if h is None:
        return False, np.nan, np.nan
    valid_bounds = (
        h["E_in"] < 0.0 and h["E_out"] < 0.0 and
        np.isfinite(h["a_in"]) and h["a_in"] > 0.0 and
        np.isfinite(h["a_out"]) and h["a_out"] > 0.0 and
        0.0 <= h["e_in"] < 1.0 and 0.0 <= h["e_out"] < 1.0
    )
    if not valid_bounds:
        return False, np.nan, np.nan
        
    threshold = 2.8 * (1.5**0.4) * ((1.0 + h["e_out"])**0.4) / ((1.0 - h["e_out"])**1.2)
    ratio = h["a_out"] / h["a_in"]
    return True, ratio, threshold

# run a single 3-body encounter with period-scaled t_max and empirical survival labeling
def simulate_encounter(a_out, e_out, phase_out, r_star=0.005, a_in=0.5, n_orbits=250.0,
                       n_traj_steps=20, dt_traj=1.0, save_traj=True):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.collision = "direct"
    sim.collision_resolve = "halt"
    
    # inner binary system
    sim.add(m=1.0, r=r_star)
    sim.add(m=1.0, a=a_in, e=0.0, f=0.0, r=r_star)
    sim.move_to_com()
    
    # bound outer 3rd body (mean anomaly M=phase_out avoids pericenter sampling bias)
    sim.add(m=1.0, a=a_out, e=e_out, M=phase_out, r=r_star)
    sim.move_to_com()
    
    E0, Lz0 = get_conserved_quantities(sim)
    traj_array = np.full((n_traj_steps, 12), np.nan, dtype=np.float64)
    traj_complete = False
    exchange = 0
    
    # filter out t=0 physical overlaps and positive-energy initialization artifacts
    p = sim.particles
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        if np.hypot(p[i].x - p[j].x, p[i].y - p[j].y) <= 2.0 * r_star:
            return "Artifact", traj_array, False, E0, Lz0, np.nan, np.nan, np.nan, 0
    if E0 >= 0.0:
        return "Artifact", traj_array, False, E0, Lz0, np.nan, np.nan, np.nan, 0

    # scale integration time and check windows to the outer orbital period P_out
    P_out = 2.0 * np.pi * np.sqrt((a_out**3) / 3.0)
    t_max = n_orbits * P_out
    dt_check = 0.5 * P_out
    eval_window = 2.0 * P_out
    n_eval_checks = 5
    dt_eval = eval_window / (n_eval_checks - 1)
    t_traj_end = n_traj_steps * dt_traj
    
    try:
        # 1. Integrate through fixed early trajectory logging window (t = 1..20)
        for step_idx in range(n_traj_steps):
            t_target = (step_idx + 1) * dt_traj
            sim.integrate(t_target)
            if save_traj:
                pts = sim.particles
                traj_array[step_idx] = (
                    [pts[n].x for n in range(3)] +
                    [pts[n].y for n in range(3)] +
                    [pts[n].vx for n in range(3)] +
                    [pts[n].vy for n in range(3)]
                )
            h_step = find_bound_hierarchy(sim)
            if h_step is not None and h_step["pair"] != (0, 1):
                exchange = 1
        traj_complete = True
        
        # 2. Integrate to (t_max - eval_window) with periodic early-exit ejection checks
        t_curr = t_traj_end
        while t_curr + dt_check <= t_max - eval_window:
            t_curr += dt_check
            sim.integrate(t_curr)
            h = find_bound_hierarchy(sim)
            if h is not None:
                if h["pair"] != (0, 1):
                    exchange = 1
                # early exit if inner pair is bound and outer body is unbound, moving out, and > 50 * a_in away with an energy buffer
                if h["E_in"] < 0.0 and h["E_out"] > 0.01 and h["rv_out"] > 0.0 and h["r_out"] > 50.0 * a_in:
                    Ef, _ = get_conserved_quantities(sim)
                    rel_dE = abs(Ef - E0) / abs(E0)
                    return "Ejection", traj_array, traj_complete, E0, Lz0, rel_dE, np.nan, np.nan, exchange
                
        # 3. Integrate across final verification window [t_max - eval_window, t_max]
        final_states = []
        for idx in range(n_eval_checks):
            t_target = (t_max - eval_window) + idx * dt_eval
            sim.integrate(t_target)
            h_win = find_bound_hierarchy(sim)
            if h_win is not None and h_win["pair"] != (0, 1):
                exchange = 1
            final_states.append(h_win)
            
    except rebound.Collision:
        return "Collision", traj_array, traj_complete, E0, Lz0, np.nan, np.nan, np.nan, exchange
        
    Ef, _ = get_conserved_quantities(sim)
    rel_dE = abs(Ef - E0) / abs(E0)
    h_final = final_states[-1]
    
    # if no bound pair exists at t_max, system is in an unresolved close encounter
    if h_final is None:
        return "Unresolved", traj_array, traj_complete, E0, Lz0, rel_dE, np.nan, np.nan, exchange
        
    # final ejection check (requires bound inner binary, positive outer energy, outward motion, and separation > 10 * a_in)
    if h_final["E_in"] < 0.0 and h_final["E_out"] > 0.0 and h_final["rv_out"] > 0.0 and h_final["r_out"] > 10.0 * a_in:
        return "Ejection", traj_array, traj_complete, E0, Lz0, rel_dE, np.nan, np.nan, exchange
        
    # empirical survival check: hierarchy stays bound with the same inner pair across final window
    pairs_seen = set()
    ratios, thresholds = [], []
    for h_step in final_states:
        is_valid, ratio, thresh = compute_ma_quantities(h_step)
        if not is_valid:
            return "Unresolved", traj_array, traj_complete, E0, Lz0, rel_dE, np.nan, np.nan, exchange
        pairs_seen.add(h_step["pair"])
        ratios.append(ratio)
        thresholds.append(thresh)
        
    if len(pairs_seen) == 1:
        return "Survived", traj_array, traj_complete, E0, Lz0, rel_dE, np.mean(ratios), np.mean(thresholds), exchange
        
    return "Unresolved", traj_array, traj_complete, E0, Lz0, rel_dE, np.nan, np.nan, exchange

# generate grid dataset saving scalars to CSV and trajectories to compressed NPZ
def generate_dataset(density_samples=10, a_in=0.5, n_orbits=250.0,
                     output_csv="pilot_3body_metadata.csv",
                     output_npz="pilot_3body_trajectories.npz"):
    a_outs = np.linspace(0.9, 3.6, density_samples)
    e_outs = np.linspace(0.0, 0.95, density_samples)
    phis = np.linspace(0.0, 2.0*np.pi, density_samples, endpoint=False)
    
    combinations = list(itertools.product(a_outs, e_outs, phis))
    total_runs = len(combinations)
    n_traj_steps = 20
    
    valid_trajectories = []
    valid_labels = []
    results_counter = Counter()
    artifacts_dropped = 0
    max_rel_dE = 0.0
    valid_id = 0
    
    print(f"Starting {total_runs} grid simulations...")
    
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "run_id", "a_out", "e_out", "phase_out",
            "r_peri_init", "init_overlapping",
            "E0", "Lz0", "rel_dE", "traj_complete",
            "ma_ratio_init", "ma_thresh_init",
            "ma_ratio_final", "ma_thresh_final",
            "exchange", "label"
        ])
        
        for idx, (a_out, e_out, p) in enumerate(combinations):
            #if idx % 250 == 0:
                #print(f"Running {idx}/{total_runs}...")
                
            label, traj, traj_ok, E0, Lz0, rel_dE, ma_ratio_f, ma_thresh_f, exchange = simulate_encounter(
                a_out, e_out, p, a_in=a_in, n_orbits=n_orbits, n_traj_steps=n_traj_steps, save_traj=True
            )
            
            # skip invalid initialization artifacts
            if label == "Artifact":
                artifacts_dropped += 1
                continue
                
            results_counter[label] += 1
            valid_trajectories.append(traj)
            valid_labels.append(label)
            
            if np.isfinite(rel_dE) and rel_dE > max_rel_dE:
                max_rel_dE = rel_dE
                
            r_peri_init = a_out * (1.0 - e_out)
            init_overlapping = int(r_peri_init <= a_in)
            ma_ratio_init = a_out / a_in
            ma_thresh_init = 2.8 * (1.5**0.4) * ((1.0 + e_out)**0.4) / ((1.0 - e_out)**1.2)
            
            writer.writerow([
                valid_id, a_out, e_out, p,
                r_peri_init, init_overlapping,
                E0, Lz0, rel_dE, int(traj_ok),
                ma_ratio_init, ma_thresh_init,
                ma_ratio_f, ma_thresh_f,
                exchange, label
            ])
            valid_id += 1
            
    np.savez_compressed(
        output_npz,
        trajectories=np.array(valid_trajectories, dtype=np.float64),
        labels=np.array(valid_labels, dtype=object)
    )
    
    print(f"Total valid runs saved: {valid_id} (Artifacts filtered out: {artifacts_dropped})")
    print("Class Proportions:")
    for label, count in results_counter.items():
        print(f"  {label}: {count} ({count/max(1, valid_id):.1%})")
    print(f"Max relative energy drift on non-colliding runs: {max_rel_dE:.2e}")

# generate off-grid test datasets (both standard single-run and K-nudge majority-vote ensemble)
def generate_test_datasets(n_test_runs=2000, k_ensemble=5, nudge=1e-10, a_in=0.5, n_orbits=250.0, seed=42,
                           standard_csv="test_standard_metadata.csv",
                           ensemble_csv="test_ensemble_metadata.csv",
                           output_npz="test_trajectories.npz"):
    rng = np.random.default_rng(seed)
    n_traj_steps = 20
    valid_trajectories = []
    std_labels = []
    maj_labels = []
    agreements = []
    valid_id = 0
    
    base_cols = [
        "run_id", "a_out", "e_out", "phase_out",
        "r_peri_init", "init_overlapping",
        "E0", "Lz0", "rel_dE", "traj_complete",
        "ma_ratio_init", "ma_thresh_init",
        "ma_ratio_final", "ma_thresh_final",
        "exchange", "label"
    ]
    ensemble_cols = base_cols + [
        "standard_label", "agreement_fraction",
        "votes_Survived", "votes_Ejection", "votes_Collision", "votes_Unresolved"
    ]
    
    print(f"\nStarting {n_test_runs} test simulations (with K={k_ensemble} ensemble)...")
    
    with open(standard_csv, mode="w", newline="") as std_f, open(ensemble_csv, mode="w", newline="") as ens_f:
        std_writer = csv.writer(std_f)
        ens_writer = csv.writer(ens_f)
        std_writer.writerow(base_cols)
        ens_writer.writerow(ensemble_cols)
        
        while valid_id < n_test_runs:
            a_out = rng.uniform(0.9, 3.6)
            e_out = rng.uniform(0.0, 0.95)
            p = rng.uniform(0.0, 2.0*np.pi)
            
            # 1. Standard unperturbed run
            label, traj, traj_ok, E0, Lz0, rel_dE, ma_ratio_f, ma_thresh_f, exchange = simulate_encounter(
                a_out, e_out, p, a_in=a_in, n_orbits=n_orbits, n_traj_steps=n_traj_steps, save_traj=True
            )
            if label == "Artifact":
                continue
                
            # 2. K-1 perturbed runs around initial phase
            votes = [label]
            attempts = 0
            while len(votes) < k_ensemble and attempts < 25:
                attempts += 1
                dp = rng.uniform(-nudge, nudge)
                p_nudged = (p + (dp if dp != 0.0 else nudge)) % (2.0 * np.pi)
                lbl_k, *_ = simulate_encounter(
                    a_out, e_out, p_nudged, a_in=a_in, n_orbits=n_orbits, n_traj_steps=n_traj_steps, save_traj=False
                )
                if lbl_k != "Artifact":
                    votes.append(lbl_k)
                    
            if len(votes) < k_ensemble:
                continue
                
            counts = Counter(votes)
            maj_label, top_count = counts.most_common(1)[0]
            agreement = top_count / k_ensemble
            
            r_peri_init = a_out * (1.0 - e_out)
            init_overlapping = int(r_peri_init <= a_in)
            ma_ratio_init = a_out / a_in
            ma_thresh_init = 2.8 * (1.5**0.4) * ((1.0 + e_out)**0.4) / ((1.0 - e_out)**1.2)
            
            row = [
                valid_id, a_out, e_out, p,
                r_peri_init, init_overlapping,
                E0, Lz0, rel_dE, int(traj_ok),
                ma_ratio_init, ma_thresh_init,
                ma_ratio_f, ma_thresh_f,
                exchange, label
            ]
            std_writer.writerow(row)
            
            ens_row = row[:-1] + [
                maj_label, label, agreement,
                counts["Survived"], counts["Ejection"], counts["Collision"], counts["Unresolved"]
            ]
            ens_writer.writerow(ens_row)
            
            valid_trajectories.append(traj)
            std_labels.append(label)
            maj_labels.append(maj_label)
            agreements.append(agreement)
            valid_id += 1
            
            #if valid_id % 100 == 0:
             #   print(f"Test progress: {valid_id}/{n_test_runs} (Mean agreement: {np.mean(agreements):.1%})")
                
    np.savez_compressed(
        output_npz,
        trajectories=np.array(valid_trajectories, dtype=np.float64),
        labels=np.array(std_labels, dtype=object),
        majority_labels=np.array(maj_labels, dtype=object)
    )
    print(f"Test datasets complete. Mean ensemble agreement: {np.mean(agreements):.1%}")

if __name__ == "__main__":
    # 1. Generate main training/validation grid dataset
    generate_dataset(
        density_samples=25,
        output_csv="pilot_3body_metadata.csv",
        output_npz="pilot_3body_trajectories.npz"
    )
    
    # 2.generate the standard + ensemble test sets
    generate_test_datasets(n_test_runs=2000, k_ensemble=5)

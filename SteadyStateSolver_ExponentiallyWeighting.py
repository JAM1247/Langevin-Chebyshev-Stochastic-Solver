import functools
import jax
import jax.numpy as jnp
from jax import random, lax, vmap, value_and_grad
import numpy as np
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import optax
import time
import os

# For something annoying happening on my machine 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ouptut directory 
OUTPUT_DIR = "graphs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Original cell fate potential and dynamics functions remain unchanged
def cell_fate_potential(x, y, a, theta1, theta2):
    return (x**4 + y**4 + x**3 - 2*x*y**2 + 
            a*(x**2 + y**2) + theta1*x + theta2*y)

# Drift Functions 
def create_cell_fate_dynamics(a, theta1, theta2):
    def fx(x, y):
        return -(4*x**3 + 3*x**2 - 2*y**2 + 2*a*x + theta1)
    
    def fy(x, y):
        return -(4*y**3 - 4*x*y + 2*a*y + theta2)
    
    return fx, fy

# The steady state finder 
def find_x_steady_states_y0(a, theta1):
    coeffs = [4, 3, 2*a, theta1]
    roots = np.roots(coeffs)
    # Returns only real roots
    real_roots = []
    for root in roots:
        if np.abs(np.imag(root)) < 1e-10:
            real_roots.append(np.real(root))
    return sorted(real_roots)

def find_steady_states_numerical(a, theta1, theta2, search_range=3.0, n_points=100):
    fx, fy = create_cell_fate_dynamics(a, theta1, theta2)
    
    # Create a grid
    x = np.linspace(-search_range, search_range, n_points)
    y = np.linspace(-search_range, search_range, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute dynamics
    FX = fx(X, Y)
    FY = fy(X, Y)
    
    # Find where both are close to zero
    magnitude = np.sqrt(FX**2 + FY**2)
    
    # Find local minima
    steady_states = []
    threshold = 0.01
    
    for i in range(1, n_points-1):
        for j in range(1, n_points-1):
            if magnitude[i,j] < threshold:
                # Check if it's a local minimum
                if (magnitude[i,j] < magnitude[i-1,j] and 
                    magnitude[i,j] < magnitude[i+1,j] and
                    magnitude[i,j] < magnitude[i,j-1] and 
                    magnitude[i,j] < magnitude[i,j+1]):
                    steady_states.append((X[i,j], Y[i,j]))
    
    # Removing duplicates
    unique_states = []
    for state in steady_states:
        is_duplicate = False
        for unique_state in unique_states:
            if (np.abs(state[0] - unique_state[0]) < 0.1 and 
                np.abs(state[1] - unique_state[1]) < 0.1):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_states.append(state)
    
    return unique_states

#  Chebyshev basis functions for control parameterization
def chebyshev_basis(order, t):
    t_norm = 2.0 * t - 1.0  
    T = [jnp.ones_like(t_norm), t_norm]
    for k in range(2, order):
        T.append(2.0 * t_norm * T[-1] - T[-2])
    return jnp.stack(T[:order], axis=1)

#   Computing U1(t) and U2(t) using Chebyshev coefficients
@jax.jit
def get_controls(coeffs_x, coeffs_y, basis):
    return basis @ coeffs_x, basis @ coeffs_y

#  Single Euler-Maruyama step for Langevin equations
def _euler_step(x, y, fx, fy, u1, u2, sigma1, sigma2, dt, noise):
    dx = (fx(x, y) + u1) * dt + sigma1 * jnp.sqrt(dt) * noise[0]
    dy = (fy(x, y) + u2) * dt + sigma2 * jnp.sqrt(dt) * noise[1]
    return x + dx, y + dy


# function to simluate a single trajectory 
@functools.partial(jax.jit, static_argnums=(2, 3))
def simulate_trajectory(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, key):
    
    n_steps = len(u1_vec)
    noises = random.normal(key, (n_steps - 1, 2))

    def scan_fn(carry, inp):
        x, y = carry
        u1, u2, n = inp
        x_next, y_next = _euler_step(x, y, fx, fy, u1, u2, sigma1, sigma2, dt, n)
        return (x_next, y_next), jnp.array([x_next, y_next])

    (_, _), traj = lax.scan(
        scan_fn,
        (x0, y0),
        (u1_vec[:-1], u2_vec[:-1], noises)
    )

    initial_state = jnp.array([[x0, y0]])
    traj_full = jnp.vstack([initial_state, traj])
    return traj_full


# Ensemble simulation for noise realization 
def simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, n_traj, key):
    keys = random.split(key, n_traj)

    def sim_traj_wrapper(x, y, u1, u2, s1, s2, d, k):
        return simulate_trajectory(x, y, fx, fy, u1, u2, s1, s2, d, k)

    vmapped = vmap(
        sim_traj_wrapper,
        in_axes=(None, None, None, None, None, None, None, 0)
    )

    return vmapped(x0, y0, u1_vec, u2_vec, sigma1, sigma2, dt, keys)

#  Cost Function with Exponential Weighting
@functools.partial(jax.jit, static_argnums=(4, 5, 13))
def compute_cost_with_components(coeffs, basis, x0, y0, fx, fy, sigma1, sigma2, dt, 
                                targ_x, targ_y, lam, beta, n_traj, key):
    order = basis.shape[1]
    coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
    u1_vec, u2_vec = get_controls(coeffs_x, coeffs_y, basis)

    traj = simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec,
                             sigma1, sigma2, dt, n_traj, key)

    # Computing the time vector
    n_steps = traj.shape[1]
    t_vec = jnp.arange(n_steps) * dt
    
    # Exponential weights
    weights = jnp.exp(-beta * t_vec)
    
    # Normalized weights
    weights = weights / jnp.sum(weights)
    
    # Weighted distance cost over the trajectory
    target = jnp.array([targ_x, targ_y])
    distances_squared = jnp.sum((traj - target) ** 2, axis=2)  
    weighted_distances = jnp.sum(distances_squared * weights[None, :], axis=1) 
    j_target = jnp.mean(weighted_distances)
    finals = traj[:, -1, :]
    j_terminal = jnp.mean(jnp.sum((finals - target) ** 2, axis=1))
    
    # Regularization cost 
    j_reg = dt * jnp.mean(u1_vec ** 2 + u2_vec ** 2)
    
    total_cost = j_target + lam * j_reg
    
    return total_cost, j_target, j_reg, finals, j_terminal

# Optimizer with convergence checking for different ensemble sizes
def optimize_cell_fate_control(params, verbose=True, convergence_window=50, 
                              convergence_tol=1e-4, random_seed=42):
    print("\n" + "="*60)
    print("CELL FATE CONTROL OPTIMIZATION")
    print("="*60)
    
    # Initializing the dynamics 
    fx, fy = create_cell_fate_dynamics(params['a'], params['theta1'], params['theta2'])
    
    # Converts noise parameter D into standard deviations
    sigma = jnp.sqrt(2 * params['D'])
    
    # Debugging
    if verbose:
        print(f"\nSTEADY STATE ANALYSIS:")
        print(f"  Parameters: a={params['a']}, θ1={params['theta1']}, θ2={params['theta2']}")
        
        # Find steady states along y=0
        x_roots = find_x_steady_states_y0(params['a'], params['theta1'])
        print(f"  Steady states: {[f'{x:.3f}' for x in x_roots]}")
        
        # Verify the target is close to a steady state
        target_is_steady = any(abs(x - params['target_x']) < 0.1 for x in x_roots) and abs(params['target_y']) < 0.1
        
        if target_is_steady:
            print(f"  Target ({params['target_x']:.3f}, {params['target_y']:.3f}) is a steady state!")
        else:
            print(f"  WARNING: Target ({params['target_x']:.3f}, {params['target_y']:.3f}) is NOT a steady state!")
            print(f"     This may result in poor control performance.")
    
    if verbose:
        print(f"\nPHYSICS PARAMETERS:")
        print(f"  Potential parameters: a={params['a']}, θ1={params['theta1']}, θ2={params['theta2']}")
        print(f"  Noise intensity: D={params['D']} → σ={sigma:.4f}")
    
    x0, y0 = params['x0'], params['y0']
    T, dt = params['T'], params['dt']
    n_traj = params['N']
    targ_x, targ_y = params['target_x'], params['target_y']
    lam = params['lambda_reg']
    beta = params.get('beta', 0.5)  # will change in test cases 
    order = params['chebyshev_order']
    lr = params['learning_rate']
    max_epochs = params['max_epochs']

    if verbose:
        print(f"\nCONTROL PARAMETERS:")
        print(f"  Initial state: ({x0}, {y0})")
        print(f"  Target state: ({targ_x}, {targ_y})")
        print(f"  Time horizon: T={T}, dt={dt} → {int(T/dt)} time steps")
        print(f"  Ensemble size: {n_traj} cells")
        print(f"  Control basis: Chebyshev order {order} → {2*order} coefficients")
        print(f"  Regularization: λ={lam}")
        print(f"  Exponential weight: β={beta}")
        print(f"  Learning rate: {lr}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Convergence criteria: {convergence_tol} over {convergence_window} epochs")
    
    t_vec = jnp.arange(0, T + dt, dt)
    basis = chebyshev_basis(order, t_vec / T)
    
    # Initializing coefficients (it can also be done randomly)
    coeffs = jnp.zeros(2 * order)

    master = random.PRNGKey(random_seed)
    opt = optax.adam(lr)
    opt_state = opt.init(coeffs)
    losses = []
    terminal_costs = []
    weighted_costs = []
    reg_costs = []
    distances = []
    
    if verbose:
        print(f"\nOPTIMIZATION PROGRESS:")
        print("-" * 85)
        print("Epoch  | Total Loss | Weighted Cost | Terminal Cost | Control Cost | Mean Distance | Time (s)")
        print("-" * 85)

    # Tracking for convergence check
    convergence_losses = []
    converged = False
    start_time = time.time()
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        key = random.fold_in(master, epoch)

        def loss_fn(c):
            total, weighted, reg, finals, terminal = compute_cost_with_components(
                c, basis, x0, y0, fx, fy,
                sigma, sigma, dt,
                targ_x, targ_y, lam, beta,
                n_traj, key)
            return total

        loss, grads = value_and_grad(loss_fn)(coeffs)
        
        # Getting the cost components and final states
        total, j_weighted, j_reg, finals, j_terminal = compute_cost_with_components(
            coeffs, basis, x0, y0, fx, fy,
            sigma, sigma, dt,
            targ_x, targ_y, lam, beta,
            n_traj, key)
        
        # Computing mean distance to our target
        target = jnp.array([targ_x, targ_y])
        mean_distance = jnp.mean(jnp.sqrt(jnp.sum((finals - target) ** 2, axis=1)))
        
        updates, opt_state = opt.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        losses.append(float(loss))
        weighted_costs.append(float(j_weighted))
        terminal_costs.append(float(j_terminal))
        reg_costs.append(float(j_reg))
        distances.append(float(mean_distance))
        
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(f"{epoch:5d}  | {total:10.6f} | {j_weighted:13.6f} | {j_terminal:13.6f} | "
                  f"{j_reg:12.6f} | {mean_distance:13.6f} | {epoch_time:.4f}")
        
        # Checking for convergence
        convergence_losses.append(float(loss))
        if epoch >= convergence_window:
            # Keep only the last window_size losses
            convergence_losses = convergence_losses[-convergence_window:]
            
            # Calculating relative change over window
            start_loss = convergence_losses[0]
            end_loss = convergence_losses[-1]
            
            if start_loss > 0:  # safeguard against divding by 0 (error was popping up based on this as well)
                relative_change = abs((end_loss - start_loss) / start_loss)
                
                if relative_change < convergence_tol:
                    converged = True
                    if verbose:
                        print(f"\nConverged at epoch {epoch} (relative change: {relative_change:.2e})")
                    break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
    u1_final, u2_final = get_controls(coeffs_x, coeffs_y, basis)
    key_final = random.fold_in(master, 9999)
    traj = simulate_ensemble(x0, y0, fx, fy,
                             u1_final, u2_final,
                             sigma, sigma,
                             dt, n_traj, key_final)

    # Computing the final statistics
    final_states = traj[:, -1, :]
    target = jnp.array([targ_x, targ_y])
    final_distances = jnp.sqrt(jnp.sum((final_states - target) ** 2, axis=1))
    success_rate = jnp.mean(final_distances < 0.5)
    
    if verbose:
        print("-" * 85)
        print(f"\nFINAL RESULTS:")
        print(f"  Optimization time: {total_time:.2f}s ({len(losses)} epochs)")
        print(f"  Converged: {'Yes' if converged else 'No'}")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Success rate: {success_rate*100:.1f}% (within 0.5 units)")
        print(f"  Mean distance: {jnp.mean(final_distances):.4f} ± {jnp.std(final_distances):.4f}")
        
        # Used to debug controls statistics 
        control_energy = jnp.mean(u1_final**2 + u2_final**2)
        print(f"  Control energy: {control_energy:.6f}")
        print(f"  Max control: |U₁|={jnp.max(jnp.abs(u1_final)):.3f}, |U₂|={jnp.max(jnp.abs(u2_final)):.3f}")
        print("="*60 + "\n")
    
    return dict(coeffs=coeffs,
                U1=u1_final,
                U2=u2_final,
                trajectories=traj,
                losses=np.asarray(losses),
                weighted_costs=np.asarray(weighted_costs),
                terminal_costs=np.asarray(terminal_costs),
                reg_costs=np.asarray(reg_costs),
                distances=np.asarray(distances),
                t_vec=np.asarray(t_vec),
                a=params['a'], 
                theta1=params['theta1'], 
                theta2=params['theta2'], 
                D=params['D'],
                beta=beta,
                lambda_reg=lam,
                epochs_completed=len(losses),
                converged=converged,
                optimization_time=total_time,
                success_rate=float(success_rate),
                ensemble_size=n_traj,
                chebyshev_order=order)

# Function to compute and plot potential contours
def plot_potential_contours(ax, a, theta1, theta2, xlim=(-3, 3), ylim=(-3, 3), levels=20):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cell_fate_potential(X, Y, a, theta1, theta2)
    
    # Contour plots
    contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, colors='gray', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
    contourf = ax.contourf(X, Y, Z, levels=levels, alpha=0.1, cmap='viridis')
    
    return contour, contourf

# graphs
def comprehensive_graph(results, target_x, target_y, save_name):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Cell Fate Control Optimization Results', fontsize=16, fontweight='bold')
    
    # Trajectory visualization with potential background
    ax1 = fig.add_subplot(3, 3, 1)
    plot_potential_contours(ax1, results['a'], results['theta1'], results['theta2'])
    
    trajectories = results['trajectories']
    n_plot = min(50, trajectories.shape[0])
    for i in range(n_plot):
        ax1.plot(trajectories[i, :, 0], trajectories[i, :, 1], 
                'b-', alpha=0.3, linewidth=0.5)
    
    mean_traj = jnp.mean(trajectories, axis=0)
    ax1.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2, label='Mean trajectory')
    ax1.scatter(trajectories[0, 0, 0], trajectories[0, 0, 1], 
               color='green', s=100, label='Initial state', zorder=5)
    ax1.scatter(target_x, target_y, color='red', s=100, marker='*', 
               label='Target state', zorder=5)
    
    # Steady states
    x_roots = find_x_steady_states_y0(results['a'], results['theta1'])
    for x_root in x_roots:
        ax1.scatter(x_root, 0, color='orange', s=100, marker='s', alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Controlled Trajectories with Potential Landscape')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control functions
    ax2 = fig.add_subplot(3, 3, 2)
    t_vec = results['t_vec']
    ax2.plot(t_vec, results['U1'], label='U₁(t)', linewidth=2, color='blue')
    ax2.plot(t_vec, results['U2'], label='U₂(t)', linewidth=2, color='orange')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Control amplitude')
    ax2.set_title('Optimal Control Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss evolution
    ax3 = fig.add_subplot(3, 3, 3)
    epochs = np.arange(len(results['losses']))
    ax3.semilogy(epochs, results['losses'], label='Total loss', linewidth=2)
    if 'weighted_costs' in results:
        ax3.semilogy(epochs, results['weighted_costs'], label='Trajectory cost', alpha=0.7)
    if 'terminal_costs' in results:
        ax3.semilogy(epochs, results['terminal_costs'], label='Terminal cost', alpha=0.7, linestyle='--')
    ax3.semilogy(epochs, results['reg_costs'], label='Control cost', alpha=0.7)
    ax3.set_xlabel('Optimization epoch')
    ax3.set_ylabel('Cost')
    ax3.set_title(f'Cost Evolution (N={results["ensemble_size"]})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final state distribution with potential background
    ax4 = fig.add_subplot(3, 3, 4)
    plot_potential_contours(ax4, results['a'], results['theta1'], results['theta2'])
    final_states = trajectories[:, -1, :]
    h = ax4.hist2d(final_states[:, 0], final_states[:, 1], bins=20, cmap='Blues', alpha=0.8)
    fig.colorbar(h[3], ax=ax4, label='Trajectory count')
    ax4.scatter(target_x, target_y, color='red', s=100, marker='*', zorder=5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Final State Distribution with Potential')
    
    # Distance convergence
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(epochs, results['distances'], linewidth=2)
    ax5.set_xlabel('Optimization epoch')
    ax5.set_ylabel('Mean distance to target')
    ax5.set_title('Distance Convergence')
    ax5.grid(True, alpha=0.3)
    
    # Control energy
    ax6 = fig.add_subplot(3, 3, 6)
    ctrl_energy = results['U1']**2 + results['U2']**2
    ax6.plot(t_vec, ctrl_energy, linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('U₁² + U₂²')
    ax6.set_title('Instantaneous Control Energy')
    ax6.grid(True, alpha=0.3)
    
    # Final distance CDF
    ax7 = fig.add_subplot(3, 3, 7)
    final_distances = jnp.sqrt(jnp.sum((final_states - jnp.array([target_x, target_y])) ** 2, axis=1))
    sorted_d = np.sort(final_distances)
    cdf = np.arange(len(sorted_d)) / len(sorted_d)
    ax7.plot(sorted_d, cdf, linewidth=2)
    ax7.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Success threshold')
    ax7.set_xlabel('Final distance to target')
    ax7.set_ylabel('Cumulative probability')
    ax7.set_title('Success Rate Analysis')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Parameter summary
    ax8 = fig.add_subplot(3, 3, 8)
    summary_text = (
        f"OPTIMIZATION SUMMARY\n"
        f"{'='*19}\n"
        f"Ensemble size: {results['ensemble_size']}\n"
        f"Chebyshev order: {results['chebyshev_order']}\n"
        f"Epochs: {results['epochs_completed']}\n"
        f"Converged: {'Yes' if results['converged'] else 'No'}\n"
        f"Final loss: {results['losses'][-1]:.6f}\n"
        f"Success rate: {results['success_rate']*100:.1f}%\n"
        f"Runtime: {results.get('optimization_time',0):.1f}s\n\n"
        f"POTENTIAL PARAMETERS\n"
        f"{'='*19}\n"
        f"a = {results['a']}\n"
        f"θ₁ = {results['theta1']}\n"
        f"θ₂ = {results['theta2']}\n"
        f"D = {results['D']}\n\n"
        f"CONTROL PARAMETERS\n"
        f"{'='*18}\n"
    )
    if 'beta' in results:
        summary_text += f"β = {results['beta']}\n"
    if 'lambda_reg' in results:
        summary_text += f"λ = {results['lambda_reg']}"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax8.axis('off')
    
    # Phase portrait detail with potential heatmap
    ax9 = fig.add_subplot(3, 3, 9)
    x_range = np.linspace(-2, 2, 50)
    y_range = np.linspace(-2, 2, 50)
    X_heat, Y_heat = np.meshgrid(x_range, y_range)
    Z_heat = cell_fate_potential(X_heat, Y_heat, results['a'], results['theta1'], results['theta2'])
    
    im = ax9.imshow(Z_heat, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis', alpha=0.6)
    contour = ax9.contour(X_heat, Y_heat, Z_heat, levels=15, colors='gray', alpha=0.4, linewidths=0.5)
    
    # Ploting mean trajectory with direction arrows
    mean_traj = jnp.mean(trajectories, axis=0)
    ax9.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3, alpha=0.9)
    n_arrows = 5
    arrow_indices = np.linspace(0, len(mean_traj)-2, n_arrows, dtype=int)
    for i in arrow_indices:
        dx = mean_traj[i+1, 0] - mean_traj[i, 0]
        dy = mean_traj[i+1, 1] - mean_traj[i, 1]
        ax9.arrow(mean_traj[i, 0], mean_traj[i, 1], dx*0.3, dy*0.3, 
                 head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)

    ax9.scatter(trajectories[0, 0, 0], trajectories[0, 0, 1], 
               color='white', s=150, marker='o', zorder=5, edgecolors='black', linewidth=2)
    ax9.scatter(target_x, target_y, color='yellow', s=150, marker='*', 
               zorder=5, edgecolors='black', linewidth=2)
    
    # Steady states
    x_roots = find_x_steady_states_y0(results['a'], results['theta1'])
    for x_root in x_roots:
        ax9.scatter(x_root, 0, color='orange', s=100, marker='s', alpha=0.9, 
                   edgecolors='black', linewidth=1, zorder=5)
    
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    ax9.set_title('Phase Portrait with Potential Heatmap')
    ax9.set_xlim(-2, 2)
    ax9.set_ylim(-2, 2)
    
    plt.tight_layout()
    # making sure files go into the right place
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive graphs saved: {save_name}")
    
    return fig

# Lambda sweep function thats tests different values of lambda to see control behavior
def sweep_lambda(base_params, lambda_values, save_name="S1_lambda_parameter_sweep", verbose=False):
    print(f"\n{'='*60}")
    print("REGULARIZATION PARAMETER SWEEP")
    print(f"{'='*60}")
    
    # Main trajectory comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Effect of Regularization Parameter λ on Control Performance', 
                fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, lam in enumerate(lambda_values):
        # Create params for this lambda
        params = base_params.copy()
        params['lambda_reg'] = lam
        
        print(f"\nTesting λ = {lam:.6f}")
        
        # Running optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # Subplot
        ax = axes[idx]
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
        mean_traj = jnp.mean(results['trajectories'], axis=0) # mean trajectory
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3)
        
        n_plot = min(20, results['trajectories'].shape[0]) # individual trajectories
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        max_control = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        ax.set_title(f'λ = {lam:.1e}\nSuccess: {results["success_rate"]*100:.0f}%, Max|U| = {max_control:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}_trajectories.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis figures
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Regularization Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Extracting metrics
    max_controls = []
    avg_controls = []
    success_rates = []
    final_losses = []
    
    for results in results_list:
        max_u = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        avg_u = jnp.mean(results['U1']**2 + results['U2']**2)
        max_controls.append(max_u)
        avg_controls.append(avg_u)
        success_rates.append(results['success_rate'])
        final_losses.append(results['losses'][-1])
    
    # Control magnitude vs Lambda
    ax1.loglog(lambda_values, max_controls, 'o-', label='Max |U|', markersize=8, linewidth=2)
    ax1.loglog(lambda_values, avg_controls, 's-', label='Avg U²', markersize=8, linewidth=2)
    ax1.set_xlabel('λ (regularization parameter)')
    ax1.set_ylabel('Control magnitude')
    ax1.set_title('Control Magnitude vs Regularization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #Success rate vs Lambda
    ax2.semilogx(lambda_values, 100*np.array(success_rates), 'o-', markersize=8, linewidth=2, color='green')
    ax2.set_xlabel('λ (regularization parameter)')
    ax2.set_ylabel('Success rate (%)')
    ax2.set_title('Success Rate vs Regularization')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Final loss
    ax3.loglog(lambda_values, final_losses, 'o-', markersize=8, linewidth=2, color='purple')
    ax3.set_xlabel('λ (regularization parameter)')
    ax3.set_ylabel('Final loss')
    ax3.set_title('Final Loss vs Regularization')
    ax3.grid(True, alpha=0.3)
    
    # Summary table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    headers = ['λ', 'Success Rate (%)', 'Max |U|', 'Avg U²', 'Final Loss']
    
    for lam, results in zip(lambda_values, results_list):
        max_u = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        avg_u = jnp.mean(results['U1']**2 + results['U2']**2)
        table_data.append([
            f"{lam:.1e}",
            f"{results['success_rate']*100:.1f}",
            f"{max_u:.2f}",
            f"{avg_u:.3f}",
            f"{results['losses'][-1]:.5f}"
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Parameter Sweep Summary', fontweight='bold')
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lambda sweep figures saved: {save_name}_*.png")
    
    # Summary statistics
    print("\nLAMBDA SWEEP SUMMARY:")
    print("-" * 70)
    print(f"{'Lambda':<12} {'Success Rate':<15} {'Max |U|':<15} {'Avg U²':<15} {'Final Loss':<15}")
    print("-" * 70)
    for lam, results in zip(lambda_values, results_list):
        max_u = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        avg_u = jnp.mean(results['U1']**2 + results['U2']**2)
        print(f"{lam:<12.6f} {results['success_rate']*100:>12.1f}%  "
              f"{max_u:>13.4f}  {avg_u:>13.6f}  {results['losses'][-1]:>15.6f}")
    
    return results_list

# Beta sweep function 
def sweep_beta(base_params, beta_values, save_name="S2_beta_parameter_sweep", verbose=False):
    print(f"\n{'='*60}")
    print("EXPONENTIAL WEIGHTING PARAMETER SWEEP")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle('Effect of Exponential Weighting Parameter β', fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, beta in enumerate(beta_values):
        # Creating params for this beta
        params = base_params.copy()
        params['beta'] = beta
        
        print(f"\nTesting β = {beta}")
        
        # Runing optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        ax = axes[idx]        
        
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
    
        # mean trajectory graphs
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3)
        
        # Individual trajectories
        n_plot = min(20, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        ax.set_title(f'β = {beta:.1f}\nSuccess: {results["success_rate"]*100:.0f}%, '
                    f'Epochs: {results["epochs_completed"]}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Beta sweep figure saved: {save_name}.png")
    
    return results_list

# Parameter sweep function 
# Comparing control performance across different potential landscapes to see where the potential lansdscepa can be
def sweep_potentials(base_params, scenarios, save_name="potential_landscape_comparison", verbose=False):
    print(f"\n{'='*60}")
    print("POTENTIAL LANDSCAPE COMPARISON")
    print(f"{'='*60}")
    
    # Potential landscape visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Control Performance Across Different Potential Landscapes', 
                fontsize=16, fontweight='bold')
    
    #subplot layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # issue that was coming up  while plotting
    try:
        cmap = plt.colormaps['tab10']
    except:
        cmap = plt.cm.get_cmap('tab10')
    
    results_list = []
    
    # Plotting each scenario's potential and trajectory
    for idx, scenario in enumerate(scenarios[:6]):  
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Potential heatmap background
        x_range = np.linspace(-2.5, 2.5, 40)
        y_range = np.linspace(-2.5, 2.5, 40)
        X_heat, Y_heat = np.meshgrid(x_range, y_range)
        Z_heat = cell_fate_potential(X_heat, Y_heat, scenario['a'], scenario['theta1'], scenario['theta2'])
        
        # Adding heatmap with custom colormap
        im = ax.imshow(Z_heat, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', 
                      cmap='viridis', alpha=0.4, vmin=np.min(Z_heat), vmax=np.min(Z_heat) + 20)
        
        contour = ax.contour(X_heat, Y_heat, Z_heat, levels=10, colors='gray', alpha=0.6, linewidths=0.5)
        
        #  Params for this scenario
        params = base_params.copy()
        params.update({
            'a': scenario['a'],
            'theta1': scenario['theta1'],
            'theta2': scenario['theta2'],
            'target_x': scenario['target_x'],
            'target_y': scenario['target_y']
        })
        
        print(f"\nScenario: {scenario['label']}")
        print(f"  Parameters: a={scenario['a']}, θ1={scenario['theta1']}, θ2={scenario['theta2']}")
        print(f"  Target: ({scenario['target_x']:.3f}, {scenario['target_y']:.3f})")
        
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # mean trajectory
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 
                color='red', linewidth=3, alpha=0.9, zorder=5)
        
        # individual trajectories
        n_plot = min(10, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'white', alpha=0.3, linewidth=0.5, zorder=3)
        
        # initial and target states
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='lime', s=150, marker='o', zorder=6, 
                  edgecolor='black', linewidth=2, label='Start')
        ax.scatter(scenario['target_x'], scenario['target_y'], 
                  color='yellow', s=150, marker='*', zorder=6,
                  edgecolor='black', linewidth=2, label='Target')
        
        # Steady states if they exist
        try:
            x_roots = find_x_steady_states_y0(scenario['a'], scenario['theta1'])
            for x_root in x_roots:
                if -2.5 <= x_root <= 2.5:  # Only plot if in visible range
                    ax.scatter(x_root, 0, color='orange', s=100, marker='s', 
                             alpha=0.9, edgecolor='black', linewidth=1, zorder=6)
        except:
            pass
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f'{scenario["label"]}\nSuccess: {results["success_rate"]*100:.0f}%', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Comparison plots in bottom row
    ax_comp1 = fig.add_subplot(gs[2, :2])
    ax_comp2 = fig.add_subplot(gs[2, 2:])
    
    # Success rates comparison
    scenario_names = [s['label'] for s in scenarios[:len(results_list)]]
    success_rates = [r['success_rate']*100 for r in results_list]
    colors = [cmap(i) for i in range(len(results_list))]
    
    bars1 = ax_comp1.bar(range(len(success_rates)), success_rates, color=colors, alpha=0.7)
    ax_comp1.set_xlabel('Scenario')
    ax_comp1.set_ylabel('Success Rate (%)')
    ax_comp1.set_title('Success Rate Comparison')
    ax_comp1.set_xticks(range(len(scenario_names)))
    ax_comp1.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax_comp1.grid(True, alpha=0.3)
    
    # Labeling
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax_comp1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Control energy vs Success Rate
    control_energies = [jnp.mean(r['U1']**2 + r['U2']**2) for r in results_list]
    
    scatter = ax_comp2.scatter(control_energies, success_rates, 
                              c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, (energy, success, name) in enumerate(zip(control_energies, success_rates, scenario_names)):
        ax_comp2.annotate(name, (energy, success), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8, ha='left')
    
    ax_comp2.set_xlabel('Average Control Energy')
    ax_comp2.set_ylabel('Success Rate (%)')
    ax_comp2.set_title('Control Energy vs Success Rate')
    ax_comp2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Landscape comparison saved: {save_name}.png")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("LANDSCAPE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<25} {'Success Rate':<12} {'Mean Distance':<15} {'Control Energy':<15}")
    print("-" * 67)
    
    for scenario, results in zip(scenarios[:len(results_list)], results_list):
        control_energy = jnp.mean(results['U1']**2 + results['U2']**2)
        print(f"{scenario['label']:<25} {results['success_rate']*100:>9.1f}%     "
              f"{results['distances'][-1]:>12.4f}      {control_energy:>12.6f}")
    
    return results_list

# Function to test control to actual steady states
def test_steady_state_scenarios(save_name="steady_state_analysis", verbose=True):
    # Test control performance when targeting actual steady states
    print(f"\n{'='*60}")
    print("STEADY STATE TARGETING ANALYSIS")
    print(f"{'='*60}")
    
    scenarios = [
        # Original parameters
        {
            'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
            'x0': 0.0, 'y0': 0.0,
            'label': 'Original Parameters'
        },
        # Negative a value with monostable potential
        {
            'a': -1.0, 'theta1': 0.0, 'theta2': 0.0,
            'x0': 0.0, 'y0': 0.0,
            'label': 'Monostable (a=-1)'
        },
        # Negative a value with bias
        {
            'a': -2.0, 'theta1': 2.0, 'theta2': 0.0,
            'x0': -0.5, 'y0': 0.0,
            'label': 'Monostable with Bias'
        },
        # Symmetric bistable
        {
            'a': 1.0, 'theta1': 0.0, 'theta2': 0.0,
            'x0': 0.1, 'y0': 0.0, 
            'label': 'Symmetric Bistable'
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle('Steady State Targeting Validation', fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, scenario in enumerate(scenarios):
        print(f"\nAnalyzing: {scenario['label']}")
        print(f"Parameters: a={scenario['a']}, θ1={scenario['theta1']}, θ2={scenario['theta2']}")
        
        # Finds steady states 
        x_roots = find_x_steady_states_y0(scenario['a'], scenario['theta1'])
        print(f"Steady states along y=0: {x_roots}")
        
        # Target based on steady states
        if len(x_roots) > 0:
            # For scenarios with multiple steady states, we choose the one furthest from origin
            if len(x_roots) > 1:
                # Preferring positive ones 
                positive_roots = [x for x in x_roots if x > 0.1]
                if positive_roots:
                    target_x = positive_roots[0]
                else:
                    target_x = x_roots[-1]  # Take the last one
            else:
                target_x = x_roots[0]
        else:
            print("WARNING: No steady states found!")
            target_x = 1.0  # Fallback
        
        # Setting up parameters
        params = {
            'x0': scenario['x0'], 'y0': scenario['y0'],
            'target_x': target_x, 'target_y': 0.0,
            'a': scenario['a'],
            'theta1': scenario['theta1'],
            'theta2': scenario['theta2'],
            'D': 0.05,
            'T': 5.0,
            'dt': 0.01,
            'N': 500,
            'lambda_reg': 0.001,  # Lower regularization for better control
            'beta': 0.1,  # Small exponential weighting
            'chebyshev_order': 20,
            'learning_rate': 0.01,
            'max_epochs': 500
        }
        
        print(f"Targeting steady state at ({target_x:.3f}, 0)")
        
        # Optimization
        results = optimize_cell_fate_control(params, verbose=verbose)
        results_list.append(results)
        
        # Graphs
        ax = axes[idx]
        plot_potential_contours(ax, scenario['a'], scenario['theta1'], scenario['theta2'])
        
        # Trajectories graphs
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3, label='Mean trajectory')
        
        n_plot = min(15, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # Start, target, and steady states
        ax.scatter(scenario['x0'], scenario['y0'], color='green', s=120, 
                  marker='o', zorder=5, label='Initial', edgecolor='black')
        ax.scatter(target_x, 0, color='red', s=120, marker='*', 
                  zorder=5, label='Target', edgecolor='black')
        
        # Mark all steady states
        for x_ss in x_roots:
            ax.scatter(x_ss, 0, color='orange', s=100, marker='s', 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_title(f"{scenario['label']}\nSuccess: {results['success_rate']*100:.0f}%")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Steady state analysis saved: {save_name}.png")
    
    # Summary
    print("\n" + "="*60)
    print("STEADY STATE TARGETING SUMMARY:")
    print("-" * 70)
    print(f"{'Scenario':<30} {'Target':<10} {'Success Rate':<15} {'Final Distance':<15}")
    print("-" * 70)
    for scenario, results in zip(scenarios, results_list):
        target_x = results['trajectories'][0,-1,0]  # Get the actual target used
        print(f"{scenario['label']:<30} "
              f"({target_x:.2f},0)  "
              f"{results['success_rate']*100:>10.1f}%     "
              f"{results['distances'][-1]:>10.4f}")
    
    return results_list

def main():
  
    print("\COMPREHENSIVE STEADY STATE ANALYSIS")
    steady_state_results = test_steady_state_scenarios()
    
    x_roots = find_x_steady_states_y0(2.0, 5.0) 
    target_x = x_roots[0] if x_roots else 1.0
    
    main_params = {
        'x0': 0.0, 'y0': 0.0,
        'target_x': target_x, 'target_y': 0.0,
        'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
        'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 1000,
        'lambda_reg': 0.001, 'beta': 0.1,
        'chebyshev_order': 25, 'learning_rate': 0.01, 'max_epochs': 500
    }
    
    print(f"\nMain Demonstration: Targeting steady state at ({target_x:.3f}, 0)")
    main_results = optimize_cell_fate_control(main_params, verbose=True)
    comprehensive_graph(main_results, main_params['target_x'], main_params['target_y'], 
                       "optimal_steady_state_control.png")
    

    print("\nPARAMETER OPTIMIZATION")
    
    # Lambda sweep called
    base_params_optimized = {
        'x0': 0.0, 'y0': 0.0,
        'target_x': target_x, 'target_y': 0.0,
        'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
        'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 1000,
        'lambda_reg': 0.01, 'beta': 0.1,
        'chebyshev_order': 25, 'learning_rate': 0.01, 'max_epochs': 500
    }
    
    lambda_values = np.logspace(-4, -1, 6)
    lambda_results = sweep_lambda(base_params_optimized, lambda_values)
    
    # Beta sweep with correct parameters
    beta_values = [0, 0.1, 0.5, 1.0]
    beta_results = sweep_beta(base_params_optimized, beta_values)
    
    # Potential Landscape Comparison 
    print("\nPOTENTIAL LANDSCAPE COMPARISON")
    
    # Test scenarios with automatic steady state targeting
    test_cases = [
        {'a': -1.0, 'theta1': 0.0, 'theta2': 0.0, 'label': 'Monostable (a=-1)'},
        {'a': -2.0, 'theta1': 1.0, 'theta2': 0.0, 'label': 'Monostable with Bias'},
        {'a': 1.0, 'theta1': 0.0, 'theta2': 0.0, 'label': 'Symmetric Bistable'},
        {'a': 2.0, 'theta1': 3.0, 'theta2': -3.0, 'label': 'Asymmetric Bistable'},
        {'a': 0.5, 'theta1': 2.0, 'theta2': 0.0, 'label': 'Shallow Wells'},
        {'a': 3.0, 'theta1': 0.0, 'theta2': 0.0, 'label': 'Deep Wells'}
    ]
    
    # Auto-generated steady state targets 
    potential_scenarios = []
    for case in test_cases:
        x_ss = find_x_steady_states_y0(case['a'], case['theta1'])
        if x_ss:
            # preferring positive steady state if available
            positive_ss = [x for x in x_ss if x > 0.1]
            target = positive_ss[0] if positive_ss else x_ss[-1]
        else:
            target = 1.0  # fallback
        
        scenario = case.copy()
        scenario['target_x'] = target
        scenario['target_y'] = 0.0
        potential_scenarios.append(scenario)
        print(f"  {case['label']}: targeting steady state at ({target:.3f}, 0)")
    
    base_params_landscape = {
        'x0': 0.0, 'y0': 0.0,
        'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 500,
        'lambda_reg': 0.001, 'beta': 0.1,
        'chebyshev_order': 20, 'learning_rate': 0.01, 'max_epochs': 500
    }
    
    landscape_results = sweep_potentials(base_params_landscape, potential_scenarios)



if __name__ == "__main__":
    main()

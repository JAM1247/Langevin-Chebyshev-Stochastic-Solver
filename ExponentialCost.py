import functools
import jax
import jax.numpy as jnp
from jax import random, lax, vmap, value_and_grad
import numpy as np
import matplotlib.pyplot
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import optax
import time
import os


# For something annoying happening on my machine 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


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

#  Chebyshev basis functions for control parameterization
def chebyshev_basis(order, t):
    t_norm = 2.0 * t - 1.0  # defining polynomials 
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
    # dy = fy dt + U2 dt + σ2 dW2
    dx = (fx(x, y) + u1) * dt + sigma1 * jnp.sqrt(dt) * noise[0]

    # dx = fx dt + U1 dt + σ1 dW1
    dy = (fy(x, y) + u2) * dt + sigma2 * jnp.sqrt(dt) * noise[1]
    return x + dx, y + dy

@functools.partial(jax.jit, static_argnums=(2, 3))
def simulate_trajectory(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, key):
    # Simulates a single trajectory 
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

def simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, n_traj, key):
    # Ensemble simulation for noise realization 
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

    # Compute time vector
    n_steps = traj.shape[1]
    t_vec = jnp.arange(n_steps) * dt
    
    # Exponential weights
    weights = jnp.exp(-beta * t_vec)
    # Normalize weights
    weights = weights / jnp.sum(weights)
    
    # Compute weighted distance cost over entire trajectory
    target = jnp.array([targ_x, targ_y])
    distances_squared = jnp.sum((traj - target) ** 2, axis=2)  # Shape: (n_traj, n_steps)
    
    # Apply exponential weighting
    weighted_distances = jnp.sum(distances_squared * weights[None, :], axis=1)  # Shape: (n_traj,)
    j_target = jnp.mean(weighted_distances)
    
    # Also compute terminal cost for comparison
    finals = traj[:, -1, :]
    j_terminal = jnp.mean(jnp.sum((finals - target) ** 2, axis=1))
    
    # Regularization cost to understand the control effort
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
    
    # Convert noise parameter D into standard deviations
    sigma = jnp.sqrt(2 * params['D'])
    
    if verbose:
        print(f"\nPHYSICS PARAMETERS:")
        print(f"  Potential parameters: a={params['a']}, θ1={params['theta1']}, θ2={params['theta2']}")
        print(f"  Noise intensity: D={params['D']} → σ={sigma:.4f}")
    
    x0, y0 = params['x0'], params['y0']
    T, dt = params['T'], params['dt']
    n_traj = params['N']
    targ_x, targ_y = params['target_x'], params['target_y']
    lam = params['lambda_reg']
    beta = params.get('beta', 0.5)  # Default beta if not specified
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
        
        # Track metrics
        losses.append(float(loss))
        weighted_costs.append(float(j_weighted))
        terminal_costs.append(float(j_terminal))
        reg_costs.append(float(j_reg))
        distances.append(float(mean_distance))
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(f"{epoch:5d}  | {total:10.6f} | {j_weighted:13.6f} | {j_terminal:13.6f} | "
                  f"{j_reg:12.6f} | {mean_distance:13.6f} | {epoch_time:.4f}")
        
        # Check for convergence
        convergence_losses.append(float(loss))
        if epoch >= convergence_window:
            # Keep only the last window_size losses
            convergence_losses = convergence_losses[-convergence_window:]
            
            # Calculate relative change over window
            start_loss = convergence_losses[0]
            end_loss = convergence_losses[-1]
            
            if start_loss > 0:  # Avoid division by zero
                relative_change = abs((end_loss - start_loss) / start_loss)
                
                if relative_change < convergence_tol:
                    converged = True
                    if verbose:
                        print(f"\nConverged at epoch {epoch} with relative change {relative_change:.8f}")
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
        print(f"  Completed epochs: {len(losses)}")
        print(f"  Total optimization time: {total_time:.2f} seconds")
        print(f"  Converged: {converged}")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Success rate (within 0.5 of target): {success_rate*100:.1f}%")
        print(f"  Mean final distance to target: {jnp.mean(final_distances):.4f}")
        print(f"  Std final distance to target: {jnp.std(final_distances):.4f}")
        
        # Used to debug controls statistics 
        control_energy = jnp.mean(u1_final**2 + u2_final**2)
        print(f"\nCONTROL STATISTICS:")
        print(f"  Average control energy: {control_energy:.6f}")
        print(f"  Max |U1|: {jnp.max(jnp.abs(u1_final)):.4f}")
        print(f"  Max |U2|: {jnp.max(jnp.abs(u2_final)):.4f}")
        
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
    """Add contour plot of the potential to existing axis"""
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cell_fate_potential(X, Y, a, theta1, theta2)
    
    # Create contour plot
    contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, colors='gray', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
    
    # Also add a subtle filled contour
    contourf = ax.contourf(X, Y, Z, levels=levels, alpha=0.1, cmap='viridis')
    
    return contour, contourf

# Updated graphs with potential background
def plot_results(results, target_x, target_y, save_name='cell_fate_control.png'):
    fig = plt.figure(figsize=(16, 12))
    
    # Trajectories with potential background
    ax1 = fig.add_subplot(3, 3, 1)
    
    # Add potential contours
    plot_potential_contours(ax1, results['a'], results['theta1'], results['theta2'])
    
    trajectories = results['trajectories']
    n_plot = min(50, trajectories.shape[0])
    for i in range(n_plot):
        ax1.plot(trajectories[i, :, 0], trajectories[i, :, 1], 
                alpha=0.3, color='blue', linewidth=0.5)
    mean_traj = jnp.mean(trajectories, axis=0)
    ax1.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2, label='Mean')
    ax1.scatter(0, 0, color='green', s=100, label='Start', zorder=5)
    ax1.scatter(target_x, target_y, color='red', s=100, marker='*', label='Target', zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Cell Fate Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Controls
    ax2 = fig.add_subplot(3, 3, 2)
    t_vec = results['t_vec']
    ax2.plot(t_vec, results['U1'], label='U1(t)', linewidth=2)
    ax2.plot(t_vec, results['U2'], label='U2(t)', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_title('Control Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss
    ax3 = fig.add_subplot(3, 3, 3)
    epochs = np.arange(len(results['losses']))
    ax3.semilogy(epochs, results['losses'], label='Total Loss')
    ax3.semilogy(epochs, results['weighted_costs'], label='Weighted Cost', alpha=0.7)
    ax3.semilogy(epochs, results['terminal_costs'], label='Terminal Cost', alpha=0.7, linestyle='--')
    ax3.semilogy(epochs, results['reg_costs'], label='Regularization Cost', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'Optimization Progress (N={results["ensemble_size"]}, β={results["beta"]})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final distribution with potential background
    ax4 = fig.add_subplot(3, 3, 4)
    plot_potential_contours(ax4, results['a'], results['theta1'], results['theta2'])
    final_states = trajectories[:, -1, :]
    h = ax4.hist2d(final_states[:, 0], final_states[:, 1], bins=20, cmap='Blues', alpha=0.8)
    fig.colorbar(h[3], ax=ax4, label='Count')
    ax4.scatter(target_x, target_y, color='red', s=100, marker='*', zorder=5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Final State Distribution')
    
    # Mean distance to target over epochs
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(epochs, results['distances'])
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Mean Distance to Target')
    ax5.set_title('Convergence of Mean Distance')
    ax5.grid(True, alpha=0.3)
    
    # Instantaneous Control Energy
    ax6 = fig.add_subplot(3, 3, 6)
    ctrl_energy = results['U1']**2 + results['U2']**2
    ax6.plot(t_vec, ctrl_energy)
    ax6.set_xlabel('Time')
    ax6.set_ylabel(r'U$_1^2$ + U$_2^2$')
    ax6.set_title('Instantaneous Control Energy')
    ax6.grid(True, alpha=0.3)
    
    # CDF of Final Distances
    ax7 = fig.add_subplot(3, 3, 7)
    final_distances = jnp.sqrt(jnp.sum((final_states - jnp.array([target_x, target_y])) ** 2, axis=1))
    sorted_d = np.sort(final_distances)
    cdf = np.arange(len(sorted_d)) / len(sorted_d)
    ax7.plot(sorted_d, cdf)
    ax7.set_xlabel('Distance to Target')
    ax7.set_ylabel('CDF')
    ax7.set_title('Final-distance CDF')
    ax7.grid(True, alpha=0.3)
    
    # Summary text box
    ax8 = fig.add_subplot(3, 3, 8)
    summary_text = (
        f"OPTIMIZATION SUMMARY\n"
        f"-------------------\n"
        f"Ensemble size: {results['ensemble_size']}\n"
        f"Chebyshev order: {results['chebyshev_order']}\n"
        f"Epochs completed: {results['epochs_completed']}\n"
        f"Converged: {results['converged']}\n"
        f"Final loss: {results['losses'][-1]:.6f}\n"
        f"Success rate: {results['success_rate']*100:.1f}%\n"
        f"Optimization time: {results.get('optimization_time',0):.2f}s\n"
        f"Potential params:\n"
        f"  a = {results['a']}\n"
        f"  θ1 = {results['theta1']}\n"
        f"  θ2 = {results['theta2']}\n"
        f"  D = {results['D']}\n"
        f"Control params:\n"
        f"  β = {results['beta']}\n"
        f"  λ = {results['lambda_reg']}"
    )
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax8.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Plot saved as: {save_name}")
    
    return fig

# Lambda sweep function
def sweep_lambda(base_params, lambda_values, save_name="lambda_sweep.png", verbose=False):
    """Test different values of lambda to see control behavior"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    results_list = []
    
    for idx, lam in enumerate(lambda_values):
        # Create params for this lambda
        params = base_params.copy()
        params['lambda_reg'] = lam
        
        print(f"\n{'='*60}")
        print(f"Testing λ = {lam}")
        print('='*60)
        
        # Run optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # Plot on subplot
        ax = axes[idx]
        
        # Add potential contours
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
        
        # Plot mean trajectory
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2)
        
        # Plot some individual trajectories
        n_plot = min(20, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        # Add title with key metrics
        max_control = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        ax.set_title(f'λ={lam:.4f}\n'
                    f'Success: {results["success_rate"]*100:.0f}%, '
                    f'Max|U|: {max_control:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nLambda sweep plot saved as: {save_name}")
    
    # Control magnitude analysis
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    max_controls = []
    avg_controls = []
    success_rates = []
    
    for results in results_list:
        max_u = max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2'])))
        avg_u = jnp.mean(results['U1']**2 + results['U2']**2)
        max_controls.append(max_u)
        avg_controls.append(avg_u)
        success_rates.append(results['success_rate'])
    
    # Plot control magnitude vs lambda
    ax1.loglog(lambda_values, max_controls, 'o-', label='Max |U|', markersize=8)
    ax1.loglog(lambda_values, avg_controls, 's-', label='Avg U²', markersize=8)
    ax1.set_xlabel('λ (regularization parameter)')
    ax1.set_ylabel('Control magnitude')
    ax1.set_title('Control Magnitude vs Regularization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot success rate vs lambda
    ax2.semilogx(lambda_values, 100*np.array(success_rates), 'o-', markersize=8)
    ax2.set_xlabel('λ (regularization parameter)')
    ax2.set_ylabel('Success rate (%)')
    ax2.set_title('Success Rate vs Regularization')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_name.replace('.png', '_analysis.png'), dpi=300)
    plt.close()
    print(f"Lambda analysis plot saved as: {save_name.replace('.png', '_analysis.png')}")
    
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
def sweep_beta(base_params, beta_values, save_name="beta_sweep.png", verbose=False):
    """Test different values of beta to see convergence behavior"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    results_list = []
    
    for idx, beta in enumerate(beta_values):
        # Creating params for this beta
        params = base_params.copy()
        params['beta'] = beta
        
        print(f"\n{'='*60}")
        print(f"Testing β = {beta}")
        print('='*60)
        
        # Runing optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        ax = axes[idx]        

        # Add potential contours
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
        
        # mean trajectory graphs
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2)
        
        # Plotting individual trajectories
        n_plot = min(20, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        ax.set_title(f'β={beta:.1f}\n'
                    f'Success: {results["success_rate"]*100:.0f}%, '
                    f'Epochs: {results["epochs_completed"]}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBeta sweep plot saved as: {save_name}")
    
    return results_list

# Parameter sweep function updated with potential backgrounds
def sweep_potentials(base_params, scenarios, save_name="mean_trajectory_sweep.png", verbose=False):

    # Figures posted on subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap('tab10')
    
    results_list = []
    
    # First, add potential contours for the first scenario as reference
    plot_potential_contours(ax1, scenarios[0]['a'], scenarios[0]['theta1'], 
                           scenarios[0]['theta2'], xlim=(-2, 2), ylim=(-2, 2))
    
    for idx, scenario in enumerate(scenarios):
        #  Params for this scenario
        params = base_params.copy()
        params.update({
            'a': scenario['a'],
            'theta1': scenario['theta1'],
            'theta2': scenario['theta2']
        })
        
        print(f"\n{'='*60}")
        print(f"Scenario {idx+1}: {scenario['label']}")
        print(f"a={scenario['a']}, θ1={scenario['theta1']}, θ2={scenario['theta2']}")
        print('='*60)
        
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # Extracting mean trajectory through jax numpty 
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        
        # Plotting mean trajectory
        ax1.plot(mean_traj[:, 0], mean_traj[:, 1], 
                color=cmap(idx), linewidth=2.5, 
                label=scenario['label'], alpha=0.8)
        
        # Plotting final positions as ascatter plot 
        final_states = results['trajectories'][:, -1, :]
        ax2.scatter(final_states[:, 0], final_states[:, 1], 
                   color=cmap(idx), alpha=0.3, s=10,
                   label=scenario['label'] if idx == 0 else "")
    
    # Trajectory plot
    ax1.scatter(base_params['x0'], base_params['y0'], 
               color='green', s=150, marker='o', zorder=5, 
               edgecolor='black', linewidth=2, label='Start')
    ax1.scatter(base_params['target_x'], base_params['target_y'], 
               color='red', s=200, marker='*', zorder=5,
               edgecolor='black', linewidth=2, label='Target')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Mean Trajectories for Different Potential Wells', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Final distribution plot
    ax2.scatter(base_params['target_x'], base_params['target_y'], 
               color='red', s=200, marker='*', zorder=5,
               edgecolor='black', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Final State Distributions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # scatter plot elgend
    handles = []
    for idx, scenario in enumerate(scenarios):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=cmap(idx), markersize=8,
                                 label=scenario['label']))
    ax2.legend(handles=handles, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved as: {save_name}")
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 60)
    print(f"{'Scenario':<20} {'Success Rate':<15} {'Mean Distance':<15} {'Final Loss':<15}")
    print("-" * 60)
    for idx, (scenario, results) in enumerate(zip(scenarios, results_list)):
        print(f"{scenario['label']:<20} "
              f"{results['success_rate']*100:>12.1f}%  "
              f"{results['distances'][-1]:>13.4f}  "
              f"{results['losses'][-1]:>15.6f}")
    
    return results_list

# main function being run 
if __name__ == "__main__":
    base_params = {
        # Initial and target states
        'x0': 0.0, 'y0': 0.0,      # Initial states of X and Y
        'target_x': 1.0,           # Target cell fate X
        'target_y': 0.0,           # Target cell fate Y
        
        # Potential parameters
        'a': 2.0,                  # Well depth parameter
        'theta1': 5.0,             # X-direction bias
        'theta2': -5.0,            # Y-direction bias
        
        # Noise and time parameters
        'D': 0.05,                  # Noise intensity
        'T': 5.0,                  # Time horizon
        'dt': 0.01,                # Time step
        
        # Optimization parameters (using best values)
        'N': 1000,                 # Ensemble size
        'lambda_reg': 0.01,        # Control regularization
        'beta': 1.0,               # Exponential weight parameter
        'chebyshev_order': 25,     # Control basis dimension
        'learning_rate': 0.01,
        'max_epochs': 1000         # Max epochs before forced stop
    }
    
    # Lambda sweep
    print("\n" + "="*60)
    print("LAMBDA PARAMETER SWEEP")
    print("="*60)
    
    lambda_values = np.logspace(-4, 0, 6)  # From 0.0001 to 1.0
    lambda_results = sweep_lambda(
        base_params=base_params,
        lambda_values=lambda_values,
        save_name="lambda_sweep_results.png",
        verbose=False
    )
    
    # Beta sweep
    print("\n" + "="*60)
    print("BETA PARAMETER SWEEP")
    print("="*60)
    
    beta_values = [0.1, 0.5, 1.0, 2.0]
    beta_results = sweep_beta(
        base_params=base_params,
        beta_values=beta_values,
        save_name="beta_sweep_results.png",
        verbose=False
    )
    
    # Potential scenarios that were tested (created some extremes to see what each variable is doing)
    potential_scenarios = [
        {
            'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
            'label': 'Baseline (balanced)'
        },
        {
            'a': 2.0, 'theta1': 3.0, 'theta2': -3.0,
            'label': 'Weaker bias'
        },
        {
            'a': 1.0, 'theta1': 5.0, 'theta2': -5.0,
            'label': 'Shallower wells'
        },
        {
            'a': 3.0, 'theta1': 5.0, 'theta2': -5.0,
            'label': 'Deeper wells'
        },
        {
            'a': 2.0, 'theta1': 7.0, 'theta2': -3.0,
            'label': 'Asymmetric bias'
        },
        {
            'a': 2.0, 'theta1': 0.0, 'theta2': -8.0,
            'label': 'Strong y-bias only'
        }
    ]
    
    # Parameter sweep
    print("\n" + "="*60)
    print("PARAMETER SWEEP: EXPLORING DIFFERENT POTENTIAL WELLS")
    print("="*60)
    
    sweep_results = sweep_potentials(
        base_params=base_params,
        scenarios=potential_scenarios,
        save_name="potential_sweep_comparison.png",
        verbose=False  
    )
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS: BASELINE SCENARIO WITH EXPONENTIAL WEIGHTING")
    print("="*60)
    
    baseline_params = base_params.copy()
    baseline_params.update({
        'a': 2.0,
        'theta1': 5.0,
        'theta2': -5.0,
        'beta': 1.0  # Exponential weight parameter
    })
    
    baseline_results = optimize_cell_fate_control(baseline_params, verbose=True)
    plot_results(baseline_results, baseline_params['target_x'], baseline_params['target_y'], 
                save_name='baseline_detailed_exponential.png')

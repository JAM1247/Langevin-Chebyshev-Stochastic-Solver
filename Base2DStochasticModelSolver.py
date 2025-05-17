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

#  Cost Function with Component Tracking
@functools.partial(jax.jit, static_argnums=(4, 5, 12))
def compute_cost_with_components(coeffs, basis, x0, y0, fx, fy, sigma1, sigma2, dt, 
                                targ_x, targ_y, lam, n_traj, key):

    order = basis.shape[1]
    coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
    u1_vec, u2_vec = get_controls(coeffs_x, coeffs_y, basis)

    traj = simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec,
                             sigma1, sigma2, dt, n_traj, key)

    # Terminal cost to compute distance to target cell fate
    finals = traj[:, -1, :]
    target = jnp.array([targ_x, targ_y])
    j_target = jnp.mean(jnp.sum((finals - target) ** 2, axis=1))
    
    # Regularization cost to understand the control effort
    j_reg = dt * jnp.mean(u1_vec ** 2 + u2_vec ** 2)
    
    total_cost = j_target + lam * j_reg
    
    return total_cost, j_target, j_reg, finals

# Optimizer with convergence checking for different ensemble sizes
def optimize_cell_fate_control_extended(a, theta1, theta2, D, params, initial_coeffs=None, 
                                       verbose=True, convergence_window=50, 
                                       convergence_tol=1e-4, random_seed=42):
    print("\n" + "="*60)
    print("EXTENDED CELL FATE CONTROL OPTIMIZATION")
    print("="*60)
    
    # Initializing the dynamics 
    fx, fy = create_cell_fate_dynamics(a, theta1, theta2)
    
    # Convert noise parameter D into standard deviations
    sigma = jnp.sqrt(2 * D)
    
    if verbose:
        print(f"\nPHYSICS PARAMETERS:")
        print(f"  Potential parameters: a={a}, θ1={theta1}, θ2={theta2}")
        print(f"  Noise intensity: D={D} → σ={sigma:.4f}")
    
    x0, y0 = params['x0'], params['y0']
    T, dt = params['T'], params['dt']
    n_traj = params['N']
    targ_x, targ_y = params['target_x'], params['target_y']
    lam = params['lambda_reg']
    order = params['chebyshev_order']
    lr = params['learning_rate']
    n_epochs = params['n_epochs']
    max_epochs = params.get('max_epochs', 10000) 

    if verbose:
        print(f"\nCONTROL PARAMETERS:")
        print(f"  Initial state: ({x0}, {y0})")
        print(f"  Target state: ({targ_x}, {targ_y})")
        print(f"  Time horizon: T={T}, dt={dt} → {int(T/dt)} time steps")
        print(f"  Ensemble size: {n_traj} cells")
        print(f"  Control basis: Chebyshev order {order} → {2*order} coefficients")
        print(f"  Regularization: λ={lam}")
        print(f"  Learning rate: {lr}")
        print(f"  Initial epochs: {n_epochs}, Max epochs: {max_epochs}")
        print(f"  Convergence criteria: {convergence_tol} over {convergence_window} epochs")
    
    t_vec = jnp.arange(0, T + dt, dt)
    basis = chebyshev_basis(order, t_vec / T)
    
    # Initializing coefficients (it can also be done randomly)
    if initial_coeffs is None:
        if params.get('random_init', False):
            init_key = random.PRNGKey(random_seed)
            coeffs = random.normal(init_key, (2 * order,)) * params.get('init_scale', 0.1)
            if verbose:
                print(f"  Using random initial controls with scale {params.get('init_scale', 0.1)}")
        else:
            coeffs = jnp.zeros(2 * order)
    else:
        coeffs = initial_coeffs

    master = random.PRNGKey(random_seed)
    opt = optax.adam(lr)
    opt_state = opt.init(coeffs)
    losses = []
    terminal_costs = []
    reg_costs = []
    distances = []
    
    if verbose:
        print(f"\nOPTIMIZATION PROGRESS:")
        print("-" * 70)
        print("Epoch  | Total Loss | Terminal Cost | Control Cost | Mean Distance | Time (s)")
        print("-" * 70)

    # Tracking for convergence check
    convergence_losses = []
    converged = False
    start_time = time.time()
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        key = random.fold_in(master, epoch)

        def loss_fn(c):
            total, terminal, reg, finals = compute_cost_with_components(
                c, basis, x0, y0, fx, fy,
                sigma, sigma, dt,
                targ_x, targ_y, lam,
                n_traj, key)
            return total

        loss, grads = value_and_grad(loss_fn)(coeffs)
        
        # Getting the cost components and final states
        total, j_terminal, j_reg, finals = compute_cost_with_components(
            coeffs, basis, x0, y0, fx, fy,
            sigma, sigma, dt,
            targ_x, targ_y, lam,
            n_traj, key)
        
        # Computing mean distance to our target
        target = jnp.array([targ_x, targ_y])
        mean_distance = jnp.mean(jnp.sqrt(jnp.sum((finals - target) ** 2, axis=1)))
        
        updates, opt_state = opt.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        # Track metrics
        losses.append(float(loss))
        terminal_costs.append(float(j_terminal))
        reg_costs.append(float(j_reg))
        distances.append(float(mean_distance))
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(f"{epoch:5d}  | {total:10.6f} | {j_terminal:13.6f} | "
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
        
        # Stop if we've reached specified epochs (for fixed-epoch runs)
        if epoch + 1 >= n_epochs:
            if not params.get('check_convergence', True):
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
        print("-" * 70)
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
                terminal_costs=np.asarray(terminal_costs),
                reg_costs=np.asarray(reg_costs),
                distances=np.asarray(distances),
                t_vec=np.asarray(t_vec),
                a=a, theta1=theta1, theta2=theta2, D=D,
                epochs_completed=len(losses),
                converged=converged,
                optimization_time=total_time,
                success_rate=float(success_rate),
                ensemble_size=n_traj,
                chebyshev_order=order)

# graphs 
def plot_extended_results(results, target_x, target_y, save_name='extended_cell_fate_control.png'):
    fig = plt.figure(figsize=(16, 12))
    
    # Trajectories
    ax1 = fig.add_subplot(3, 3, 1)
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
    ax2.set_ylabel('Control')
    ax2.set_title('Control Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss
    ax3 = fig.add_subplot(3, 3, 3)
    epochs = np.arange(len(results['losses']))
    ax3.semilogy(epochs, results['losses'], label='Total Loss')
    ax3.semilogy(epochs, results['terminal_costs'], label='Terminal Cost', alpha=0.7)
    ax3.semilogy(epochs, results['reg_costs'], label='Regularization Cost', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'Optimization Progress (N={results["ensemble_size"]})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final distribution
    ax4 = fig.add_subplot(3, 3, 4)
    final_states = trajectories[:, -1, :]
    h = ax4.hist2d(final_states[:, 0], final_states[:, 1], bins=20, cmap='Blues')
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
        f"  D = {results['D']}"
    )
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax8.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Plot saved as: {save_name}")
    
    return fig

# Function to run convergence study with different epochs
def run_epoch_convergence_study(base_params, ensemble_sizes=[1000, 10000], save_prefix='epoch_convergence'):
    
    # We are running the optimization with extended epochs until we converge for different ensemble sizes 
    results_dict = {}
    
    for N in ensemble_sizes:
        # Updates parameters for this run
        params = base_params.copy()
        params['N'] = N
        params['max_epochs'] = 3000  # Allowing up to 3000 epochs (arbirtary)
        params['check_convergence'] = True  # Enabling convergence checking
        
        print(f"\nRunning epoch convergence study with ensemble size N = {N}")
        results = optimize_cell_fate_control_extended(
            params['a'], params['theta1'], params['theta2'], params['D'], 
            params, verbose=True, 
            convergence_window=50,  # Checking over 50 epochs
            convergence_tol=1e-4    # 0.01% change threshold is being implemented here
        )
        
        save_name = f"{save_prefix}_N{N}.png"
        plot_extended_results(results, params['target_x'], params['target_y'], save_name)
        results_dict[N] = results
        
    return results_dict

# Function to check convergence with different ensemble sizes
def run_ensemble_size_study(base_params, ensemble_sizes=[100, 500, 1000, 5000, 10000], 
                           fixed_epochs=500, save_prefix='ensemble_convergence'):
    
    
    results_dict = {}
    success_rates = []
    final_losses = []
    mean_distances = []
    optimization_times = []
    
    for N in ensemble_sizes:
        # Updating parameters for this run
        params = base_params.copy()
        params['N'] = N
        params['n_epochs'] = fixed_epochs
        params['check_convergence'] = False  # Disable convergence checking for fixed epochs
        
        print(f"\nRunning ensemble size study with N = {N}")
        results = optimize_cell_fate_control_extended(
            params['a'], params['theta1'], params['theta2'], params['D'], 
            params, verbose=True
        )
        
        # Recording results
        save_name = f"{save_prefix}_N{N}.png"
        plot_extended_results(results, params['target_x'], params['target_y'], save_name)
        
        # Collecting metrics
        results_dict[N] = results
        success_rates.append(results['success_rate'])
        final_losses.append(results['losses'][-1])
        mean_distances.append(results['distances'][-1])
        optimization_times.append(results['optimization_time'])
    
    # Plotting ensemble size comparison
    plt.figure(figsize=(15, 10))
    
    # Success rate
    plt.subplot(2, 2, 1)
    plt.plot(ensemble_sizes, [sr * 100 for sr in success_rates], 'o-')
    plt.xlabel('Ensemble Size')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Ensemble Size')
    plt.grid(True, alpha=0.3)
    
    # Final loss
    plt.subplot(2, 2, 2)
    plt.plot(ensemble_sizes, final_losses, 'o-')
    plt.xlabel('Ensemble Size')
    plt.ylabel('Final Loss')
    plt.title('Final Loss vs Ensemble Size')
    plt.grid(True, alpha=0.3)
    
    # Mean distance
    plt.subplot(2, 2, 3)
    plt.plot(ensemble_sizes, mean_distances, 'o-')
    plt.xlabel('Ensemble Size')
    plt.ylabel('Mean Distance to Target')
    plt.title('Mean Distance vs Ensemble Size')
    plt.grid(True, alpha=0.3)
    
    # Optimization time
    plt.subplot(2, 2, 4)
    plt.plot(ensemble_sizes, optimization_times, 'o-')
    plt.xlabel('Ensemble Size')
    plt.ylabel('Optimization Time (s)')
    plt.title('Computation Cost vs Ensemble Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=300)
    plt.close()
    
    return results_dict

# This function checks the different number of Chebyshev ters
def run_chebyshev_order_study(base_params, orders=[5, 10, 15, 20, 25], 
                             fixed_epochs=500, save_prefix='chebyshev_order'):
    """Run the optimization with different Chebyshev orders and compare performance."""
    
    results_dict = {}
    success_rates = []
    final_losses = []
    mean_distances = []
    control_energies = []
    
    for order in orders:
        # Update parameters for this run
        params = base_params.copy()
        params['chebyshev_order'] = order
        params['n_epochs'] = fixed_epochs
        params['check_convergence'] = False  # Disable convergence checking for fixed epochs
        
        print(f"\nRunning Chebyshev order study with order = {order}")
        results = optimize_cell_fate_control_extended(
            params['a'], params['theta1'], params['theta2'], params['D'], 
            params, verbose=True
        )
        
        # Save results
        save_name = f"{save_prefix}_order{order}.png"
        plot_extended_results(results, params['target_x'], params['target_y'], save_name)
        
        # Compute control energy
        u1, u2 = results['U1'], results['U2']
        control_energy = float(jnp.mean(u1**2 + u2**2))
        
        # Collect metrics
        results_dict[order] = results
        success_rates.append(results['success_rate'])
        final_losses.append(results['losses'][-1])
        mean_distances.append(results['distances'][-1])
        control_energies.append(control_energy)
    
    # Plot Chebyshev order comparison
    plt.figure(figsize=(15, 10))
    
    # Success rate
    plt.subplot(2, 2, 1)
    plt.plot(orders, [sr * 100 for sr in success_rates], 'o-')
    plt.xlabel('Chebyshev Order')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Chebyshev Order')
    plt.grid(True, alpha=0.3)
    
    # Final loss
    plt.subplot(2, 2, 2)
    plt.plot(orders, final_losses, 'o-')
    plt.xlabel('Chebyshev Order')
    plt.ylabel('Final Loss')
    plt.title('Final Loss vs Chebyshev Order')
    plt.grid(True, alpha=0.3)
    
    # Mean distance
    plt.subplot(2, 2, 3)
    plt.plot(orders, mean_distances, 'o-')
    plt.xlabel('Chebyshev Order')
    plt.ylabel('Mean Distance to Target')
    plt.title('Mean Distance vs Chebyshev Order')
    plt.grid(True, alpha=0.3)
    
    # Control energy
    plt.subplot(2, 2, 4)
    plt.plot(orders, control_energies, 'o-')
    plt.xlabel('Chebyshev Order')
    plt.ylabel('Average Control Energy')
    plt.title('Control Energy vs Chebyshev Order')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=300)
    plt.close()
    
    return results_dict

# Checking the performance with different random initializations
def run_random_init_study(base_params, num_trials=5, init_scale=0.1, 
                         fixed_epochs=500, save_prefix='random_init'):
    
    results_dict = {}
    success_rates = []
    final_losses = []
    mean_distances = []
    
    # Base parameters with random initialization
    params = base_params.copy()
    params['random_init'] = True
    params['init_scale'] = init_scale
    params['n_epochs'] = fixed_epochs
    params['check_convergence'] = False  # We are disabling convergence checking for fixed epochs
    
    for trial in range(num_trials):
        print(f"\nRunning random initialization study - Trial {trial+1}/{num_trials}")
        
        # Using a different random seed for each trial
        results = optimize_cell_fate_control_extended(
            params['a'], params['theta1'], params['theta2'], params['D'], 
            params, verbose=True, random_seed=42+trial
        )
        
        # saving results
        save_name = f"{save_prefix}_trial{trial+1}.png"
        plot_extended_results(results, params['target_x'], params['target_y'], save_name)
        
        # data table
        results_dict[trial] = results
        success_rates.append(results['success_rate'])
        final_losses.append(results['losses'][-1])
        mean_distances.append(results['distances'][-1])
    
    # Ploting summary statistics
    plt.figure(figsize=(15, 6))
    
    # Success rates across trials
    plt.subplot(1, 3, 1)
    plt.bar(range(1, num_trials+1), [sr * 100 for sr in success_rates])
    plt.xlabel('Trial')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate across Random Initializations')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Final losses across trials
    plt.subplot(1, 3, 2)
    plt.bar(range(1, num_trials+1), final_losses)
    plt.xlabel('Trial')
    plt.ylabel('Final Loss')
    plt.title('Final Loss across Random Initializations')
    plt.grid(True, alpha=0.3)
    
    # Mean distances across trials
    plt.subplot(1, 3, 3)
    plt.bar(range(1, num_trials+1), mean_distances)
    plt.xlabel('Trial')
    plt.ylabel('Mean Distance to Target')
    plt.title('Mean Distance across Random Initializations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=300)
    plt.close()
    
    # Calculate and print statistical summary
    print("\nRANDOM INITIALIZATION STUDY SUMMARY")
    print("=" * 40)
    print(f"Number of trials: {num_trials}")
    print(f"Initialization scale: {init_scale}")
    print(f"Success rate: Mean = {np.mean(success_rates)*100:.2f}%, Std = {np.std(success_rates)*100:.2f}%")
    print(f"Final loss: Mean = {np.mean(final_losses):.6f}, Std = {np.std(final_losses):.6f}")
    print(f"Mean distance: Mean = {np.mean(mean_distances):.6f}, Std = {np.std(mean_distances):.6f}")
    print("=" * 40)
    
    return results_dict

# Main function to have all of oure xtended analyses be run in once place
def run_extended_cell_fate_analyses():
    # Base parameters
    a = 2.0        # Central well stability
    theta1 = 5.0   # x-bias  
    theta2 = -5.0  # y-bias
    D = 0.2        # Noise intensity
    
    base_params = {
        'x0': 0.0, 'y0': 0.0,      # Start at progenitor state
        'T': 5.0,                  # Time horizon
        'dt': 0.01,                # Time step
        'N': 1000,                 # Default ensemble size
        'target_x': 1.0,           # Target cell fate X
        'target_y': 0.0,           # Target cell fate Y
        'lambda_reg': 0.01,        # Control regularization
        'chebyshev_order': 10,     # Default control basis dimension
        'learning_rate': 0.01,
        'n_epochs': 500,           # Default number of epochs
        'a': a, 'theta1': theta1, 'theta2': theta2, 'D': D
    }
    
    print("\n===== EXTENDED CELL FATE ANALYSES =====")
    
    # Runs with more epochs until convergence
    print("\nEPOCH CONVERGENCE STUDY")
    epoch_results = run_epoch_convergence_study(
        base_params, 
        ensemble_sizes=[1000, 10000],
        save_prefix='epoch_convergence'
    )
    
    # Checking convergence with ensemble size
    print("\nENSEMBLE SIZE CONVERGENCE STUDY")
    ensemble_results = run_ensemble_size_study(
        base_params,
        ensemble_sizes=[100, 500, 1000, 5000, 10000],
        save_prefix='ensemble_convergence'
    )
    
    # Checking the different numbers of Chebyshev terms
    print("\nCHEBYSHEV ORDER STUDY")
    chebyshev_results = run_chebyshev_order_study(
        base_params,
        orders=[5, 10, 15, 20, 25],
        save_prefix='chebyshev_order'
    )
    
    # Checking the different random initializations
    print("\nRANDOM INITIALIZATION STUDY")
    random_init_results = run_random_init_study(
        base_params,
        num_trials=5,
        init_scale=0.1,
        save_prefix='random_init'
    )
    
    print("\n===== ALL ANALYSES COMPLETE =====")
    
    return {
        'epoch_results': epoch_results,
        'ensemble_results': ensemble_results,
        'chebyshev_results': chebyshev_results,
        'random_init_results': random_init_results
    }

# main function being run 
if __name__ == "__main__":
    all_results = run_extended_cell_fate_analyses()
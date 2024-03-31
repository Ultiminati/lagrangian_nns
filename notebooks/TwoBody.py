import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
from functools import partial
import os, sys
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR + "/hyperopt")

@jit
def kinetic_energy(state, m1=1, m2=1):
    q, q_dot = jnp.split(state, 2)
    (x1, x2, y1, y2), (x_dot1, x_dot2, y_dot1, y_dot2) = q, q_dot

    T1 = 0.5 * m1 * ((x_dot1)**2 + (y_dot1)**2)
    T2 = 0.5 * m2 * ((x_dot2)**2 + (y_dot2)**2)
    T = T1 + T2
    return T

@jit
def potential_energy(state, m1=1, m2=1):
    q, q_dot = jnp.split(state, 2)
    (x1, x2, y1, y2), (x_dot1, x_dot2, y_dot1, y_dot2) = q, q_dot
    # For numerical convenience, we choose G = 1 for now
    G = 1
    r = jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    V = -G * m1 * m2 / r
    return V 


def raw_lagrangian_eom(lagrangian, state, t=None):
  q, q_t = jnp.split(state, 2)
  q = q
  q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
          @ (jax.grad(lagrangian, 0)(q, q_t)
             - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
  return jnp.concatenate([q_t, q_tt])


def learned_dynamics(params, nn_forward_fn):
  @jit
  def dynamics(q, q_t):
#     assert q.shape == (2,)
    state = jnp.concatenate([q, q_t])
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return dynamics

# Not sure about where to put the masses
# Also check the parameter "state"
def analytical_fn(state, m1=1, m2=1, t=0,):
    (x1, x2, y1, y2), (x_dot1, x_dot2, y_dot1, y_dot2) = jnp.split(state, 2)
    G = 1
    r = jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    ax_1 = G * m2 / jnp.power(r, 3)*(x2-x1)
    ax_2 = -ax_1*m1/m2
    ay_1 = G * m2 / jnp.power(r, 3)*(y2-y1)
    ay_2 = -ax_2*m1/m2
    return jnp.stack([x_dot1, x_dot2, y_dot1, y_dot2, ax_1, ax_2, ay_1, ay_2])

def get_trajectory_analytic(y0, times, **kwargs):
    return odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)

vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxstep=100), (0, None), 0))
vget_unlimited = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic), (0, None), 0))

def new_get_dataset(rng, samples=1, t_span=[0, 10], fps=100, test_split=0.5, lookahead=1,
                    unlimited_steps=False, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs

    frames = int(fps*(t_span[1]-t_span[0]))
    times = jnp.linspace(t_span[0], t_span[1], frames)
    y0 = jnp.concatenate([
        #There are two coordinates for each body, so we need 4 position and 4 velocity components.
        jax.random.uniform(rng, (samples, 4)),
        jax.random.uniform(rng+1, (samples, 4))*0.1
    ], axis=1)

    if not unlimited_steps:
        y = vget(y0, times)
    else:
        y = vget_unlimited(y0, times)
    
    data['x'] = y[:, :-lookahead]
    data['dx'] = y[:, lookahead:] - data['x']
    data['x'] = jnp.concatenate(data['x'])
    data['dx'] = jnp.concatenate(data['dx'])
    data['t'] = jnp.tile(times[:-lookahead], (samples,))

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data



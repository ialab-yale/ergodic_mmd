import jax
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import jax.debug as deb
import jaxopt
import numpy as onp
import sys


# solver = jaxopt.LBFGS(fun=ridge_reg_objective, maxiter=maxiter)
# res = solver.run(init_params, l2reg=l2reg, X=X, y=y)
jax.config.update('jax_enable_x64', True)
class AugmentedLagrangeSolver(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, max_stepsize=1, c=.1):
        self.def_args = args
        self.loss = loss 
        self._c_def = c
        self.c = c
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self.solution = x0
        self.dual_solution = {'lam' : lam, 'mu' : mu}

        def lagrangian(solution, dual_solution, args, c):
            lam = dual_solution['lam']
            mu  = dual_solution['mu']
            _eq_constr   = eq_constr(solution, args)
            _ineq_constr = ineq_constr(solution, args)
            a = np.sum(lam * _eq_constr + c*0.5 * (_eq_constr)**2)
            b = (1/c)*0.5 * np.sum(np.maximum(0., mu + c*_ineq_constr)**2 - mu**2)
            return loss(solution, args) \
                + a \
                + b

        # self._unc_solver = jaxopt.NonlinearCG(fun=lagrangian, linesearch="zoom", max_stepsize=max_stepsize)#, maxls=1000)
        # self._unc_solver = jaxopt.LBFGS(fun=lagrangian, linesearch="backtracking")
        self._unc_solver = jaxopt.GradientDescent(fun=lagrangian)

        self._solver_state = self._unc_solver.init_state(self.solution, self.dual_solution, args, c)

        val_dldx = jit(value_and_grad(lagrangian))


        # @jit
        def step(solution, solver_state, dual_solution, args, c):
            (solution, solver_state) = self._unc_solver.update(
                params=solution,
                state=solver_state,
                dual_solution=dual_solution,
                args=args,
                c=c
            )
            _val, _dldx   = val_dldx(solution, dual_solution, args, c)
            # dual_solution['lam'] = np.clip(dual_solution['lam'] + c*eq_constr(solution, args),-10000,10000)
            # dual_solution['mu']  = np.clip(np.maximum(0, dual_solution['mu'] + c*ineq_constr(solution, args)), -10000,10000)
            dual_solution['lam'] = dual_solution['lam'] + c*eq_constr(solution, args)
            dual_solution['mu']  = np.maximum(0, dual_solution['mu'] + c*ineq_constr(solution, args))


            return solution, dual_solution, solver_state, _dldx

        self.lagrangian = lagrangian
        self.step = step

    # def reset(self):
    #     for _key in self.avg_sq_grad:
    #         self.avg_sq_grad.update({_key : np.zeros_like(self.avg_sq_grad[_key])})
    #     self.c = self._c_def
    # def get_solution(self):
    #     return self.solution
    #     # return self._unravel(self._flat_solution)
    def update_progress(self, k, _grad_total):
        """Update the progress bar on the same line."""
        sys.stdout.write('\033[1A')  # Move cursor up one line
        sys.stdout.write('\033[K')   # Clear the line
        sys.stdout.write(f"{k}, {_grad_total}\n")
        sys.stdout.flush()

    def solve(self, args=None, max_iter=100000, eps=1e-4, alpha=1.00001):
        print(' ')

        if args is None:
            args = self.def_args

        for k in range(max_iter):
            # self.solution, _val, self.avg_sq_grad = self.step(self.solution, args, self.avg_sq_grad, self.c)
            self.solution, self.dual_solution, self._solver_state, _dldx = self.step(self.solution, self._solver_state, self.dual_solution, args, self.c)
            _grad_total = 0.0
            for _key in _dldx:
                _grad_total = _grad_total + np.linalg.norm(_dldx[_key])
            # print(_grad_total)
            # self.c = np.clip(alpha*self.c, 0, 10000)
            self.c = alpha * self.c
            self.update_progress(k, _grad_total)

            if _grad_total < eps:
                print('done in ', k, ' iterations', _grad_total)
                return
        print('unsuccessful, tol: ', _grad_total)

if __name__=='__main__':
    '''
        Example use case 
    '''

    def f(params, args=None) : 
        x = params['x']
        return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]
    def g(params, args) : 
        x = params['x']
        return np.array([2*x[0]-5*x[1]-2])
    def h(params, args) : 
        x = params['x']
        return x[0] + x[1] -1

    x0 = np.array([.5,-0.3])
    params = {'x' : x0}
    opt = AugmentedLagrangeSolver(params,f,g,h)#, step_size=0.1)
    opt.solve(max_iter=1000)
    sol = opt.solution
    print(f(sol), sol)
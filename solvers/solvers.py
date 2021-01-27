try:
  from solvers.scs import SCSSolver
except:
  print('SCS import failed')
  SCSSolver = None

try:
  from solvers.ecos import ECOSSolver
except:
  print('ECOS import failed')
  ECOSSolver = None

try:
  from solvers.gurobi import GUROBISolver
except:
  print('Gurobi import failed')
  GUROBISolver = None

try:
  from solvers.mosek import MOSEKSolver
except:
  print('MOSEK import failed')
  MOSEKSolver = None

try:
  from solvers.osqp import OSQPSolver
except:
  print('OSQP import failed')
  OSQPSolver = None

try:
  from solvers.qpoases import qpOASESSolver
except:
  print('qpOASES import failed')
  qpOASESSolver = None

try:
  from solvers.cosmo import COSMOSolver
except:
  print('COSMO import failed')
  COSMOSolver = None

try:
  from solvers.qpalm import QPALMSolver
except:
  print('QPALM import failed')
  QPALMSolver = None

SCS = 'SCS'
SCS_AA = 'SCS_AA'
SCS_INDIRECT = 'SCS_INDIRECT'
SCS_INDIRECT_AA = 'SCS_INDIRECT_AA'
COSMO = 'COSMO'
SCS_high = SCS + '_high'
ECOS = 'ECOS'
ECOS_high = ECOS + '_high'
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + '_high'
OSQP = 'OSQP'
OSQP_high = OSQP + '_high'
OSQP_polish = OSQP + '_polish'
OSQP_polish_high = OSQP_polish + '_high'
MOSEK = 'MOSEK'
MOSEK_high = MOSEK + '_high'
qpOASES = 'qpOASES'
QPALM = 'QPALM'

# solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]
# SOLVER_MAP = {solver.name(): solver for solver in solvers}

SOLVER_MAP = {
    OSQP: OSQPSolver,
    OSQP_high: OSQPSolver,
    OSQP_polish: OSQPSolver,
    OSQP_polish_high: OSQPSolver,
    GUROBI: GUROBISolver,
    GUROBI_high: GUROBISolver,
    MOSEK: MOSEKSolver,
    MOSEK_high: MOSEKSolver,
    ECOS: ECOSSolver,
    ECOS_high: ECOSSolver,
    COSMO: COSMOSolver,
    SCS: SCSSolver,
    SCS_AA: SCSSolver,
    SCS_INDIRECT: SCSSolver,
    SCS_INDIRECT_AA: SCSSolver,
    qpOASES: qpOASESSolver,
    QPALM: QPALMSolver
}

time_limit = 1000.  # Seconds
eps_abs_low = 1e-03
eps_rel_low = 1e-04
eps_high = 1e-05
eps_infeas = 1e-12

DEBUG = False
MAX_ITERS = int(1e5)

if DEBUG:
  NORMALIZE = True
  SCALE = 1.0
  ALPHA = 1.0
  ADAPTIVE_SCALING = False
else:
  NORMALIZE = True
  SCALE = 0.1
  ALPHA = 1.5
  ADAPTIVE_SCALING = True

# Solver settings
settings = {
    OSQP: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'polish': False,
        'max_iter': MAX_ITERS,
        'eps_prim_inf': eps_infeas,  # Disable infeas check
        'eps_dual_inf': eps_infeas,
        'rho': SCALE,
        'alpha': ALPHA,
        'scaling': NORMALIZE,
        'adaptive_rho': ADAPTIVE_SCALING,
    },
    QPALM: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_prim_inf': eps_infeas,
        'eps_dual_inf': eps_infeas,
        #'verbose': True
    },
    SCS: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_infeas': eps_infeas,
        'max_iters': MAX_ITERS,
        'acceleration_lookback': 0,
        'acceleration_interval': 50,
        'scale': SCALE,
        'alpha': ALPHA,
        'normalize': NORMALIZE,
        'adaptive_scaling': ADAPTIVE_SCALING,
        'use_indirect': False,
        #'rho_x': 1.,
    },
    SCS_AA: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_infeas': eps_infeas,
        'max_iters': MAX_ITERS,
        'acceleration_lookback': 20,
        'acceleration_interval': 50,
        'scale': SCALE,
        'alpha': ALPHA,
        'normalize': NORMALIZE,
        'adaptive_scaling': ADAPTIVE_SCALING,
        'use_indirect': False,
        #'rho_x': 1.,
    },
    SCS_INDIRECT: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_infeas': eps_infeas,
        'max_iters': MAX_ITERS,
        'acceleration_lookback': 0,
        'acceleration_interval': 50,
        'scale': SCALE,
        'alpha': ALPHA,
        'normalize': NORMALIZE,
        'adaptive_scaling': ADAPTIVE_SCALING,
        'use_indirect': True,
        #'rho_x': 1.,
    },
    SCS_INDIRECT_AA: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_infeas': eps_infeas,
        'max_iters': MAX_ITERS,
        'acceleration_lookback': 20,
        'acceleration_interval': 50,
        'scale': SCALE,
        'alpha': ALPHA,
        'normalize': NORMALIZE,
        'adaptive_scaling': ADAPTIVE_SCALING,
        'use_indirect': True,
        #'rho_x': 1.,
    },
    SCS_high: {
        'eps_abs': eps_high,
        'eps_rel': eps_high,
        'eps_infeas': eps_infeas,
        'max_iters': int(1e09),
        'acceleration_lookback': 0,
    },
    COSMO: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'eps_prim_inf': eps_infeas,
        'eps_dual_inf': eps_infeas,
        'max_iter': MAX_ITERS,
        'rho': SCALE,
        'alpha': ALPHA,
        'check_infeasibility': 100,
        'check_termination': 100,
        'decompose': False,
        'scaling': 1 if NORMALIZE else 0,
        'adaptive_rho': ADAPTIVE_SCALING
    },
    OSQP_high: {
        'eps_abs': eps_high,
        'eps_rel': eps_high,
        'polish': False,
        'max_iter': int(1e09),
        'eps_prim_inf': eps_infeas,  # Disable infeas check
        'eps_dual_inf': eps_infeas
    },
    OSQP_polish: {
        'eps_abs': eps_abs_low,
        'eps_rel': eps_rel_low,
        'polish': True,
        'max_iter': int(1e09),
        'eps_prim_inf': eps_infeas,  # Disable infeas check
        'eps_dual_inf': eps_infeas
    },
    OSQP_polish_high: {
        'eps_abs': eps_high,
        'eps_rel': eps_high,
        'polish': True,
        'max_iter': int(1e09),
        'eps_prim_inf': eps_infeas,  # Disable infeas check
        'eps_dual_inf': eps_infeas
    },
    GUROBI: {
        'TimeLimit': time_limit,
        'FeasibilityTol': eps_abs_low,
        'OptimalityTol': eps_abs_low,
    },
    GUROBI_high: {
        'TimeLimit': time_limit,
        'FeasibilityTol': eps_high,
        'OptimalityTol': eps_high,
    },
    MOSEK: {
        'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':
            eps_abs_low,  # Primal feasibility tolerance
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':
            eps_abs_low,  # Dual feasibility tolerance
    },
    MOSEK_high: {
        'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':
            eps_high,  # Primal feasibility tolerance
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_high,  # Dual feasibility tolerance
    },
    ECOS: {
        'abstol': eps_abs_low,
        'reltol': eps_rel_low
    },
    ECOS_high: {
        'abstol': eps_high,
        'reltol': eps_high
    },
    qpOASES: {}
}

for key in settings:
  settings[key]['verbose'] = False
  #settings[key]['time_limit'] = time_limit

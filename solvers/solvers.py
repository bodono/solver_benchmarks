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

TIME_LIMIT = 1000.  # Seconds
HIGH = '_high'

SCS = 'SCS-3.0'
SCS_ALT = 'SCS_ALT'
SCS_AA1 = 'SCS_AA1'
SCS_AA2 = 'SCS_AA2'
SCS_INDIRECT = 'SCS_INDIRECT'
SCS_INDIRECT_AA1 = 'SCS_INDIRECT_AA1'
SCS_INDIRECT_AA2 = 'SCS_INDIRECT_AA2'
COSMO = 'COSMO'
COSMO_high = COSMO + HIGH
SCS_high = SCS + HIGH
SCS_AA1_high = SCS_AA1 + HIGH
SCS_AA2_high = SCS_AA2 + HIGH
SCS_ALT_high = SCS_ALT + HIGH
SCS_INDIRECT_high = SCS_INDIRECT + HIGH
SCS_INDIRECT_AA1_high = SCS_INDIRECT_AA1 + HIGH
SCS_INDIRECT_AA2_high = SCS_INDIRECT_AA2 + HIGH
ECOS = 'ECOS'
ECOS_high = ECOS + HIGH
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + HIGH
OSQP = 'OSQP'
OSQP_high = OSQP + HIGH
OSQP_polish = OSQP + '_polish'
OSQP_polish_high = OSQP_polish + HIGH
MOSEK = 'MOSEK'
MOSEK_high = MOSEK + HIGH
qpOASES = 'qpOASES'
QPALM = 'QPALM'
QPALM_high = QPALM + HIGH

# solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]
# SOLVER_MAP = {solver.name(): solver for solver in solvers}

SOLVER_MAP_REGULAR = {
    OSQP: OSQPSolver,
    OSQP_polish: OSQPSolver,
    GUROBI: GUROBISolver,
    MOSEK: MOSEKSolver,
    ECOS: ECOSSolver,
    COSMO: COSMOSolver,
    SCS: SCSSolver,
    SCS_ALT: SCSSolver,
    SCS_AA1: SCSSolver,
    SCS_AA2: SCSSolver,
    SCS_INDIRECT: SCSSolver,
    SCS_INDIRECT_AA1: SCSSolver,
    SCS_INDIRECT_AA2: SCSSolver,
    qpOASES: qpOASESSolver,
    QPALM: QPALMSolver
}
SOLVER_MAP_HIGH = {
    # high accuracy (tighter tolerances):
    OSQP_high: OSQPSolver,
    OSQP_polish_high: OSQPSolver,
    COSMO_high: COSMOSolver,
    QPALM_high: COSMOSolver,
    GUROBI_high: GUROBISolver,
    MOSEK_high: MOSEKSolver,
    ECOS_high: ECOSSolver,
    SCS_high: SCSSolver,
    SCS_ALT_high: SCSSolver,
    SCS_AA1_high: SCSSolver,
    SCS_AA2_high: SCSSolver,
    SCS_INDIRECT_high: SCSSolver,
    SCS_INDIRECT_AA1_high: SCSSolver,
    SCS_INDIRECT_AA2_high: SCSSolver,
}

SOLVER_MAP = {**SOLVER_MAP_REGULAR, **SOLVER_MAP_HIGH}


DEBUG = False
MAX_ITERS = int(1e5)

if DEBUG:
  NORMALIZE = False
  SCALE = 1.0
  ALPHA = 1.0
  ADAPTIVE_SCALING = False
else:
  NORMALIZE = True
  SCALE = 0.1
  ALPHA = 1.5
  ADAPTIVE_SCALING = True 

# Solver settings
eps_abs_low = 1e-03
eps_rel_low = 1e-04
eps_abs_high = 1e-05
eps_rel_high = 1e-06
eps_infeas = 1e-04
eps_infeas_high = 1e-05

def get_settings(infeasible=False):
    if infeasible:
        _eps_abs_low = 1e-18
        _eps_rel_low = 1e-18
        _eps_abs_high = 1e-18
        _eps_rel_high = 1e-18
        _eps_infeas = eps_infeas
        _eps_infeas_high = eps_infeas_high
    else:
        _eps_abs_low = eps_abs_low
        _eps_rel_low = eps_rel_low
        _eps_abs_high = eps_abs_high
        _eps_rel_high = eps_rel_high
        _eps_infeas = 1e-18
        _eps_infeas_high = 1e-18

    settings = {
        OSQP: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'polish': False,
            'max_iter': MAX_ITERS,
            'eps_prim_inf': _eps_infeas,  # Disable infeas check
            'eps_dual_inf': _eps_infeas,
            'rho': SCALE,
            'alpha': ALPHA,
            'scaling': NORMALIZE,
            'adaptive_rho': ADAPTIVE_SCALING,
            'time_limit': TIME_LIMIT,
        },
        QPALM: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_prim_inf': _eps_infeas,
            'eps_dual_inf': _eps_infeas,
            #'verbose': True
        },
        SCS: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': 0,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': False,
            'time_limit_secs': TIME_LIMIT,
            # 'write_data_filename': 'LISWET1'
            #'rho_x': 1.,
        },
        SCS_ALT: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': 0,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': False,
            'time_limit_secs': TIME_LIMIT,
            #'rho_x': 1.,
        },
        SCS_AA1: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': 20,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': False,
            'time_limit_secs': TIME_LIMIT,
            #'rho_x': 1.,
        },
        SCS_AA2: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': -20,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': False,
            'time_limit_secs': TIME_LIMIT,
            #'rho_x': 1.,
        },
        SCS_INDIRECT: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': 0,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': True,
            'time_limit_secs': 5 * TIME_LIMIT, # higher time limit for better understanding
            #'rho_x': 1.,
        },
        SCS_INDIRECT_AA1: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': 20,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': True,
            'time_limit_secs': 5 * TIME_LIMIT, # higher time limit for better understanding
            #'rho_x': 1.,
        },
        SCS_INDIRECT_AA2: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_infeas': _eps_infeas,
            'max_iters': MAX_ITERS,
            'acceleration_lookback': -20,
            'acceleration_interval': 50,
            'scale': SCALE,
            'alpha': ALPHA,
            'normalize': NORMALIZE,
            'adaptive_scaling': ADAPTIVE_SCALING,
            'use_indirect': True,
            'time_limit_secs': 5 * TIME_LIMIT, # higher time limit for better understanding
            #'rho_x': 1.,
        },
        COSMO: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'eps_prim_inf': _eps_infeas,
            'eps_dual_inf': _eps_infeas,
            'max_iter': MAX_ITERS,
            'rho': SCALE,
            'alpha': ALPHA,
            'check_infeasibility': 100,
            'check_termination': 100,
            'decompose': False,
            'scaling': 10 if NORMALIZE else 0,
            'adaptive_rho': ADAPTIVE_SCALING,
            'time_limit': TIME_LIMIT
        },
        OSQP_polish: {
            'eps_abs': _eps_abs_low,
            'eps_rel': _eps_rel_low,
            'polish': True,
            'max_iter': int(1e09),
            'eps_prim_inf': _eps_infeas,  # Disable infeas check
            'eps_dual_inf': _eps_infeas
        },
        GUROBI: {
            'TimeLimit':  TIME_LIMIT,
            'FeasibilityTol': _eps_abs_low,
            'OptimalityTol': _eps_abs_low,
        },
        GUROBI_high: {
            'TimeLimit': TIME_LIMIT,
            'FeasibilityTol': _eps_abs_high,
            'OptimalityTol': _eps_abs_high,
        },
        MOSEK: {
            'MSK_DPAR_OPTIMIZER_MAX_TIME': TIME_LIMIT,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS':
                _eps_abs_low,  # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS':
                _eps_abs_low,  # Dual feasibility tolerance
        },
        MOSEK_high: {
            'MSK_DPAR_OPTIMIZER_MAX_TIME': TIME_LIMIT,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS':
                _eps_abs_high,  # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': _eps_abs_high,  # Dual feasibility tolerance
        },
        ECOS: {
            'abstol': _eps_abs_low,
            'reltol': _eps_rel_low
        },
        ECOS_high: {
            'abstol': _eps_abs_high,
            'reltol': _eps_rel_high
        },
        qpOASES: {}
    }

    high_solvers = [OSQP_high, COSMO_high, SCS_high, SCS_ALT_high, SCS_AA1_high,
                    SCS_AA2_high, SCS_INDIRECT_high, SCS_INDIRECT_AA1_high,
                    SCS_INDIRECT_AA2_high, QPALM_high]

    for solver in high_solvers:
        settings[solver] = settings[solver.rstrip(HIGH)].copy()
        settings[solver]['eps_abs'] = _eps_abs_high
        settings[solver]['eps_rel'] = _eps_rel_high
        if solver.startswith('SCS'):
            settings[solver]['eps_infeas'] = _eps_infeas_high
        else:
            settings[solver]['eps_prim_inf'] = _eps_infeas_high
            settings[solver]['eps_dual_inf'] = _eps_infeas_high

    for key in settings:
      settings[key]['verbose'] = False

    return settings

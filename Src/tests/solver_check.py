from pyomo.environ import SolverFactory

def check_solver(solver_name):
    solver = SolverFactory(solver_name)
    return solver.available()

def check_ipopt_linear_solvers():
    if not check_solver('ipopt'):
        print("IPOPT is not available.")
        return

    ipopt = SolverFactory('ipopt')
    linear_solvers = ['ma27', 'ma57', 'ma86', 'ma97', 'mumps', 'pardiso', 'wsmp', 'hsl_ma77', 'hsl_ma86', 'hsl_ma97', 'HSL_MA97', 'HSL_MA86']
    available_solvers = []

    for solver in linear_solvers:
        try:
            ipopt.options['linear_solver'] = solver
            available_solvers.append(solver)
        except:
            pass

    print("Available IPOPT linear solvers:")
    for solver in available_solvers:
        print(f"- {solver}")

# Check IPOPT availability
if check_solver('ipopt'):
    print("IPOPT is available.")
    check_ipopt_linear_solvers()
else:
    print("IPOPT is not available.")

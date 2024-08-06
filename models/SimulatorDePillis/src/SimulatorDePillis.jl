module SimulatorDePillis

using SciMLBase
using DifferentialEquations
using ModelingToolkit

# Create parameter and variable symbolics
# (They are of type `Num`)
@parameters Ab0, r1, r2, r3, r4, k1, k2, start, alpha
@variables t, AB(t), V(t)

# From `ModelingToolkit`: differential operator
D = Differential(t)

# ODEs
eqs = [
    D(V) ~ start * (alpha - ((k1 * V)/(k2 + V))),
    D(AB) ~ start * (r1 * V + r2 * AB * V + AB * (r3 - r4 * AB))
]
# Set of variables defined at `@variables`
vars = [V, AB]::Vector{Num}
# Set of parameters defined at `@parameters`
pars = [Ab0, r1, r2, r3, r4, k1, k2, start, alpha]::Vector{Num}

@named sys = ODESystem(eqs, t, vars, pars)
# sys = structural_simplify(sys)

function simulateDePillis(
    parameters::Vector{Float64},
    x0s::Vector{Float64},
    dosetimes::Vector{Float64},
    t_measurements::Vector{Float64})
    # Mapping from model parameter names to parameters for simulation
    parmap = [
        Ab0 => parameters[1],
        r1 => parameters[2],
        r2 => parameters[3],
        r3 => parameters[4],
        r4 => parameters[5],
        k1 => parameters[6],
        k2 => parameters[7],
        start => 0.0,
        alpha => 0.0]::Vector{Pair{Num, Float64}}

    # Mapping from state variable names to initial values
    x0 = [
        V => x0s[1],
        AB => parameters[1]]::Vector{Pair{Num, Float64}}

    # At the time of `dosetimes` we'll add `dose_amount` to the state
    # variables that change with the dosing
    # start
    affectStart!(integrator) = integrator.p[8] = 1
    cbStart = PresetTimeCallback(dosetimes[1], affectStart!)
    # Vaccine
    # affectV!(integrator) = integrator.u[2] += dose_amount
    # cbV = PresetTimeCallback(dosetimes, affectV!)
    # alpha
    oneDayAfterDose = dosetimes + repeat(
        [1.0]::Vector{Float64}, length(dosetimes))
    affectAlpha!(integrator) = integrator.p[9] = 1
    affectAlpha2!(integrator) = integrator.p[9] = 0
    cbAlpha = PresetTimeCallback(dosetimes, affectAlpha!)
    cbAlpha2 = PresetTimeCallback(oneDayAfterDose, affectAlpha2!)

    # Create callbackset
    cbs = CallbackSet(cbStart, cbAlpha, cbAlpha2)

    # Set timespan to solve the ODE
    tspan = (0, t_measurements[end])

    # Define ODE problem
    problem = ODEProblem(sys, x0, tspan, parmap)
    # Solve using `alg` algorithm with addition of callbacks
    solver = solve(
        problem,
        alg = Tsit5(),
        tstops = vcat(t_measurements, dosetimes),
        callback = cbs;
        verbose = false)

    # Return observed
    return hcat(solver(t_measurements).u...)
end

# Export function
export simulateDePillis

end

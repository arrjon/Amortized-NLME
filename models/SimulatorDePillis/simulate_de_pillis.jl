using Pkg
Pkg.activate("/home/clemens/Documents/phd/projects/amortized_nlme/Monolix_NLME")

cd("/home/clemens/Documents/phd/projects/amortized_nlme/Monolix_NLME/models/julia")

# Load libraries for data frames
using DataFrames
using CSV
# For sampling from distributions
using Random
using Distributions
# Plotting
using Plots

# Load model
include("SimulatorDePillis.jl")

# Load priors
priors = CSV.read("priors_de_pillis.csv", DataFrame)

# Sample many parameters from priors and simulate model to see if we get any failures
nSamples = 10000
parameterIds = unique(priors.parameter)
parameterValues = zeros(nSamples, length(parameterIds))
Threads.@threads for i in eachindex(parameterIds)
    p = parameterIds[i]
    df = priors[priors.parameter .== p, :]
    dist = df.distribution[1]
    mean = df.mean[1]
    var = df.variance[1]
    fixedMean = df.fixed_mean[1]
    fixedVariance = df.fixed_variance[1]
    print("Parameter: $p    Distribution: $dist    Mean: $mean\n")
    if dist == "log-normal"
        mean = log(mean)
    end
    parameterValues[:, i] = exp.(rand(Normal(mean, sqrt(var)), nSamples))
end

# Plot sampled values to see if what I'm doing is correct
i = 7
histogram(log.(parameterValues[:, i]))
xlabel!(parameterIds[i])

# Simulate model for all different parameters
time_measurements = [0.0:0.5:400;]
simulationsAb = zeros(size(parameterValues, 1), length(time_measurements))
simulationsV = zeros(size(parameterValues, 1), length(time_measurements))
Threads.@threads for i in [1:size(parameterValues, 1);]
    p = parameterValues[i, :]
    rdata = SimulatorDePillis.simulateDePillis(
        p,
        [0.0],
        [2.0, 20.0, 250.0],
        time_measurements)
    simulationsAb[i, :] = rdata[2, :]
    simulationsV[i, :] = rdata[1, :]
    # simulations[i, :][simulations[i, :] .< 0] .= 0
end

# simulations
simulations = simulationsAb
indices = rand([1:size(parameterValues, 1);], 100)
p = plot(time_measurements, simulations[1,:])
for i in indices
    # if length(simulations[i, :][simulations[i, :] .< 0.0]) > 0
        # continue
    # else
        p = plot!(
        time_measurements,
        simulations[i,:],
        alpha = 0.5,
        color = "blue"
    )
    print("Max: $(maximum(simulations[i, :]))\n")
    # end
end
# p = plot!()
p = plot!(legend = false)
p = plot!(ylims=(0, 5000))
p = xlabel!("Days post first vaccination")
p = ylabel!("Antibody level")
# display(p)

savefig(p, "./outputs/simulation_de_pillis_model.png")
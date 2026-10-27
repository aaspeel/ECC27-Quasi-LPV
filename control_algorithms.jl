using JuMP
using LinearAlgebra
import HiGHS
import SCS

"""
    synthesize_controller(A_vertices, B, X_vertices, H_U, h_U, ell;
                          alpha=nothing, rho=nothing,
                          objective=:alpha, common_gain=false, p=1,
                          alpha_max=1-1e-6,
                          rho_max=1-1e-6, alpha_weight=1.0,
                          rho_weight=1.0, silent=false)

Solve the finite convex program in the paper. `A_vertices[i]` is an `n × n`
matrix, `B` is `n × m`, the columns of `X_vertices` are the vertices of X,
and U = `{u : H_U*u <= h_U}`. `ell` is a uniform Lipschitz constant of
`h(x, ·)` from the input p-norm to the scheduling 1-norm.

Set `alpha=1` for forward invariance, or fix `alpha` in `[0, 1)` for
stabilization. When minimizing alpha, omit it; its upper bound is `alpha_max`.
The objective can be `:feasibility`, `:alpha`, `:rho`, or `:both`. A rate not
being minimized must be supplied through its keyword; it is then an upper
bound, not an equality. With `:both`, a positive weighted sum of alpha and
rho is minimized. Optimized rates are bounded by `alpha_max` and `rho_max`,
both of which must be below 1 for scheduled synthesis. For common gains,
`alpha_max` may exceed 1; set `alpha_max=Inf` to omit its upper bound.
An optimum alpha above 1 does not certify invariance or stabilization.

`common_gain=true` imposes K_1 = ... = K_N and omits all contraction
constraints. This is the comparison class in the paper; rho is then reported
as zero. `p` accepts 1, 2, or `Inf`. Linear programs use HiGHS; programs
with second-order cone constraints use SCS.

Returns a named tuple with `gains` (a vector of `m × n` matrices, or `nothing`
if no primal solution exists), `alpha`, `rho` (the achieved certificate),
`rho_limit` (the imposed upper bound), `termination_status`,
`primal_status`, `objective_value`, and the JuMP `model`.
"""
function synthesize_controller(
    A_vertices::AbstractVector{<:AbstractMatrix},
    B::AbstractMatrix,
    X_vertices::AbstractMatrix,
    H_U::AbstractMatrix,
    h_U::AbstractVector,
    ell::Real;
    alpha::Union{Nothing,Real}=nothing,
    rho::Union{Nothing,Real}=nothing,
    objective::Symbol=:alpha,
    common_gain::Bool=false,
    p::Real=1,
    alpha_max::Real=1-1e-6,
    rho_max::Real=1-1e-6,
    alpha_weight::Real=1.0,
    rho_weight::Real=1.0,
    silent::Bool=true,
)
    N = length(A_vertices)
    n, M = size(X_vertices)
    n_B, m = size(B)
    n_H, m_H = size(H_U)
    N > 0 || throw(ArgumentError("at least one A vertex is required"))
    n > 0 && M > 0 && m > 0 || throw(ArgumentError("dimensions must be positive"))
    n_B == n || throw(DimensionMismatch("B and X_vertices have different state dimensions"))
    m_H == m || throw(DimensionMismatch("H_U and B have different input dimensions"))
    length(h_U) == n_H || throw(DimensionMismatch("h_U and H_U have different row counts"))
    all(size(A) == (n, n) for A in A_vertices) ||
        throw(DimensionMismatch("each A vertex must be n × n"))
    all(isfinite, B) && all(isfinite, X_vertices) &&
        all(isfinite, H_U) && all(isfinite, h_U) &&
        all(A -> all(isfinite, A), A_vertices) ||
        throw(ArgumentError("plant and set data must be finite"))
    isfinite(ell) && ell >= 0 || throw(ArgumentError("ell must be finite and nonnegative"))
    objective in (:feasibility, :alpha, :rho, :both) ||
        throw(ArgumentError("objective must be :feasibility, :alpha, :rho, or :both"))
    p in (1, 2, Inf) || throw(ArgumentError("p must be 1, 2, or Inf"))
    if common_gain
        alpha_max >= 0 || throw(ArgumentError("alpha_max must be nonnegative (Inf is allowed)"))
    else
        0 <= alpha_max < 1 || throw(ArgumentError("alpha_max must be in [0, 1) for scheduled synthesis"))
    end
    0 <= rho_max < 1 || throw(ArgumentError("rho_max must be in [0, 1)"))
    alpha_weight > 0 && rho_weight > 0 ||
        throw(ArgumentError("objective weights must be positive"))

    minimize_alpha = objective in (:alpha, :both)
    minimize_rho = objective in (:rho, :both)
    if minimize_alpha
        isnothing(alpha) || throw(ArgumentError("omit alpha when minimizing it"))
    else
        isnothing(alpha) && throw(ArgumentError("provide alpha when it is not minimized"))
        0 <= alpha <= 1 || throw(ArgumentError("fixed alpha must be in [0, 1]"))
    end
    if common_gain
        minimize_rho && throw(ArgumentError("minimizing rho is vacuous with a common gain"))
    elseif minimize_rho
        isnothing(rho) || throw(ArgumentError("omit rho when minimizing it"))
    else
        isnothing(rho) && throw(ArgumentError("provide rho when it is not minimized"))
        0 <= rho < 1 || throw(ArgumentError("fixed rho must be in [0, 1)"))
    end

    is_soc = p == 2 && !common_gain && ell > 0 && N > 1
    if is_soc
        # Tighter tolerances make the reported norm certificate useful without
        # relying on SCS's relatively loose defaults.
        model = Model(optimizer_with_attributes(
            SCS.Optimizer,
            "eps_abs" => 1e-8,
            "eps_rel" => 1e-8,
            "max_iters" => 100_000,
        ))
    else
        model = Model(HiGHS.Optimizer)
    end
    silent && set_silent(model)
    @variable(model, K[1:m, 1:n, 1:N])
    @variable(model, q[1:N, 1:M, 1:M] >= 0)

    if minimize_alpha
        @variable(model, alpha_variable >= 0)
        isfinite(alpha_max) && set_upper_bound(alpha_variable, alpha_max)
        alpha_rate = alpha_variable
    else
        alpha_rate = Float64(alpha)
    end
    if common_gain
        rho_rate = 0.0
    elseif minimize_rho
        @variable(model, 0 <= rho_variable <= rho_max)
        rho_rate = rho_variable
    else
        rho_rate = Float64(rho)
    end

    # Input and successor conditions at every (mode, state-vertex) pair.
    for i in 1:N, a in 1:M
        for row in 1:n_H
            @constraint(model,
                sum(H_U[row, k] * K[k, j, i] * X_vertices[j, a]
                    for k in 1:m, j in 1:n) <= h_U[row])
        end
        for row in 1:n
            @constraint(model,
                sum(A_vertices[i][row, j] * X_vertices[j, a] for j in 1:n) +
                sum(B[row, k] * K[k, j, i] * X_vertices[j, a]
                    for k in 1:m, j in 1:n) ==
                sum(q[i, a, b] * X_vertices[row, b] for b in 1:M))
        end
        @constraint(model, sum(q[i, a, b] for b in 1:M) == alpha_rate)
    end

    if common_gain
        for i in 2:N, row in 1:m, col in 1:n
            @constraint(model, K[row, col, i] == K[row, col, 1])
        end
    elseif ell > 0 && N > 1
        pairs = [(i, j) for i in 1:N-1 for j in i+1:N]
        if p == 1
            @variable(model, s[1:length(pairs), 1:M, 1:m] >= 0)
            for (pair_index, (i, j)) in enumerate(pairs), a in 1:M
                for row in 1:m
                    difference = @expression(model,
                        sum((K[row, col, i] - K[row, col, j]) * X_vertices[col, a]
                            for col in 1:n))
                    @constraint(model, -s[pair_index, a, row] <= difference)
                    @constraint(model, difference <= s[pair_index, a, row])
                end
                @constraint(model,
                    (ell / 2) * sum(s[pair_index, a, row] for row in 1:m) <= rho_rate)
            end
        elseif p == Inf
            for (i, j) in pairs, a in 1:M, row in 1:m
                difference = @expression(model,
                    sum((K[row, col, i] - K[row, col, j]) * X_vertices[col, a]
                        for col in 1:n))
                @constraint(model, (ell / 2) * difference <= rho_rate)
                @constraint(model, -(ell / 2) * difference <= rho_rate)
            end
        else # p == 2
            for (i, j) in pairs, a in 1:M
                components = [@expression(model,
                    (ell / 2) * sum((K[row, col, i] - K[row, col, j]) *
                                    X_vertices[col, a] for col in 1:n)) for row in 1:m]
                @constraint(model, [rho_rate; components] in SecondOrderCone())
            end
        end
    end

    if objective == :alpha
        @objective(model, Min, alpha_rate)
    elseif objective == :rho
        @objective(model, Min, rho_rate)
    elseif objective == :both
        @objective(model, Min, alpha_weight * alpha_rate + rho_weight * rho_rate)
    end
    optimize!(model)

    status = termination_status(model)
    primal = primal_status(model)
    if !has_values(model)
        return (
            gains=nothing, alpha=nothing, rho=nothing,
            rho_limit=common_gain ? nothing : (minimize_rho ? Float64(rho_max) : Float64(rho)),
            termination_status=status, primal_status=primal,
            objective_value=nothing, model=model,
        )
    end

    gains = [Matrix{Float64}([value(K[row, col, i]) for row in 1:m, col in 1:n])
             for i in 1:N]
    achieved_rho = common_gain || N == 1 || ell == 0 ? 0.0 :
        (ell / 2) * maximum(norm((gains[i] - gains[j]) * X_vertices[:, a], p)
                            for i in 1:N-1 for j in i+1:N for a in 1:M)
    return (
        gains=gains,
        alpha=minimize_alpha ? value(alpha_rate) : Float64(alpha),
        rho=achieved_rho,
        rho_limit=common_gain ? nothing : (minimize_rho ? value(rho_rate) : Float64(rho)),
        termination_status=status,
        primal_status=primal,
        objective_value=objective_value(model),
        model=model,
    )
end

"""
    banach_iteration(x, h, gains, u0; iterations=50)

Apply `u^(r+1) = sum_i h_i(x, u^(r)) K_i*x` for exactly `iterations`
steps. `x` is a state vector and `h(x,u)` returns the scheduling weights.
For a scalar input, pass scalar `u0` and `h` receives scalars; for a vector
input, pass vector `u0` and `h` receives vectors. The result has `input`,
`history` (including `u0`), and Euclidean `residuals` between iterates.
"""
function banach_iteration(
    x::AbstractVector,
    h::Function,
    gains::AbstractVector{<:AbstractMatrix},
    u0;
    iterations::Integer=50,
)
    iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    N = length(gains)
    N > 0 || throw(ArgumentError("at least one gain is required"))
    m, n = size(first(gains))
    length(x) == n || throw(DimensionMismatch("x and gains have incompatible dimensions"))
    all(size(gain) == (m, n) for gain in gains) ||
        throw(DimensionMismatch("all gains must have the same size"))
    scalar_input = u0 isa Number
    scalar_input && m != 1 &&
        throw(DimensionMismatch("scalar u0 requires a one-dimensional input"))
    !scalar_input && length(u0) != m &&
        throw(DimensionMismatch("u0 and gains have incompatible input dimensions"))
    u = scalar_input ? Float64(u0) : Vector{Float64}(u0)
    history = [scalar_input ? u : copy(u)]
    residuals = Float64[]
    for _ in 1:iterations
        weights = h(x, u)
        length(weights) == N ||
            throw(DimensionMismatch("h must return one weight per gain"))
        all(isfinite, weights) && all(w -> w >= -1e-8, weights) &&
            abs(sum(weights) - 1) <= 1e-8 ||
            throw(ArgumentError("h(x,u) must return simplex weights"))
        next_vector = zeros(Float64, m)
        for i in 1:N
            next_vector .+= weights[i] .* (gains[i] * x)
        end
        next_u = scalar_input ? next_vector[1] : next_vector
        push!(residuals, scalar_input ? abs(next_u - u) : norm(next_u - u))
        u = next_u
        push!(history, scalar_input ? u : copy(u))
    end
    return (input=u, history=history, residuals=residuals)
end

scalar_input(u::Number) = Float64(u)
scalar_input(u::AbstractVector) = Float64(only(u))

"""
    evaluate_feedback_grid(problem, gains; resolution=101, iterations=40)

Approximate the implicit scalar feedback on a uniform grid over the
two-dimensional box defined by `problem.X_vertices`. Each point uses
`iterations` Banach steps starting from zero. Rows of `inputs` correspond
to `x2`, and columns correspond to `x1`.
"""
function evaluate_feedback_grid(problem, gains; resolution=101, iterations=40)
    size(problem.X_vertices, 1) == 2 && size(first(gains), 1) == 1 ||
        throw(ArgumentError("the feedback grid requires two states and one input"))
    resolution >= 2 || throw(ArgumentError("resolution must be at least two"))
    iterations >= 1 || throw(ArgumentError("iterations must be positive"))
    x1 = range(extrema(problem.X_vertices[1, :])...; length=resolution)
    x2 = range(extrema(problem.X_vertices[2, :])...; length=resolution)
    inputs = [scalar_input(banach_iteration([a, b], problem.h, gains, 0.0;
                                            iterations).input)
              for b in x2, a in x1]
    return (; x1, x2, inputs)
end

function plant_step(problem, x, u)
    input = scalar_input(u)
    weights = problem.h(x, input)
    successor = zeros(length(x))
    for i in eachindex(problem.A_vertices)
        successor .+= weights[i] .* (problem.A_vertices[i] * x)
    end
    successor .+= problem.B[:, 1] .* input
    return successor
end

"""
    simulate_closed_loop(problem, gains, x0; horizon=20, iterations=40)

Simulate the actual nonlinear plant.  Each input uses the prescribed
number of Banach iterations, initialized at the previously applied input.
The finite-iteration trajectories are numerical illustrations; the
paper's invariance and stability theorems concern the exact fixed point.
"""
function simulate_closed_loop(problem, gains, x0; horizon=20, iterations=40)
    horizon >= 0 || throw(ArgumentError("horizon must be nonnegative"))
    iterations >= 1 || throw(ArgumentError("iterations must be positive"))
    states = Matrix{Float64}(undef, length(x0), horizon + 1)
    inputs = Vector{Float64}(undef, horizon)
    final_residuals = Vector{Float64}(undef, horizon)
    states[:, 1] = Float64.(x0)
    previous_input = 0.0

    for t in 1:horizon
        x = states[:, t]
        result = banach_iteration(x, problem.h, gains, previous_input;
                                  iterations=iterations)
        input = scalar_input(result.input)
        inputs[t] = input
        weights = problem.h(x, input)
        fixed_point_update = sum(weights[i] * (gains[i] * x)[1]
                                 for i in eachindex(gains))
        final_residuals[t] = abs(input - fixed_point_update)
        states[:, t + 1] = plant_step(problem, x, input)
        previous_input = input
    end
    return (; states, inputs, final_residuals)
end

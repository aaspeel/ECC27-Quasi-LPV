include(joinpath(@__DIR__, "control_algorithms.jl"))
include(joinpath(@__DIR__, "example_problem.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

"""
    generate_example_plots(problem_number=3; output_dir=joinpath(@__DIR__, "figures"),
                           horizon=20, iterations=40)

Synthesize a scheduled and a common-gain controller with JuMP, simulate
the feasible controllers, and save trajectory, Minkowski function, input, Banach
iteration, and implicit-feedback figures. An infeasible common-gain
program is reported and its comparison curves are omitted.
Returns the synthesis results and saved PDF figure paths.
Select problem 1, 2, or 3. Figure filenames start with `problemN_`, where
N is the selected problem number, so different examples do not overwrite each other.
For problem 2, the common-gain program minimizes alpha without an upper bound.
"""
function generate_example_plots(problem_number::Integer=1;
                                output_dir=joinpath(@__DIR__, "figures"),
                                horizon=20, iterations=10)
    constructors = (problem1, problem2, problem3)
    1 <= problem_number <= length(constructors) ||
        throw(ArgumentError("problem_number must be 1, 2, or 3"))
    problem = constructors[problem_number]()
    args = (problem.A_vertices, problem.B, problem.X_vertices,
            problem.H_U, problem.h_U, problem.ell)
    scheduled = synthesize_controller(args...;
        objective=:both, p=1,
        alpha_weight=1.0, rho_weight=0.1)
    common = synthesize_controller(args...;
        objective=:alpha, p=1,
        common_gain=true, alpha_max=problem_number == 2 ? Inf : 1-1e-6)
    scheduled.termination_status == JuMP.MOI.OPTIMAL && !isnothing(scheduled.gains) ||
        error("scheduled synthesis failed: $(scheduled.termination_status)")
    common_feasible = common.termination_status == JuMP.MOI.OPTIMAL &&
                      !isnothing(common.gains)
    common_feasible || common.termination_status == JuMP.MOI.INFEASIBLE ||
        error("common-gain synthesis failed: $(common.termination_status)")

    initial_states = ([1.0, 0.0], [1.0, 1.0], [-1.0, -1.0])
    scheduled_runs = [simulate_closed_loop(problem, scheduled.gains, x0;
                      horizon, iterations) for x0 in initial_states]
    common_runs = common_feasible ?
        [simulate_closed_loop(problem, common.gains, x0;
                              horizon, iterations) for x0 in initial_states] : []
    common_run = common_feasible ? common_runs[3] : nothing
    feedback = evaluate_feedback_grid(problem, scheduled.gains; iterations)

    mkpath(output_dir)
    paths = (
        trajectories=joinpath(output_dir, "problem$(problem_number)_trajectories.pdf"),
        gauge=joinpath(output_dir, "problem$(problem_number)_gauge_decay.pdf"),
        inputs=joinpath(output_dir, "problem$(problem_number)_inputs.pdf"),
        banach=joinpath(output_dir, "problem$(problem_number)_banach_residuals.pdf"),
        feedback=joinpath(output_dir, "problem$(problem_number)_feedback.pdf"),
    )
    save_trajectory_plot(paths.trajectories, scheduled_runs, common_runs)
    save_gauge_plot(paths.gauge, scheduled_runs[3], common_run,
                    scheduled.alpha, common.alpha)
    save_input_plot(paths.inputs, scheduled_runs[3], common_run)
    save_banach_plot(paths.banach, problem, scheduled.gains, scheduled.rho)
    save_feedback_plot(paths.feedback, feedback)

    println("Scheduled controller: alpha=$(scheduled.alpha), rho=$(scheduled.rho)")
    for (i, gain) in enumerate(scheduled.gains)
        println("  K_$i = ", gain)
    end
    if common_feasible
        println("Common gain: alpha=$(common.alpha)")
        println("  K = ", first(common.gains))
    else
        println("Common-gain stabilization is infeasible; comparison curves omitted.")
    end
    println("Maximum final Banach residual: ",
            maximum(vcat((run.final_residuals for run in scheduled_runs)...)))
    println("Problem $problem_number figures saved to ", output_dir)
    return (; problem_number, problem, scheduled, common, paths, scheduled_runs, common_runs)
end

generate_example_plots(1);
generate_example_plots(2);

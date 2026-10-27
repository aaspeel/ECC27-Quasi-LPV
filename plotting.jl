using LinearAlgebra
using LaTeXStrings
using Plots

# Shared font size (points) for text in every figure created by this file.
const FIGURE_FONTSIZE = 14
# Shared base marker size; initial-state stars remain twice the base size.
const FIGURE_MARKERSIZE = 6
default(tickfontsize=FIGURE_FONTSIZE,
        guidefontsize=FIGURE_FONTSIZE,
        legendfontsize=FIGURE_FONTSIZE,
        legendtitlefontsize=FIGURE_FONTSIZE,
        titlefontsize=FIGURE_FONTSIZE,
        plot_titlefontsize=FIGURE_FONTSIZE,
        colorbar_tickfontsize=FIGURE_FONTSIZE,
        colorbar_titlefontsize=FIGURE_FONTSIZE,
        annotationfontsize=FIGURE_FONTSIZE,
        markersize=FIGURE_MARKERSIZE)

# GR caps markers on legend lines independently of the plotted marker size.
# Raise only that cap while exporting these figures, then restore the backend
# setting so other figures (and their legend markers) retain their defaults.
function save_with_larger_legend_markers(fig, path)
    factor = Plots.gr_legend_marker_to_line_factor
    original = factor[]
    try
        factor[] = 2 * original
        savefig(fig, path)
    finally
        factor[] = original
    end
end

function save_trajectory_plot(path, scheduled_runs, common_runs)
    fig = plot([-1.0, 1.0, 1.0, -1.0, -1.0],
               [-1.0, -1.0, 1.0, 1.0, -1.0];
               label=L"\mathcal{X}", color=:black, linestyle=:dash, linewidth=1.5,
               aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2",
               #title=L"\mathrm{Closed\!-\!loop\ trajectories}", legend=:bottomright
               )
    for (j, run) in enumerate(scheduled_runs)
        plot!(fig, run.states[1, :], run.states[2, :];
              label=j == 1 ? L"\mathrm{scheduled}" : false, color=:blue,
              linestyle=:solid, linewidth=2, marker=:circle,
              markercolor=:white, markerstrokecolor=:blue, markersize=FIGURE_MARKERSIZE)
    end
    for (j, run) in enumerate(common_runs)
        plot!(fig, run.states[1, :], run.states[2, :];
              label=j == 1 ? L"\mathrm{common\ gain}" : false, color=:red,
              linestyle=:dash, linewidth=2, marker=:diamond,
              markercolor=:white, markerstrokecolor=:red, markersize=FIGURE_MARKERSIZE)
    end
    initial_states = unique([Tuple(run.states[:, 1])
                             for run in (scheduled_runs..., common_runs...)])
    for (j, (x1, x2)) in enumerate(initial_states)
        scatter!(fig, [x1], [x2]; label=j == 1 ? L"\mathrm{initial\ state}" : false,
                 color=:black, marker=:star5, markersize=2 * FIGURE_MARKERSIZE)
    end
    save_with_larger_legend_markers(fig, path)
end

# Pass nothing for common_run and alpha_c when the common-gain problem is infeasible.
function save_gauge_plot(path, scheduled_run, common_run, alpha_s, alpha_c)
    horizon = size(scheduled_run.states, 2) - 1
    times = 0:horizon
    gauge(run) = [norm(run.states[:, t + 1], Inf) for t in times]
    initial_gauge = norm(scheduled_run.states[:, 1], Inf)
    fig = plot(times, gauge(scheduled_run); label=L"\mathrm{scheduled}",
               color=:blue, linestyle=:solid, linewidth=2,
               marker=:circle, markercolor=:white, markerstrokecolor=:blue,
               markersize=FIGURE_MARKERSIZE,
               xlabel=L"\mathrm{time\ step}\ t",
               ylabel=L"V_{\mathcal{X}}(x(t)) = \left\Vert x(t) \right\Vert_{\infty}",
               #title=L"\mathrm{Minkowski\ function\ decay}"
               )
    plot!(fig, times, [alpha_s^t * initial_gauge for t in times];
          label=L"\mathrm{scheduled:\ exact\!-\!feedback\ bound}", color=:blue,
          linestyle=:dot, linewidth=2)
    if common_run !== nothing
        plot!(fig, times, gauge(common_run); label=L"\mathrm{common\ gain}",
              color=:red, linestyle=:dash, linewidth=2,
              marker=:diamond, markercolor=:white, markerstrokecolor=:red,
              markersize=FIGURE_MARKERSIZE)
        plot!(fig, times, [alpha_c^t * initial_gauge for t in times];
              label=L"\mathrm{common\ gain:\ exact\!-\!feedback\ bound}", color=:red,
              linestyle=:dashdot, linewidth=2)
    end
    save_with_larger_legend_markers(fig, path)
end

# Pass nothing for common_run to omit an unavailable common-gain comparison.
function save_input_plot(path, scheduled_run, common_run)
    horizon = length(scheduled_run.inputs)
    times = 0:(horizon - 1)
    fig = plot(times, scheduled_run.inputs; label=L"\mathrm{scheduled}",
               color=:blue, linewidth=2, marker=:circle,
               xlabel=L"\mathrm{time\ step}\ t", ylabel=L"\mathrm{input}\ u(t)",
               #title=L"\mathrm{Applied\ inputs}", ylim=(-2.2, 2.2)
               )
    if common_run !== nothing
        plot!(fig, times, common_run.inputs; label=L"\mathrm{common\ gain}",
              color=:red, linewidth=2, marker=:diamond)
    end
    hline!(fig, [-2.0, 2.0]; label=L"\mathrm{input\ limits}",
           color=:black, linestyle=:dash)
    savefig(fig, path)
end

function save_feedback_plot(path, feedback)
    # Symmetric limits keep zero at the white midpoint, even for asymmetric inputs.
    magnitude = maximum(abs, feedback.inputs)
    magnitude = iszero(magnitude) ? 1.0 : magnitude
    # The polygon recipe keeps both the cells and the color scale vectorial in PDFs.
    palette = cgrad([:blue, :white, :red], [0.0, 0.5, 1.0])
    style = (; seriestype=:plots_heatmap, linewidth=0.5, legend=false, colorbar=false,
              fillcolor=palette, linecolor=palette,
              clims=(-magnitude, magnitude), grid=false)
    field = plot(feedback.x1, feedback.x2, feedback.inputs; style...,
                  aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2",
                  xlims=extrema(feedback.x1), ylims=extrema(feedback.x2),
                  #title=L"\mathrm{Implicit\ feedback}\ \kappa(x)"
                  )
    levels = range(-magnitude, magnitude; length=257)
    color_scale = plot([0.0, 1.0], levels, repeat(collect(levels), 1, 2); style...,
                       xlims=(0, 1), ylims=(-magnitude, magnitude), xticks=false,
                       ymirror=true, ylabel=L"u = \kappa(x)",
                       framestyle=:box)
    fig = plot(field, color_scale; layout=grid(1, 2; widths=[0.94, 0.06]))
    savefig(fig, path)
end

function save_banach_plot(path, problem, gains, rho)
    x = [1.0, 1.0]
    fig = plot(; yscale=:log10, xlabel=L"\mathrm{Banach\ iteration}\ r",
               ylabel=L"\left\Vert u^{(r+1)} - u^{(r)} \right\Vert",
               #title=L"\mathrm{Fixed\!-\!point\ convergence}"
               )
    for (j, initial_input) in enumerate((-2.0, 2.0))
        result = banach_iteration(x, problem.h, gains, initial_input;
                                  iterations=20)
        differences = [abs(result.history[r + 1] - result.history[r])
                       for r in 1:(length(result.history) - 1)]
        indices = 0:(length(differences) - 1)
        envelope = [differences[1] * rho^r for r in indices]
        color = j == 1 ? :blue : :orange
        plot!(fig, indices, max.(differences, eps(Float64));
              label=L"u^{(0)} = %$(initial_input)", color,
              linewidth=2, marker=:circle)
        plot!(fig, indices, max.(envelope, eps(Float64));
              label=L"\rho\ \mathrm{bound},\ u^{(0)} = %$(initial_input)", color,
              linestyle=:dash, linewidth=2)
    end
    savefig(fig, path)
end

using Plots
using LaTeXStrings

"""Plot the scalar example's implicit policy on X = [-1, 1] and save a PDF."""
function save_closed_form_policy_plot(
    path=joinpath(@__DIR__, "figures", "closed_form_policy.pdf"),
)
    x = range(-1.0, 1.0; length=1001)
    policy(x) = -3x^2 / (1 + sqrt(1 - 3x^3 / 4))
    fig = plot(x, policy.(x);
               color=:blue, linewidth=2, legend=false,
               xlabel=L"x", ylabel=L"u = \tilde{\pi}(x)",
               xlims=(-1, 1), title=L"\mathrm{Closed\!-\!form\ feedback\ policy}")
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

save_closed_form_policy_plot()



"""
    problem1()

Return a two-state, one-input example with a large scheduled/common-gain
performance gap on X = [-1,1]^2 and U = [-2,2].

The optimal box-contraction rates are alpha = 0.2 with scheduled gains
and alpha = 0.98 with a common gain. The second row fixes the lower bound
0.2, while the first rows differ by [1.96, 0], forcing every common gain
to have a worst-case row 1-norm of at least 0.98.

The scheduling map is nonlinear in the input. Its input-to-scheduling
1-norm Lipschitz constant is at most 0.2 on X × U.
"""
function problem1()
    A_vertices = [
        [1.2 0.2; 0.1 0.1],
        [-0.6 0.2; 0.1 0.1],
    ]
    B = reshape([1.0, 0.0], 2, 1)
    X_vertices = [-1.0 -1.0 1.0 1.0;
                  -1.0  1.0 -1.0 1.0]
    H_U = reshape([1.0, -1.0], 2, 1)
    h_U = [2.0, 2.0]

    # ||∂h/∂u||₁ = 0.1|u| sech²(8x₁ + 0.05u²) ≤ 0.2 on U.
    ell = 0.2
    function h(x, u)
        input = u isa Number ? u : only(u)
        weight = 0.5 + 0.5 * tanh(8.0 * x[1] + 0.05 * input^2)
        return [weight, 1.0 - weight]
    end

    return (; A_vertices, B, X_vertices, H_U, h_U, ell, h)
end

"""
    problem2()

Return a two-state, one-input example on X = [-1,1]^2 and U = [-2,2]
with feasible scheduled feedback and infeasible common-gain feedback.

Only the top-left entries differ from problem2(): 1.2 becomes 1.3 and
-0.6 becomes -0.8. On the one-decimal grid, these are a closest pair in
combined Frobenius distance for common-gain infeasibility at alpha = 1.
The first-row separation must exceed 2, so its smallest possible value
is 2.1. Splitting the required increase of 0.3 into changes of 0.1 and
0.2 minimizes the squared distance, 0.1^2 + 0.2^2 = 0.05.

Scheduled gains K1 = [-1.1, -0.2], K2 = [0.6, -0.2] achieve alpha = 0.2
and rho = 0.17. A common gain needs alpha >= 1.05 because the first rows
of the plant vertices differ by [2.1, 0]; even forward invariance fails.

This impossibility also holds for the actual nonlinear plant, beyond the
vertex certificate. For any admissible u = K*x, write K = [k1, k2]. At
x = [1,0] and [-1,0], the inputs are k1 and -k1, so |k1| <= 2. The sum
of the two first successor coordinates is
1.05 * (tanh(8 + 0.05*k1^2) + tanh(8 - 0.05*k1^2)) >= 2.1*tanh(7.8) > 2.
Thus at least one successor leaves X, for every admissible common gain.
"""
function problem2()
    A_vertices = [
        [1.3 0.2; 0.1 0.1],
        [-0.8 0.2; 0.1 0.1],
    ]
    B = reshape([1.0, 0.0], 2, 1)
    X_vertices = [-1.0 -1.0 1.0 1.0;
                  -1.0  1.0 -1.0 1.0]
    H_U = reshape([1.0, -1.0], 2, 1)
    h_U = [2.0, 2.0]

    # ||∂h/∂u||₁ = 0.1|u| sech²(8x₁ + 0.05u²) ≤ 0.2 on U.
    ell = 0.2
    function h(x, u)
        input = u isa Number ? u : only(u)
        weight = 0.5 + 0.5 * tanh(8.0 * x[1] + 0.05 * input^2)
        return [weight, 1.0 - weight]
    end

    return (; A_vertices, B, X_vertices, H_U, h_U, ell, h)
end

"""
    problem3()

Return a two-state, one-input instance of the paper's model.  The input
appears nonlinearly in the scheduling map, and the certified Lipschitz
bound is ell_{1<-1} = 0.32 on X × U.
"""
function problem3()
    A_vertices = [
        [1.0 0.2; 0.1 0.5],
        [-0.4 -0.2; 0.1 0.5],
    ]
    B = reshape([1.0, 0.0], 2, 1)
    X_vertices = [-1.0 -1.0 1.0 1.0;
                  -1.0  1.0 -1.0 1.0]
    H_U = reshape([1.0, -1.0], 2, 1)
    h_U = [2.0, 2.0]
    ell = 0.32

    function h(x, u)
        input = u isa Number ? u : only(u)
        weight = 0.5 + 0.4 * tanh(2.0 * x[1] + 0.1 * input^2)
        return [weight, 1.0 - weight]
    end

    return (; A_vertices, B, X_vertices, H_U, h_U, ell, h)
end
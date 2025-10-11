"""
Toy example for single-horizon three-player (one leader, two followers) Stackelberg game at time t = T.
"""
player_control_list = [
    [1,2],
    [3,4],
    [5,6]
]
function objectives(x, u, Qs, qs, Rs)
    """
    Objective function for the Stackelberg game.
    Each player has a quadratic cost function in both the state and its own control variables, but not others' control variables
    """

    # Compute the objective functions
    J = [0.5 * (x[i]' * Qs[i] * x[i]) + qs[i]' * x[i] + u[i]'* Rs[i] * u[i] for i in 1:3]

    return J
end

function dynamics(x, u, A, B)
    """
    Dynamics of the system.
    The system is linear and time-invariant.

    A is a block-diagonal matrix with the system dynamics for each player, diag(A1, A2, A3) ∈ R^(12x12)
    B is a horizontal concatenation of the input matrices for each player, hcat(B1, B2, B3) ∈ R^(12x6)
    x is a vertical concatenation of the state variables for each player, vcat(x1, x2, x3) ∈ R^(12x1)
    u is a vertical concatenation of the control inputs for each player, vcat(u1, u2, u3) ∈ R^(6x1)
    """

    # Compute the dynamics
    x_next = A * x + B * u

    return x_next
end

function kkt_conditions(Qs, qs, Rs, A, B, x, u)
    for i in 2:3
        # Compute second player and third player KKT conditions
        # (1a) Stationarity condtion (wrt u_T): R1 * u1 + B1' * λ1 + B2' * λ2 + B3' * λ3 = 0
        # (1b) Stationarity condtion (wrt x_{T+1}): -λ1 -λ2 -λ3 + q_{T+1} = 0
        # (2) Dynamics equality condition: x_{T+1} = A * x_T - B1 * u1 - B2 * u2 - B3 * u3 
        
    end
end

# solution_to_KKT = LHS \ RHS
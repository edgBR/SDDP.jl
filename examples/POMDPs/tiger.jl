using SDDP, JuMP, SDDiP, GLPKMathProgInterface, Ipopt

function tigervalue(t, i)

    function initialdynamics(p, noise)
        p1 = (noise[1] == :tiger_left) ? 0.0 : 1.0
        return dynamics((p1, p[2], p[3]), noise[2])
    end

    function dynamics(p, noise)
        tiger, belief_left, belief_right = p
        if !isapprox(belief_left, 1 - belief_right, atol=1e-5)
            error("Belief not sum: $(belief_left), $(1 - belief_right))")
        end

        hearing = if noise == :true_positive
            tiger < 0.5 ? (:left) : (:right)
        else
            tiger > 0.5 ? (:left) : (:right)
        end

        # b(s′) = p(o | s′) * p(s′| b) / p(o | b)
        # s′ = tiger left
        # b  = belief_left
        if hearing == :left
            # o  = hearing  left
            belief_left = 0.85 * belief_left / (0.85 * belief_left + 0.15 * belief_right)
        else
            # o  = hearing  right
            belief_left = 0.15 * belief_left / (0.15 * belief_left + 0.85 * belief_right)
        end

        belief_right = 1 - belief_left
        return tiger, belief_left, belief_right
    end

    TIGERS, TIGERS_prob = [:tiger_left, :tiger_right], [0.5, 0.5]
    GROWLS, GROWLS_prob = [:true_positive, :false_positive], [0.85, 0.15]
    NOISES = if t == 1
        DiscreteDistribution(
            [(tiger, growl) for tiger in TIGERS for growl in GROWLS],
            [tigerp * growlp for tigerp in TIGERS_prob for growlp in GROWLS_prob]
        )
    else
        DiscreteDistribution(GROWLS, GROWLS_prob)
    end
    DynamicPriceInterpolation(
             dynamics = (t == 1) ? initialdynamics : dynamics,
            min_price = (0.0,0.0,0.0),
            max_price = (1.0,1.0,1.0),
        initial_price = (0.5, 0.5, 0.5),
        noise = NOISES,
        lipschitz_constant = 100.0
    )
end

m = SDDPModel(
              sense = :Max,
    objective_bound = 10.0,
             solver = GLPKSolverMIP(),
             stages = 20,
             value_function = tigervalue
                            ) do sp, t
    #=
        POMDP's hack:
            the tiger state is not observable so just set the dual to zero...
    =#
    @constraint(sp, SDDP.valueoracle(sp).mu[2] == 0)

    @binarystate(sp, 0 <= stopgo′     <= 1, stopgo == 1)
    @binarystate(sp, 0 <= open_left′  <= 1, open_left == 0)
    @binarystate(sp, 0 <= open_right′ <= 1, open_right == 0)

    @variables(sp, begin
        listen,     Bin
    end)

    @constraints(sp, begin
        open_left′ + open_right′ + listen == stopgo
        stopgo′ <= 1 - (open_left′ + open_right′)
        stopgo′ <= stopgo
    end)

    function reward(p)
        tiger = p[1] # tiger = 0 => left
        return (
            10 * (
                open_left * tiger +
                open_right * (1-tiger)
            ) -
            100 * (
                open_left * (1-tiger) +
                open_right * tiger
            ) -
            1 * listen
        )
    end
    @stageobjective(sp, reward)

    setSDDiPsolver!(sp, method=LevelMethod(-50.0, quadsolver=IpoptSolver(print_level=0)))
end

solve(m, max_iterations=20)

s = simulate(m, 100, [:open_left′, :open_right′, :stopgo, :price, :listen])
plt = SDDP.newplot()
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:price][t][1],   title="Tiger",        ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:price][t][2],   title="Belief Left",  ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:price][t][3],   title="Belief Right", ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:open_left′][t],  title="Open Left",    ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:open_right′][t], title="Open Right",   ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:stopgo][t],     title="Stop-Go",      ymin=0.0, ymax=1.0)
SDDP.addplot!(plt, 1:100, 1:20, (i,t)->s[i][:listen][t],     title="Listen",       ymin=0.0, ymax=1.0)
SDDP.show(plt)

#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

struct MyRiskMeasure <: SDDP.AbstractRiskMeasure end

function dummy_model(sense, risk_measure)
    SDDPModel(
        sense           = sense,
        stages          = 2,
        objective_bound = 10,
        risk_measure    = risk_measure
        ) do sp, t
        @state(sp, x>=0, x0==0)
        @rhsnoise(sp, w=1:2, x <= w)
        @stageobjective(sp, x)
    end
end

@testset "Risk Measures" begin
    @testset "Constructors" begin
        a = Expectation()
        b = AVaR(0.5)
        c = WorstCase()
        d = 0.5a + 0.3b + 0.2c
        @test d.measures[1] == (0.5, a)
        @test d.measures[2] == (0.3, b)
        @test d.measures[3] == (0.2, c)

        aa = EAVaR(lambda=0.5, beta=0.25)
        @test aa.measures[1] == (0.5, Expectation())
        @test aa.measures[2] == (0.5, AVaR(0.25))
    end

    @testset "Expectation" begin
        measure = Expectation()
        m = dummy_model(:Max, measure)
        y = zeros(4)
        x = [0.1, 0.2, 0.3, 0.4]
        obj = ones(4)
        SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
        @test y == x
    end

    @testset "WorstCase" begin
        @testset ":Max" begin
            measure = WorstCase()
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y == [0.0, 0.0, 1.0, 0.0]
        end
        @testset ":Min" begin
            measure = WorstCase()
            m = dummy_model(:Min, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y == [0.0, 0.0, 0.0, 1.0]
        end
    end

    @testset "AV@R" begin
        @test_throws Exception AVaR(-0.1)
        @test_throws Exception AVaR(1.1)
        @testset "beta=0.2" begin
            measure = AVaR(0.2)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, [1/2, 1/2, 0, 0], atol=1e-6)
        end
        @testset "beta=0" begin
            measure = AVaR(0.0)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, [1.0, 0, 0, 0], atol=1e-6)
        end
        @testset "beta=1" begin
            measure = AVaR(1.0)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, x, atol=1e-6)
        end
    end

    @testset "EAV@R" begin
        @test_throws Exception EAVaR(lambda=1.1)
        @test_throws Exception EAVaR(lambda=-0.1)
        @test_throws Exception EAVaR(beta=1.1)
        @test_throws Exception EAVaR(beta=-0.1)
        @testset "Max - (0.25, 0.2)" begin
            measure = EAVaR(lambda=0.25, beta=0.2)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, 0.25 * x + 0.75 * [1/2, 1/2, 0, 0], atol=1e-6)
        end
        @testset "Min - (0.25, 0.2)" begin
            measure = EAVaR(lambda=0.25, beta=0.2)
            m = dummy_model(:Min, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, 0.25 * x + 0.75 * [0, 0, 0, 1.0], atol=1e-6)
        end
        @testset "Max - (0.5, 0.0)" begin
            measure = EAVaR(lambda=0.5, beta=0.0)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, 0.5 * x + 0.5 * [1.0, 0, 0, 0], atol=1e-6)
        end
        @testset "Max - (0.5, 0.0) 2" begin
            measure = EAVaR(lambda=0.5, beta=0.0)
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.0, 0.2, 0.4, 0.4]
            obj = [1.0,2.0,3.0,4.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test isapprox(y, 0.5 * x + 0.5 * [0.0, 1.0, 0, 0], atol=1e-6)
        end
    end

    @testset "Wasserstein" begin
        @test_throws Exception Wasserstein(-1.0, IpoptSolver(print_level=0))
        @testset ":Max Worst case" begin
            measure = Wasserstein(10.0, IpoptSolver(print_level=0))
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.0, 0.0, 1.0, 0.0] atol=1e-4
        end
        @testset ":Min Worst case" begin
            measure = Wasserstein(10.0, IpoptSolver(print_level=0))
            m = dummy_model(:Min, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.0, 0.0, 0.0, 1.0] atol=1e-4
        end

        @testset ":Max Expectation" begin
            measure = Wasserstein(0.0, IpoptSolver(print_level=0))
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ x atol=1e-4
        end
        @testset ":Min Expectation" begin
            measure = Wasserstein(0.0, IpoptSolver(print_level=0))
            m = dummy_model(:Min, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ x atol=1e-4
        end

        @testset ":Max Intermediate" begin
            measure = Wasserstein(0.1, IpoptSolver(print_level=0))
            m = dummy_model(:Max, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.194, 0.198, 0.416, 0.191] atol=1e-3
        end
        @testset ":Min Intermediate" begin
            measure = Wasserstein(0.1, IpoptSolver(print_level=0))
            m = dummy_model(:Min, measure)
            y = zeros(4)
            x = [0.1, 0.2, 0.3, 0.4]
            obj = [1.1, 1.2, 0.6, 1.3]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.100, 0.207, 0.156, 0.537] atol=1e-3
        end
    end

    @testset "User-defined MyRiskMeasure" begin
        measure = MyRiskMeasure()
        m = dummy_model(:Max, measure)
        y = zeros(4)
        x = [0.1, 0.2, 0.3, 0.4]
        obj = ones(4)
        @test_throws Exception SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
    end

    @testset "DRO" begin
        @testset "PopStd" begin
            x = rand(10)
            @test isapprox(SDDP.popstd(x), std(x, corrected=false), atol=1e-9)
        end
        @testset "Min - R=0.25" begin
            measure = DRO(0.25)
            m = dummy_model(:Min, measure)
            x = fill(0.2, 5)
            y = zeros(5)
            obj = [-2.0, -1.0, -3.0, -4.0, -5.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.279057,0.358114,0.2,0.120943,0.0418861] atol=1e-6
        end
        @testset "Max - R=0.25" begin
            measure = DRO(0.25)
            m = dummy_model(:Max, measure)
            x = fill(0.2, 5)
            y = zeros(5)
            obj = [2.0, 1.0, 3.0, 4.0, 5.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.279057,0.358114,0.2,0.120943,0.0418861] atol=1e-6
        end
        @testset "Min - R=0.4" begin
            measure = DRO(0.4)
            m = dummy_model(:Min, measure)
            x = fill(0.2, 5)
            y = zeros(5)
            obj = [-2.0, -1.0, -3.0, -4.0, -5.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.324162,0.472486,0.175838,0.027514,0.0] atol=1e-6
        end
        @testset "Max - R=0.4" begin
            measure = DRO(0.4)
            m = dummy_model(:Max, measure)
            x = fill(0.2, 5)
            y = zeros(5)
            obj = [2.0, 1.0, 3.0, 4.0, 5.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.324162,0.472486,0.175838,0.027514,0.0] atol=1e-6
        end
        @testset "Min - R=√0.8" begin
            measure = DRO(sqrt(0.8))
            m = dummy_model(:Min, measure)
            x = fill(0.2, 5)
            y = zeros(5)
            obj = [-2.0, -1.0, -3.0, -4.0, -5.0]
            SDDP.modify_probability(measure, y, x, obj, m, SDDP.getsubproblem(m, 1, 1))
            @test y ≈ [0.0,1.0,0.0,0.0,0.0] atol=1e-6
        end
    end

end

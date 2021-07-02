using DerivativeFreeSolvers

# JSO
using ADNLPModels, NLPModels

# stdlib
using LinearAlgebra, Logging, Random, Test

Random.seed!(1998)

function tests()
  methods = [coordinate_search, mads,
             (nlp; kwargs...) -> nelder_mead(nlp, oriented_restart=true; kwargs...),
             (nlp; kwargs...) -> nelder_mead(nlp, oriented_restart=false; kwargs...)]

  @testset "Every method solves basic problems in multiple precisions" begin
    for mtd in methods
      for T in (Float16, Float32, Float64, BigFloat)
        for (f, x₀, x) in [(x->sum(x.^2), T[2.1; 3.2], zeros(T, 2)),
                          (x->(x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(T, 2), ones(T, 2)),
                          (x->x[1]^2 + x[2]^2 + (x[1] + x[2] - 1)^2, zeros(T, 2), ones(T, 2) / 3),
                          (x->(x[1] - T(1e4))^2 + (x[2] - T(2e-4))^2 + (x[1] * x[2] - 2)^2, ones(T, 2), T[1e4; 2e-4])
                          ]

          nlp = ADNLPModel(f, x₀)

          @testset "Method $mtd in $T, function $f" begin
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=x₀, max_eval=10000)
            end
            @test norm(output.solution - x) < max(abs(f(x₀)), 1.0) * max(1e-2, sqrt(eps(T)))
            @test eltype(output.solution) == T
            @test typeof(output.objective) == T
          end
        end
      end
    end
  end

  madsX(nlp; kwargs...) = mads(nlp; extreme=true, kwargs...)
  methods = [mads, madsX]
  @testset "Constrained problems with non-empty interior" begin
    for mtd in methods
      for T in (Float16, Float32, Float64, BigFloat)
        for (nlp, sol) in [(ADNLPModel(x->sum(x.^2), T[2.1; 3.2], -ones(T, 2), 4*ones(T, 2)), zeros(T, 2)),
                          (ADNLPModel(x->sum(x.^2), T[2.1; 3.2], T[0.5; 0.5], 4*ones(T, 2)), T[0.5; 0.5]),
                          (ADNLPModel(x->sum(x.^2), T[2.1; 3.2], T[0.5; -0.5], 4*ones(T, 2)), T[0.5; 0.0]),
                          (ADNLPModel(x->sum(x.^2), T[2.1; 3.2], x->[2x[1] + x[2]], T[1.0], T[Inf]), T[0.4; 0.2]),
                          (ADNLPModel(x->sum(x.^2), T[-0.1; 0.7], zeros(T, 2), T(Inf) * ones(T, 2), x->[x[1]^2 + 2x[2]^2], T[1.0], T[Inf]), T[0.0; sqrt(2)/2]),
                          ]

          @testset "Method $mtd in $T" begin
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=nlp.meta.x0, max_eval=10000)
            end
            @test norm(output.solution - sol) < max(1e-1, sqrt(eps(T)))
            @test output.primal_feas < max(1e-1, sqrt(eps(T)))
            @test eltype(output.solution) == T
            @test typeof(output.objective) == T
          end
        end
      end
    end
  end

  @testset "Testing input" begin
    for mtd in methods
      with_logger(NullLogger()) do
        nlp = ADNLPModel(x -> x[1] + x[2], ones(2))
        output = mtd(nlp, max_eval=1)
        @test output.status == :max_eval
        output = mtd(nlp, max_iter=1)
        @test output.status == :max_iter
        output = mtd(nlp, max_time=0.0, max_eval=typemax(Int))
        @test output.status == :max_time

        nlp = ADNLPModel(x -> x[1]^2 + x[2]^2, ones(2))
        output = mtd(nlp, atol=1e-2, rtol=1e-2)
      end
    end
  end
end

tests()

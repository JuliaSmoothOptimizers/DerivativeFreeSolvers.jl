using DerivativeFreeSolvers

# JSO
using NLPModels

# stdlib
using LinearAlgebra, Logging, Test

function tests()
  methods = [coordinate_search, mads,
             (nlp; kwargs...) -> nelder_mead(nlp, oriented_restart=true; kwargs...),
             (nlp; kwargs...) -> nelder_mead(nlp, oriented_restart=false; kwargs...)]

  @testset "Every method solves basic problems in multiple precisions" begin
    for mtd in methods
      for (f, x₀, x) in [(x->sum(x.^2), [2.1; 3.2], zeros(2)),
                         (x->(x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2), ones(2)),
                         (x->x[1]^2 + x[2]^2 + (x[1] + x[2] - 1)^2, zeros(2), ones(2) / 3),
                         (x->(x[1] - eltype(x)(1e4))^2 + (x[2] - eltype(x)(2e-4))^2 + (x[1] * x[2] - 2)^2, ones(2), [1e4; 2e-4])
                        ]

        nlp = ADNLPModel(f, x₀)

        for T in (Float16, Float32, Float64, BigFloat)
          @testset "Method $mtd in $T" begin
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=T.(x₀), max_eval=10000)
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
      for (nlp, sol) in [(ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=-ones(2), uvar=4*ones(2)), zeros(2)),
                         (ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=[0.5; 0.5], uvar=4*ones(2)), [0.5; 0.5]),
                         (ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=[0.5; -0.5], uvar=4*ones(2)), [0.5; 0.0]),
                         (ADNLPModel(x->sum(x.^2), [2.1; 3.2], c=x->[2x[1] + x[2]], lcon=[1.0], ucon=[Inf]), [0.4; 0.2]),
                         (ADNLPModel(x->sum(x.^2), [-0.1; 0.7], c=x->[x[1]^2 + 2x[2]^2], lcon=[1.0], ucon=[Inf], lvar=zeros(2)), [0.0; sqrt(2)/2]),
                        ]

        for T in (Float16, Float32, Float64, BigFloat)
          @testset "Method $mtd in $T" begin
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=T.(nlp.meta.x0), max_eval=10000)
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
        output = mtd(nlp, max_time=0.0)
        @test output.status == :max_time

        nlp = ADNLPModel(x -> x[1]^2 + x[2]^2, ones(2))
        output = mtd(nlp, atol=1e-2, rtol=1e-2)
      end
    end
  end
end

tests()

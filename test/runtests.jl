using DerivativeFreeSolvers

# JSO
using NLPModels

# stdlib
using LinearAlgebra, Logging, Test

function tests()
  methods = [orthomads]

  @testset "Every method solves basic problems in multiple precisions" begin
    for mtd in methods
      @testset "Method $mtd" begin
        for (f, x₀, x) in [(x->sum(x.^2), [2.1; 3.2], zeros(2)),
                           (x->(x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2), ones(2)),
                           (x->x[1]^2 + x[2]^2 + (x[1] + x[2] - 1)^2, zeros(2), ones(2) / 3),
                           (x->(x[1] - 1e6)^2 + (x[2] - 2 * 1e-6)^2 + (x[1] * x[2] - 2)^2, ones(2), [1e6; 2e-6])
                          ]

          nlp = ADNLPModel(f, x₀)

          for T in (Float16, Float32, Float64, BigFloat)
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=T.(x₀))
            end
            @test norm(output.solution - x) < max(abs(f(x₀)), 1.0) * max(1e-2, sqrt(eps(T)))
          end
        end
      end
    end
  end

  methods = [orthomads]
  @testset "Constrained problems with non-empty interior" begin
    for mtd in methods
      @testset "Method $mtd" begin
        for (nlp, sol) in [(ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=-ones(2), uvar=4*ones(2)), zeros(2)),
                           (ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=[0.5; 0.5], uvar=4*ones(2)), [0.5; 0.5]),
                           (ADNLPModel(x->sum(x.^2), [2.1; 3.2], lvar=[0.5; -0.5], uvar=4*ones(2)), [0.5; 0.0]),
                           (ADNLPModel(x->sum(x.^2), [2.1; 3.2], c=x->[2x[1] + x[2]], lcon=[1.0], ucon=[Inf]), [0.4; 0.2]),
                           (ADNLPModel(x->sum(x.^2), [2.1; 3.2], c=x->[x[1]^2 + 2x[2]^2], lcon=[1.0], ucon=[Inf], lvar=zeros(2)), [0.0; sqrt(2)/2]),
                          ]

          for T in (Float16, Float32, Float64, BigFloat)
            reset!(nlp)
            output = with_logger(NullLogger()) do
              mtd(nlp, x=T.(nlp.meta.x0))
            end
            @test norm(output.solution - sol) < max(1e-2, sqrt(eps(T)))
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

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
end

tests()

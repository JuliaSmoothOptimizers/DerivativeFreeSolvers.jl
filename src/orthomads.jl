using Primes

export orthomads

"""
    orthomads(nlp)

OrthoMADS is implemented following the description of

    Mark A. Abramson, Charles Audet, J. E. Dennis Jr., and Sébastien Le Digabel.
    OrthoMADS: A Deterministic MADS Instance with Orthogonal Directions.
    SIAM Journal on Optimization, 2009, Vol. 20, No. 2, pp. 948-966.
"""
function orthomads(nlp :: AbstractNLPModel;
                   x :: AbstractVector=copy(nlp.meta.x0),
                   atol :: Real=√eps(eltype(x)),
                   rtol :: Real=√eps(eltype(x)),
                   max_eval :: Int=nlp.meta.nvar*200,
                   max_time :: Float64=30.0,
                   max_iter :: Int=-1,
                   stop_with_small_step :: Bool=false,
                   steptol :: Real=0.0,
                  )

  f(x) = obj(nlp, x)
  feasible(x) = if unconstrained(nlp)
    true
  else
    bl, bu, cl, cu = [getfield(nlp.meta, f) for f in [:lvar,:uvar,:lcon,:ucon]]
    all(bl .<= x .<= bu) && (nlp.meta.ncon == 0 || all(cl .<= cons(nlp, x) .<= cu))
  end

  n = nlp.meta.nvar

  if !feasible(x)
    error("Starting point must be feasible")
  end
  fx = f(x)
  ftol = atol + abs(fx) * rtol

  Δ = min(1.0, 0.1*minimum(nlp.meta.uvar - nlp.meta.lvar))

  iter = 0
  start_time = time()
  elapsed_time = 0.0
  tired = neval_obj(nlp) ≥ max_eval ≥ 0 || elapsed_time > max_time || iter > max_iter ≥ 0
  ∂f = 0.0
  satisfied = stop_with_small_step && Δ ≤ steptol

  xt = similar(x)
  ℓ = 0
  t = t₀ = Primes.PRIMES[n]
  q, _ = adjhalton(t₀, n, 0)
  smallestΔ = Δ
  maxt = t
  # Dk = [Hk -Hk]
  # Hk = qᵀq I - 2 qqᵀ
  # Hk eⱼ = qᵀq eⱼ - 2 qᵀeⱼ q = qᵀq eⱼ - 2qⱼ q

  # TODO: Use SolverTools log
  @info ("Using $max_eval f evaluations")
  @info @sprintf("%-5s  %-5s  %10s  %10s  %3s,%-3s  %10s  %s\n", "Feval", "Iter", "f(x)", "Δ", "t", "ℓ", "‖q‖", "status")
  @info @sprintf("%-5d  %5d  %10.4e  %10.4e  %3d,%-3d  %10.4e\n", neval_obj(nlp), 0, fx, Δ, t, ℓ, norm(q))

  while !(satisfied || tired)
    decrease = false
    status = "No decrease"
    besti = 0
    bestf = fx
    ∂f = 0.0
    for s = [1, -1]
      for i = 1:n
        xt .= x .- (2s * Δ * q[i]) .* q
        xt[i] += dot(q, q) * s * Δ
        if feasible(xt)
          ft = f(xt)
          ∂f = max(∂f, abs(ft - fx) / Δ)
          if ft < bestf
            decrease = true
            status = "Decrease at i = $i, s = $s"
            besti = s * i
            bestf = ft
          end
        end
      end
    end
    if decrease
      fx = bestf
      i = abs(besti)
      s = sign(besti)
      x .-= (2s * Δ * q[i]) .* q
      x[i] += dot(q, q) * s * Δ
    end
    @info @sprintf("%-5d  %5d  %10.4e  %10.4e  %3d,%-3d  %10.4e  %s\n", neval_obj(nlp), iter, fx, Δ, t, ℓ, norm(q), status)

    if !decrease
      ℓ += 1
      Δ /= 4
    else
      ℓ -= 1
      Δ *= 4
    end
    iter += 1
    if Δ < smallestΔ
      smallestΔ = Δ
      t = t₀ + ℓ
    else
      t = 1 + maxt
    end
    maxt = max(maxt, t)
    q, _ = adjhalton(t, n, ℓ)

    satisfied = (stop_with_small_step && Δ ≤ steptol) || (!decrease && ∂f ≤ ftol)
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) ≥ max_eval ≥ 0 || elapsed_time > max_time || iter > max_iter ≥ 0
  end

  status = if Δ ≤ steptol
    :small_step
  elseif ∂f ≤ ftol
    :stalled
  elseif tired
    if neval_obj(nlp) ≥ max_eval ≥ 0
      :max_eval
    elseif elapsed_time > max_time
      :max_time
    else
      :max_iter
    end
  else
    :unknown
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=fx,
                               iter=iter, elapsed_time=elapsed_time)
end

# t = ∑ᵣ aᵣ pʳ
function expansion(t :: Int, p :: Int)
  p > 1 || error("base $p not supported")
  t == 0 && return [0]
  E = Int[]
  while t > 0
    push!(E, t % p)
    t = div(t, p)
  end
  return E
end

function halton_utp(t :: Int, p :: Int)
  E = expansion(t, p)
  ne = length(E)
  return sum(E[i] / p^i for i = 1:ne)
end

# uₜ = (uₜₚ₁,…,uₜₚₙ)
# Starts at m-th prime
function halton(t :: Int, n :: Int)
  # Get n primes
  return [halton_utp(t, Primes.PRIMES[i]) for i = 1:n]
end

# qₜℓ
function adjhalton(t :: Int, n :: Int, ℓ :: Int)
  ut = halton(t, n)
  q = 2ut .- 1
  #α = sqrt(2^abs(ℓ)/n) - 0.5
  j = -1
  α = 0.0
  qt = q
  done = false
  isol = 0
  jsol = 0
  while !done
    j += 1
    done = true
    for i = 1:n
      ξ = (j + 0.5) * norm(q) / abs(q[i])
      qt = round.(Int, ξ * q / norm(q), RoundNearestTiesAway)
      qt[i] = (j + 1) * sign(q[i])
      if dot(qt, qt) <= 2^abs(ℓ)
        if α < ξ
          α = ξ
          isol = i
          jsol = j
        end
        done = false
      end
    end
  end
  i, j = isol, jsol
  α = (j + 0.5) * norm(q) / abs(q[i])
  qt = round.(Int, α * q / norm(q), RoundNearestTiesAway)
  qt[i] = (j + 1) * sign(q[i])
  return qt, α
end

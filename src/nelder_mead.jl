export nelder_mead

"""
    nelder_mead(nlp)

This implementation follows the algorithm described in chapter 9 of [1].
[1] Numerical Optimization (Jorge Nocedal and Stephen J. Wright), Springer, 2006.
"""
function nelder_mead(nlp :: AbstractNLPModel;
                     x :: AbstractVector = copy(nlp.meta.x0),
                     vertices :: Vector{<: AbstractVector{<: AbstractFloat}} = Vector{eltype(x)}[],
                     tol :: Real = √eps(eltype(x)),
                     max_time :: Float64 = 30.0,
                     max_eval :: Int = -1,
                     ref :: Real = - one(eltype(x)),
                     exp :: Real = - one(eltype(x)) * 2,
                     ocn :: Real = - one(eltype(x)) / 2,
                     icn :: Real = one(eltype(x)) / 2)

  # Initial simplex
  n = nlp.meta.nvar
  T = eltype(x)
  if n ≥ length(vertices) > 0
    error("Invalid Simplex : The number of initial vertices is less than n + 1")
  elseif n+1 < length(vertices)
    error("Invalid Simplex : The number of initial vertices is greater than n + 1")
  elseif length(vertices) == 0
    vertices = [x, [copy(x) for i = 1:n]...]
    for j in 1:n
      xt = vertices[j + 1]
      xt[j] += xt[j] == 0 ? eps(T)^T(0.25) : T(0.05)
    end
  end

  pairs = [[x, obj(nlp, x)] for x in vertices]
  sort!(pairs, by=x->x[2])

  x_trial = zeros(T, n)
  x_cen = copy(x_trial)

  k = 0
  el_time = 0.0
  start_time = time()
  tired = neval_obj(nlp) > max_eval >= 0 || el_time > max_time
  norm_opt = norm(first(pairs)[1] - last(pairs)[1])
  optimal = norm_opt < tol
  T = eltype(x)
  status =:unknown
  @info log_header([:iter, :f, :nrm], [Int, T, T], hdr_override=Dict(:f => "f(x)", :nrm => "‖x₁ - xₙ₊₁‖"))

  while !(optimal || tired)

    shrink = true
    x_cen .= sum(pairs[i][1] for i in 1:n)/T(n)
    x_trial .= (x_cen * (1 - ref) + ref * last(pairs)[1])
    f_ref = obj(nlp, x_trial)
    f_bver = pairs[1][2]

    @info log_row(Any[k, f_bver, norm_opt])
    # Reflection
    if pairs[1][2] ≤ f_ref < pairs[n][2]
      pairs[n+1][1] .= x_trial
      pairs[n+1][2] = f_ref
      shrink = false
    # Expansion
    elseif f_ref < pairs[1][2]
      x_trial .= (x_cen * (1 - exp) + exp * last(pairs)[1])
      f_exp = obj(nlp,x_trial)
      if f_exp < f_ref
        pairs[n+1][1] .= x_trial
        pairs[n+1][2] = f_exp
      else
        pairs[n+1][1] .= x_cen + ref * (pairs[n+1][1] - x_cen)
        pairs[n+1][2] = f_ref
      end
      shrink = false
      # Contraction
    elseif f_ref ≥ pairs[n][2]
      # Outside
      if pairs[n][2] ≤ f_ref < pairs[n+1][2]
        x_trial .= (x_cen * (1 - ocn) + ocn * pairs[n+1][1])
        f_ocn = obj(nlp, x_trial)
        if f_ocn ≤ f_ref
          pairs[n+1][1] .= x_trial
          pairs[n+1][2] = f_ocn
          shrink = false
        end
        # Inside
      else
        x_trial .= (x_cen * (1 - icn) + icn * pairs[n+1][1])
        f_icn = obj(nlp, x_trial)
        if f_icn < pairs[n+1][2]
          pairs[n+1][1] .= x_trial
          pairs[n+1][2] = f_icn
          shrink = false
        end
      end
    end

    if shrink
      for i = 2:n+1
        x_trial .= (pairs[i][1] + pairs[1][1])/2
        pairs[i][1] .= x_trial
        pairs[i][2] = obj(nlp, x_trial)
      end
    end

    sort!(pairs, by=x->x[2])
    k += 1
    tired = neval_obj(nlp) > max_eval ≥ 0 || el_time > max_time
    norm_opt = norm(first(pairs)[1] - last(pairs)[1])
    optimal =  norm_opt < tol
    el_time = time() - start_time
  end

  if optimal
    status = :acceptable
  elseif tired
    if neval_obj(nlp) > max_eval ≥ 0
      status = :max_eval
    elseif el_time >= max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=pairs[1][1], objective=pairs[1][2],
                               iter = k, elapsed_time = el_time)
end

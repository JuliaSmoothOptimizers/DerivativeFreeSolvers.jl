export coordinate_search

"""
    coordinate_search(nlp)

The coordinate search is a derivative free optimization method.
It minimizes a certain function f by walking on a grid and taking a step towards smaller function values.
When it can no longer find any smaller values, it reduces the grid by a factor of β, repeating the process.
"""
function coordinate_search(nlp :: AbstractNLPModel;
                           x :: AbstractVector=copy(nlp.meta.x0),
                           tol :: Real = min(√eps(eltype(x)), 1e-4),
                           α :: Real = one(eltype(x)),
                           β :: Real = 2 * one(eltype(x)),
                           max_time :: Float64  = 30.0,
                           max_eval :: Int = -1,
                           greedy :: Bool = true)

  α <= 0 && error("Invalid Parameter : α ≤ 0 ")
  β <= 1 && error("Invalid Parameter : β ≤ 1 ")
  k = 0
  el_time = 0.0
  start_time = time()
  tired = neval_obj(nlp) > max_eval >= 0 || el_time > max_time
  optimal = α < tol
  T = eltype(x)
  status = :unknown
  @info log_header([:iter, :f, :α], [Int, T, T], hdr_override=Dict(:f => "f(x)"))

  f = obj(nlp, x)
  best_f = f
  best_i = 0
  while !(optimal || tired)
    @info log_row(Any[k, f, α])
    xt = copy(x)
    success = false
    for i in 1:nlp.meta.nvar, s in [-1, 1]
      xt[i] = x[i] + α * s
      f_xt = obj(nlp, xt)
      if f_xt < best_f
        success = true
        best_f = f_xt
        best_i = i * s
        greedy && break
      end
      xt[i] = x[i]
    end

    if success
      x[abs(best_i)] += sign(best_i) * α
      f = obj(nlp, x)
    else
      α /= β
    end

    k += 1
    el_time = time() - start_time
    tired = neval_obj(nlp) > max_eval >= 0 || el_time > max_time
    optimal = α <= tol
  end

  if optimal
    status = :acceptable
  elseif tired
    if neval_obj(nlp) > max_eval >= 0
      status = :max_eval
      elseif el_time >= max_time
        status = :max_time
      end
  end

  stats = GenericExecutionStats(nlp)
  set_status!(stats, status)
  set_solution!(stats, x)
  set_objective!(stats, f)
  set_iter!(stats, k)
  set_time!(stats, el_time)
  return stats
end

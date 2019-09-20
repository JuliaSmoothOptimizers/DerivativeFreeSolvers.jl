mutable struct MemoNLP <: AbstractNLPModel
  model :: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  min_elm :: Int
  max_elm :: Int
  nelm :: Int
  memo :: Dict{AbstractVector, AbstractFloat}
  freq :: Dict{AbstractVector, Int}
end

function MemoNLP(model :: AbstractNLPModel; max_elm :: Int = 0, min_elm :: Int = 10)
  max_elm != 0 || return model

  x0 = model.meta.x0
  ncon = model.meta.ncon
  lvar = model.meta.lvar
  uvar = model.meta.uvar
  lcon = model.meta.lcon
  ucon = model.meta.ucon
  nelm = 0
  T = eltype(model.meta.x0)
  memo = Dict{AbstractVector{T}, T}()
  freq = Dict{T, Int}()
  meta = NLPModelMeta(model.meta.nvar, x0=x0, ncon = ncon, lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon, nnzj=0, nnzh=0)

  return MemoNLP(model, meta, Counters(), min_elm, max_elm, nelm, memo, freq)
end

function clean_memo!(mnlp :: MemoNLP)

  J = sort(collect(mnlp.freq), by=u->u[2], rev=true)[mnlp.min_elm+1:mnlp.nelm]
  for i in J
    delete!(mnlp.freq, i[1])
    delete!(mnlp.memo, i[1])
  end
  mnlp.nelm = mnlp.min_elm
end

function NLPModels.obj(mnlp :: MemoNLP, x :: AbstractVector)
  xt = copy(x)
  if haskey(mnlp.memo, xt)
    mnlp.freq[xt] += 1
  else
    mnlp.memo[xt] = obj(mnlp.model, xt)
    mnlp.freq[xt] = 1
  end
  xr = mnlp.memo[xt];
  length(mnlp.memo) ≤ mnlp.max_elm || clean_memo!(mnlp)
  mnlp.nelm += 1

  return xr
end

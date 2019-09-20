mutable struct MemoNLP <: AbstractNLPModel
  model :: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  min_elm :: Int
  max_elm :: Int
  memo :: Dict{AbstractVector, AbstractFloat}
  freq :: Dict{AbstractFloat, Int}
end

function MemoNLP(model :: AbstractNLPModel, max_elm :: Int, min_elm :: Int)
  x0 = model.meta.x0
  ncon = model.meta.ncon
  lvar = model.meta.lvar
  uvar = model.meta.uvar
  lcon = model.meta.lcon
  ucon = model.meta.ucon
  T = eltype(model.meta.x0)
  memo = Dict{AbstractVector{T}, T}()
  freq = Dict{T, Int}()
  meta = NLPModelMeta(model.meta.nvar, x0=x0, ncon = ncon, lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon, nnzj=0, nnzh=0)

  return MemoNLP(model, meta, Counters(), min_elm, max_elm, memo, freq)
end

function clean_memo!(mnlp :: MemoNLP)
  if length(mnlp.memo) > mnlp.max_elm
    for i in sort(collect(mnlp.freq), by=x->x[2])[1:mnlp.min_elm]
      delete!(mnlp.memo, i[1])
      delete!(mnlp.freq, i[1])
    end
  end

  return mnlp
end

function NLPModels.obj(mnlp :: MemoNLP, x :: AbstractVector)
  clean_memo!(mnlp)
  if haskey(mnlp.memo, x)
    mnlp.freq[x] += 1
  else
    mnlp.memo[x] = obj(mnlp.model, x)
    mnlp.freq[x] = 1
  end
  return mnlp.memo[x]
end

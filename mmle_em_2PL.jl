"""
guassHermite takes two arguements:
  points (Int64), the number of quadratures desired
  iterlim (Int64), the maximum number of iterations used to optimize the quadrature point estimates
    iterlim defaults to 50 iterations
hermite takes two arguements:
  points (Int64), the number of quadratures desired
  z (Float64), an estimated z-score for a given quadrature node

  hermite produces hermite polynomial weights for a given z score and number of quadrature nodes
"""

function hermite(points, z)
  p1 = 1 / pi^0.4
  p2 = 0
  for j in 1:points
    p3 = copy(p2)
    p2 = copy(p1)
    p1 = z * sqrt(2 / j) * p2 - sqrt((j - 1) / j) * p3
  end
  pp = sqrt(2 * points) * p2
  return [p1 pp]
end

function optimZ(z, points, iterlim)
  for j in 1:iterlim
    z1 = copy(z)
    p = hermite(points, z)
    z = z1 - p[1] / p[2]
    if abs(z - z1) <= 1e-15
      return z, p
    end
  end
  if j == iterlim
    print("iteration limit exceeded")
    return 0
  end
end

function gaussHermite(points, iterlim = 50)

  x = repeat([0.0], points)
  w = repeat([0.0], points)
  m = (points + 1) / 2
  z = 0.0
  for i in 1:m
    i = convert(Int64, i)
    if i == 1
      z = sqrt(2 * points + 1) - 2 * (2 * points + 1)^(-1/6)
    elseif i == 2
      z = z - sqrt(points) / z
    elseif i == 3 || i == 4
      z = 1.9 * z - 0.9 * x[i - 2]
    else
      z = 2 * z - x[i - 2]
    end

    z, p = optimZ(z, points, iterlim)

    x[i] = z
    x[points + 1 - i] = -(x[i])
    w[points + 1 - i] = 2 / p[2]^2
    w[i] = w[points + 1 - i]
  end
  pts = x * sqrt(2)
  wts = w / sum(w)
  r = hcat(pts, wts)

  return r
end

# $$P(θ, τ) = \frac{1}{(1 + e^{-d_j - a_jθ})}$$
# $$Q(θ, τ) = 1 - P(θ, τ)$$

function PX(d, a, x)
    DEV = -(d + a * x)
    EP  = exp(DEV)
    P   = 1 / (1 + EP)
    return(P)
end

# $$L(θ|τ, 𝐮) = \sum_{j = 1}^J P(θ, τ_j)^{u_j} Q(θ, τ_j)^{1 - u_j}$$

function L(D, A, x, yi)
  J = size(D, 1)
  LXK = 1
  for j in 1:J
    P = PX(D[j], A[j], x)
    Q = 1 - P
    LXK *= (P^(yi[j]) * Q^(1 - yi[j]))
  end
  return(LXK)
end

# Posterior Probability
# $$L(θ|τ, 𝐮) $$

function PLk(D, A, X, Y, Ak) # posterior probability Y[n] θ == X[k]
  N = size(Y, 1)
  q = size(Ak, 1)
  sumL = zeros(N, 1)
  LXK = zeros(N, q)
  for n in 1:N
    for k in 1:q
      LXK[n, k] = L(D, A, X[k], Y[n, :])
      sumL[n] += LXK[n, k] * Ak[k]
    end
  end
  return(sumL, LXK)
end

# r̅ and n̅ (line 420)
function rAndnGen(FPT, LXK, Ak, PL, Y)
  N, J = size(Y)
  q = size(Ak, 1)
  RJK = ones(J, q)
  NJK = ones(J, q)
  for j in 1:J
    for k in 1:q
      RJK[j, k] = 0.0 # expected # of correct responses to item j at ability X[k]
      NJK[j, k] = 0.0 # expected # of examinees at ability level X[k]
      for n in 1:N
        NT = FPT[n] * LXK[n, k] * Ak[k] / PL[n] # eq. 3.23
        RT = NT * Y[n, j] # eq. 3.24
        RJK[j, k] += RT
        NJK[j, k] += NT
      end
    end
  end
  return(RJK, NJK)
end

# marginal likelihood of pars, eq. 3.26
function marginalPars(dj, aj, X, rj, nj, iters)
  q = size(X, 1)
  s1, s2, s3, s4, s5, s6 = 0, 0, 0, 0, 0, 0
  djDelta, ajDelta = 0, 0
  for iter in iters
    for k in 1:q
      if nj[k] == 0
        continue
      end
      pExp = rj[k] / nj[k] # line 1060 - 1070
      pjk = PX(dj, aj, X[k])
      wjk = pjk * (1 - pjk)
      if wjk < 9e-7
        continue
      end
      vjk = (pExp - pjk) / wjk
      p1 = nj[k] * wjk
      p2 = p1 * vjk # (rjk/njk) * wjk * vjk
      p3 = p2 * vjk # (rjk/njk) * wjk * vjk^2
      p4 = p1 * X[k]
      p5 = p4 * X[k] # pExp * wjk * X[k]^2
      p6 = p4 * vjk
      s1 = s1 + p1
      s2 = s2 + p2
      s3 = s3 + p4
      s4 = s4 + p6
      s5 = s5 + p5
      s6 = s6 + p3
    end
    DM = s1 * s5 - s3 * s3
    if DM <= 99e-6
      return([aj, dj, ajDelta, djDelta])
    end
    djDelta = (s2 * s5 - s4 * s3) / DM
    ajDelta = (s1 * s4 - s3 * s2) / DM
    dj = dj + djDelta
    aj = aj + ajDelta
    if abs(djDelta) <= .05 && abs(ajDelta) <= .05
      return([aj, dj, ajDelta, djDelta])
    end
  end
  return([aj, dj, ajDelta, djDelta])
end

function eStep(D, A, X, Y, Ak, FPT)
  PL, LXK = PLk(D, A, X, Y, Ak)
  RJK, NJK = rAndnGen(FPT, LXK, Ak, PL, Y)
  return(RJK, NJK)
end

function mStep(D, A, RJK, NJK, X, Ak, iters = 30)
  J = size(RJK, 1)
  pars = zeros(J, 4)
  for j in 1:J
    pars[j, :] = marginalPars(D[j], A[j], X, RJK[j, :], NJK[j, :], iters)
  end
  return(pars)
end

function twoPLEst(D, A, X, Y, Ak, FPT, maxiter = 100)
  i = 0
  ajDeltaMax = 1
  djDeltaMax = 1
  ajDelta = 0
  djDelta = 0
  while i <= maxiter &&  (djDeltaMax > 0.005 || ajDeltaMax > 0.005)
    RJK, NJK = eStep(D, A, X, Y, Ak, FPT)
    mStepPars = mStep(D, A, RJK, NJK, X, Ak)
    A = mStepPars[:, 1]
    D = mStepPars[:, 2]
    ajDelta = mStepPars[:, 3]
    djDelta = mStepPars[:, 4]
    ajDeltaMax = maximum(abs.(ajDelta))
    djDeltaMax = maximum(abs.(djDelta))
    i += 1
    println("iteration = ", i)
  end
  pars = [A D]
  delta = [ajDelta djDelta]
  return(pars, delta)
end

function expandResponses(FPT, Y)
  s = sum(FPT)
  Yfull = zeros(s, size(Y, 2))
  YFullInd = 0
  for i in 1:size(Y, 1)
    YfullRange = (YFullInd + 1):(YFullInd + FPT[i])
    Yfull[YfullRange, :] = repeat(Y[i, :]', FPT[i])
    YFullInd += FPT[i]
  end
  return(Yfull)
end

function checkUniqueList(yi, list)
  for i in 1:size(list, 1)
    if yi == list[i, :]'
      return(i)
    end
  end
  return("unique")
end

function uniqueResponses(Y)
  N, J = size(Y)
  list = Y[1, :]'
  listCount = [1]
  Yindex = Array{Int64, 1}(undef, N)
  Yindex[1, ] = 1
  for i in 2:N
    yi = Y[i, :]'
    present = checkUniqueList(yi, list)
    if present == "unique"
      list = vcat(list, yi)
      listCount = vcat(listCount, 1)
      Yindex[i, ] = size(listCount, 1)
    else
      listCount[present] += 1
      Yindex[i, ] = present
    end
  end
  return(list, listCount, Yindex)
end


function julIRT(data)
  U, Un, Yind = uniqueResponses(data)
  N, J = size(data)
  d = zeros(J)
  a = ones(J)
  gh = gaussHermite(20)
  results = twoPLEst(d, a, gh[:, 1], U, gh[:, 2], Un)
  return(results)
end

function eapEst(D, A, data, qpts)
  theta = -6:(12/qpts):6
  S = size(theta, 1)
  U, Un, Yind = uniqueResponses(data)
  n, J = size(U)
  Lnx = Array{Float64, 2}(undef, n, S)
  for i in 1:n
    for x in 1:qpts
      Lnx[i, x] = L(D, A, theta[x], U[i, :])
    end
  end
  Pr_theta_mat = ones(n, 1) * pdf.(Normal(0, 1), theta)'
  theta_mat = ones(n, 1) * theta'
  top = theta_mat .* Pr_theta_mat .* Lnx
  bottom = (Pr_theta_mat .* Lnx) * ones(S, 1) * ones(1, S)
  EAP = ((top ./ bottom) * ones(S, 1))[Yind, :]
  return(EAP)
end

using Plots
using  Distributions
using Random

Random.seed!(123)
J = 20
N = 5000
alpha = rand(LogNormal(0.33, 0.22), J)
beta  = rand(Normal(0, 1), J)
theta = rand(Normal(0, 1), N)
Z = theta * alpha' .- (ones(N, 1) * (beta .* alpha)')
P = 1 ./ (1 .+ exp.(-Z))
Y = Array{Int64, 2}(undef, N, J)
for i in 1:N
  for j in 1:J
    p = P[i, j]
    Y[i, j] = rand(Binomial(1, p), 1)[1]
  end
end

results = julIRT(Y)
D = results[1][:, 2]
A = results[1][:, 1]
EAP = eapEst(D, A, Y, 60)
hcat(D, -beta./alpha)
hcat(theta, EAP)

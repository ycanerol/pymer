## Which method to choose for scipy.optimize.minimize

Newton-CG can fail unexpectedly for simulation stuff.

BFGS seems to produce some artifacts, but it stable.

CG is a reasonable choice, no artifact, no unexpected fails.

## Where you begin matters

If the starting point is chosen randomly, it might work against you.
Try to choose a reasonable initial guess, STA when available.

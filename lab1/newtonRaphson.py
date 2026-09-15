import numpy as np

def steffensen(f, x0, tol):

    max_its = 15
    xVals = np.zeros(max_its + 1)

    xVals[0] = x0

    for k in range(max_its):

        fk = f(xVals[k])

        xVals[k+1] = xVals[k] - fk**2 / (f(xVals[k] + fk) - fk)

        stopCond = abs(f(xVals[k+1]))

        if stopCond < tol:
            xVals = xVals[0:k+2]
            iter = k+2
            break

    return xVals, iter
-- slope: accept two x and y coords and return slope
slope :: (Fractional a) => (a, a) -> (a, a) -> a
slope (x1, y1) (x2, y2) = (y2 - y1) / (x2 - x1)

-- findIntercept: substitute in slope and attempt to find y-intercept
findIntercept :: (Fractional a) => a -> a -> a -> a
findIntercept slope x y = slope * x - y

-- slopeAndIntercept: take in two coords and return a tuple: (slope, y-intercept)
slopeAndIntercept :: (Fractional a) => (a, a) -> (a, a) -> (a (a, a))

slopeAndIntercept xy1 xy2 =
    let returnedSlope = slope (x1, y1) (x1, x2)
    in (returnedSlope, findIntercept returnedSlope x1 y1)

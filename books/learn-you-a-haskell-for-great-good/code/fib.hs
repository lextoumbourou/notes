fib :: (Num a, Ord a) => a -> a
fib x
    | x < 0 = error "That ain't part of the fib seq."
    | x < 2 = x
    | otherwise = fib (x - 1) + fib (x - 2)

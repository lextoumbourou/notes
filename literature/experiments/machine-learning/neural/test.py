from . import initialize_params, initialize_params_deeps


def test_initialize_params():
    """Init params returns weights and bias terms."""
    feature_units = 3
    hidden_layers = 2
    output_units = 2

    params = initialize_params(feature_units, hidden_layers, output_units)

    assert(params['W1'].shape == (hidden_layers, feature_units))
    assert(params['b1'].shape == (hidden_layers, 1))

    assert(params['W2'].shape == (output_units, hidden_layers))
    assert(params['b2'].shape == (output_units, 1))


def test_initialize_params_deep():
    """Init params deep."""
    layer_dimensions = [5, 4, 3]

    params = initialize_params_deeps(layer_dimensions)

    assert(params['W1'].shape == (4, 5))
    assert(params['W2'].shape == (3, 4))


def test_linear_forward():
    """Linear forward."""
    X = np.array([3., 2.])
    W = np.array([2., 2.])
    b = np.array([1., 1.])

    Z, cache = linear_forward(X, W, b)

    assert(Z == np.array([6., 4.]))

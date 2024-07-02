import torch
import numpy as np

EPSILON = 0.0000000001

def _cross_squared_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """
     Pairwise squared distance between two (batch) matrices' rows (2nd dim).
     Computes the pairwise distances between rows of x and rows of y

     :param x            : 3-D float Tensor
         :shape: (batch, n, d)
     :param y            : 3-D float Tensor
         :shape: (batch, m, d)
   
     :return             : squared_dists
         :shape: (batch, n, m)

     squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
  """
  x_norm_squared = torch.sum(torch.square(x), 2)
  y_norm_squared = torch.sum(torch.square(y), 2)

  # Expand so that we can broadcast.
  x_norm_squared_tile = torch.unsqueeze(x_norm_squared, 2)
  y_norm_squared_tile = torch.unsqueeze(y_norm_squared, 1)

  x_y_transpose = torch.matmul(x, torch.conj(torch.transpose(y,1,2)))

  # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile

  return squared_dists

def _pairwise_squared_distance_matrix(x: torch.Tensor) -> torch.Tensor:
  """
  Pairwise squared distance among a (batch) matrix's rows (2nd dim).
  This saves a bit of computation vs. using _cross_squared_distance_matrix(x,x)

     :param x            : 3-D float Tensor
         :shape: (batch, n, d)
     :return             : squared_dists
         :shape: (batch, n, n)

    squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2
  """

  x_x_transpose = torch.matmul(x, torch.conj(torch.transpose(x,1,2)))
  x_norm_squared = torch.diagonal(x_x_transpose,0,2)
  x_norm_squared_tile = torch.unsqueeze(x_norm_squared, 2)

  # squared_dists[b,i,j] = ||x_bi - x_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_x_transpose + torch.transpose(x_norm_squared_tile, 1,2)

  return squared_dists

def _phi(r: torch.Tensor, order: int) -> torch.Tensor:
    """
    Coordinate-wise nonlinearity used to define the order of the interpolation.
    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.

    :param r            : input Tensor
        :shape: any
    :param order        : int

    :return             : Tensor phi_k evaluated coordinate-wise on r, for k = r
        :shape: same shape with input
    """
    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.maximum(r, torch.tensor(EPSILON))
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.maximum(r, torch.tensor(EPSILON)))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(
            torch.maximum(r, torch.tensor(EPSILON)))
    elif order % 2 == 0:
        r = torch.maximum(r, torch.tensor(EPSILON))
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.maximum(r, torch.tensor(EPSILON))
        return torch.pow(r, 0.5 * order)


def _solve_interpolation(device, train_points: torch.Tensor, train_values: torch.Tensor, order: int, regularization_weight: float) -> torch.Tensor:
    """
    Solve for interpolation coefficients.
    Computes the coefficients of the polyharmonic interpolant for the 'training'
    data defined by (train_points, train_values) using the kernel phi

    :param train_points                     : 3-D float Tensor
        :shape: (b, n, d)
    :param train_values                     : 3-D float Tensor
        :shape: (b, n, k)
    :param order                            : order of the interpolation
    :param regularization_weight            : weight to place on smoothness regularization term

    :return  w, v           
        :shape: (b, n, k)
        :shape: (b, d, k)
    """
    b, n, d = list(train_points.size())
    _, _, k = list(train_values.size())
    
    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.
    c = train_points
    f = train_values

    # Next, construct the linear system
    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order).to(device)  # [b, n, n]

    if regularization_weight > 0:
        batch_identity_matrix = np.expand_dims(np.eye(n), 0)
        batch_identity_matrix = torch.tensor(batch_identity_matrix, dtype=train_points.dtype).to(device)

        matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones([b, n, 1], dtype = train_points.dtype).to(device)
    matrix_b = torch.concat([c, ones], 2).to(device)  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = torch.concat([matrix_a, torch.transpose(matrix_b, 2, 1)], 1).to(device)

    num_b_cols = matrix_b.size()[2]  # d + 1
    lhs_zeros = torch.zeros([b, num_b_cols, num_b_cols], dtype = train_points.dtype).to(device)
    right_block = torch.concat([matrix_b, lhs_zeros], 1)  # [b, n + d + 1, d + 1]
    lhs = torch.concat([left_block, right_block], 2).to(device)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros([b, d + 1, k], dtype = train_points.dtype).to(device)
    rhs = torch.concat([f, rhs_zeros], 1).to(device)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    #print("lhs", lhs.shape)
    # lhs = torch.clamp(lhs)
    #print("rhs", rhs.shape)
    # w_v = torch.linalg.lstsq(lhs, rhs).solution
    w_v = torch.linalg.solve(lhs, rhs)
    #print("w_v", w_v)
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]

    return w, v

def _apply_interpolation(device, query_points: torch.Tensor, train_points: torch.Tensor, w: torch.Tensor, v: torch.Tensor, order: int) -> torch.Tensor:
    """
    Apply polyharmonic interpolation model to data.
    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.

    :param query_points                     : 3-D float Tensor
        :shape: (b, m, d)
    :param train_points                     : 3-D float Tensor
        :shape: (b, n, d)
    :param w                                : weights on each interpolation center
        :shape: (b, n, k)
    :param v                                : weights on each input dimension
        :shape: (b, d, k)
    :param order                            : order of the interpolation

    :return                                 : Polyharmonic interpolation evaluated at points defined in query_points.
        :shape: (b, m, k)

    """

    batch_size = train_points.size()[0]
    num_query_points = query_points.size()[1]

    # First, compute the contribution from the rbf term.
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points).to(device)
    phi_pairwise_dists = _phi(pairwise_dists, order).to(device)

    rbf_term = torch.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    query_points_pad = torch.concat([query_points, 
                                    torch.ones([batch_size, num_query_points, 1], dtype = train_points.dtype).to(device)], 2).to(device)
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term
    
def interpolate_spline(device, train_points: torch.Tensor, train_values: torch.Tensor, query_points: torch.Tensor, order: int, regularization_weight: float = 0.0) -> torch.Tensor:
    """
    Interpolate signal using polyharmonic interpolation

    :param train_points                     : 3-D float Tensor
        :shape: (batch_size, n, d)
    :param train_values                     : 3-D float Tensor
        :shape: (batch_size, n, k)
    :param query_points                     : 3-D float Tensor
        :shape: (batch_size, m, d)
    :param order                            : order of the interpolation

    :param regularization_weight            : weight placed on the regularization term

    :return                                 : query_values
        :shape: [batch_size, m, k]

    """
    train_points = torch.as_tensor(train_points).to(device)
    train_values = torch.as_tensor(train_values).to(device)
    query_points = torch.as_tensor(query_points).to(device)

    # First, fit the spline to the observed data.
    w, v = _solve_interpolation(device, train_points, train_values, order, regularization_weight)

    # Then, evaluate the spline at the query locations.
    query_values = _apply_interpolation(device, query_points, train_points, w, v, order)

    return query_values

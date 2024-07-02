import torch


def _interpolate_bilinear(device, grid: torch.Tensor, query_points: torch.Tensor, indexing: str = 'ij') -> torch.Tensor:
    """
    Finds values for query points on a grid using bilinear interpolation.

    :param grid         : 4-D float Tensor
        :shape: (batch, height, width, channels)
    :param query points : 3-D float Tensor
        :shape: (batch, N, 2)
    :param indexing     : whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy)

    :return             : interp
        :shape: (batch, N, channels)

    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    grid = torch.as_tensor(grid)       
    query_points = torch.as_tensor(query_points)

    shape = list(grid.size())

    if len(shape) != 4:
        msg = 'Grid must be 4 dimensional. Received size: '
        raise ValueError(msg + str(len(grid.shape)))

    batch_size, height, width, channels = shape
     
    query_type = query_points.dtype
    grid_type = grid.dtype

    if (len(query_points.shape) != 3 or query_points.shape[2] != 2):
        msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received '
                'size: ')
        raise ValueError(msg + str(query_points.shape))
    
    _, num_queries, _ = list(query_points.size())

    if height < 2 or width < 2:
      msg = 'Grid must be at least batch_size x 2 x 2 in size. Received size: '
      raise ValueError(msg + str(grid.shape))

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == 'ij' else [1, 0]

    unstacked_query_points = torch.unbind(query_points, dim=2)
    

    for dim in index_order:
        queries = unstacked_query_points[dim]
        #print("queries", queries)
        size_in_indexing_dimension = shape[dim + 1]
        
        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type).to(device)
        min_floor = torch.tensor(0.0, dtype=query_type).to(device)
        floor = torch.minimum(torch.maximum(min_floor, torch.floor(queries)), max_floor)
        int_floor = floor.type(torch.int32)
        floors.append(int_floor)


        ceil = int_floor + 1
        ceils.append(ceil)
        #print("int_floor", int_floor.shape)
        #print("ceils", ceil.shape)
        
        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).type(grid_type)
        #print("alpha",alpha)
        # exit()
        min_alpha = torch.tensor(0.0, dtype=grid_type).to(device)
        max_alpha = torch.tensor(1.0, dtype=grid_type).to(device)
        alpha = torch.minimum(torch.maximum(min_alpha, alpha), max_alpha)
        
        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)
        
    # flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
    flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
    # batch_offsets  = torch.reshape(torch.arange(0, batch_size) * height * width, [batch_size, 1]).to(device)
    batch_offsets  = torch.reshape(torch.arange(0, batch_size) * height * width, [batch_size, 1]).to(device)
    
    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = flattened_grid[linear_coordinates]       
        return torch.reshape(gathered_values,[batch_size, num_queries, channels])
        
    # grab the pixel values in the 4 corners around each query point
    top_left        = gather(floors[0], floors[1])
    top_right       = gather(floors[0], ceils[1])
    bottom_left     = gather(ceils[0], floors[1])
    bottom_right    = gather(ceils[0], ceils[1])
    #print("corner:", floors, ceils)
    #print("corner_value:", top_left, top_right, bottom_left, bottom_right)
    
    # now, do the actual interpolation
    interp_top      = alphas[1] * (top_right     - top_left)    + top_left
    interp_bottom   = alphas[1] * (bottom_right  - bottom_left) + bottom_left
    interp          = alphas[0] * (interp_bottom - interp_top)  + interp_top
    
    return interp
    
def dense_image_warp(device, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Image warping using per-pixel flow vectors.

    :param image         : 4-D float Tensor
        :shape: (batch, height, width, channels)
    :param flow          : 4-D float Tensor
        :shape: (batch, height, width, 2)
   
    :return             : interpolated
        :shape: (batch, height, width, channels)

    """
    #print("image_size", list(image.size()))
    #print("flow", list(flow.size()))


    batch_size, height, width, channels = list(image.size())

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing = 'xy')
    
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)

    stacked_grid = torch.stack([grid_y, grid_x], axis=2).type(flow.dtype)
    batched_grid            = torch.unsqueeze(stacked_grid, axis=0)
    query_points_on_grid    = batched_grid - flow

    #print("query_points_on_grid",query_points_on_grid)

    query_points_flattened  = torch.reshape(query_points_on_grid, [batch_size, height * width, 2])

    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_bilinear(device, image, query_points_flattened)
    interpolated = torch.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated

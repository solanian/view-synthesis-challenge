
import numpy as np

def intrinsic_matrix(fx, fy, cx, cy):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])
def get_pixtocam(focal, width, height):
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * .5, height * .5)
    return np.linalg.inv(camtopix)


def pixel_coordinates(width, height):
    """Tuple of the x and y integer coordinates for a grid of pixels."""
    return np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
def pixels_to_rays(pix_x_int, pix_y_int, pixtocams, c2w):
                   # distortion_params=None,
                   # pixtocam_ndc=None,
                   # camtype=ProjectionType.PERSPECTIVE):
    """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

  Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
  inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
  distortion coefficients distortion_params and NDC space projection matrix
  pixtocam_ndc), computes the corresponding 3D camera rays.

  Vectorized over the leading dimensions of the first four arguments.

  Args:
    pix_x_int: int array, shape SH, x coordinates of image pixels.
    pix_y_int: int array, shape SH, y coordinates of image pixels.
    pixtocams: float array, broadcastable to SH + [3, 3], inverse intrinsics.
    c2w : camtoworlds: float array, broadcastable to SH + [3, 4], camera extrinsics.
    distortion_params: dict of floats, optional camera distortion parameters.
    pixtocam_ndc: float array, [3, 3], optional inverse intrinsics for NDC.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.

  Returns:
    origins: float array, shape SH + [3], ray origin points.
    directions: float array, shape SH + [3], ray direction vectors.
    viewdirs: float array, shape SH + [3], normalized ray direction vectors.
    radii: float array, shape SH + [1], ray differential radii.
    imageplane: float array, shape SH + [2], xy coordinates on the image plane.
      If the image plane is at world space distance 1 from the pinhole, then
      imageplane will be the xy coordinates of a pixel in that space (so the
      camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
  """

    # Must add half pixel offset to shoot rays through pixel centers.
    def pix_to_dir(x, y):
        return np.stack([x + .5, y + .5, np.ones_like(x)], axis=-1)

    # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
    pixel_dirs_stacked = np.stack([
        pix_to_dir(pix_x_int, pix_y_int),
        pix_to_dir(pix_x_int + 1, pix_y_int),
        pix_to_dir(pix_x_int, pix_y_int + 1)
    ], axis=0)

    mat_vec_mul = lambda A, b: np.matmul(A, b[..., None])[..., 0]

    # Apply inverse intrinsic matrices.
    camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

    # Apply camera rotation matrices.
    # Extract the offset rays.
    directions, dx, dy = mat_vec_mul(c2w[..., :3, :3], camera_dirs_stacked)

    origins = np.broadcast_to(c2w[..., :3, -1], directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its neighbors.
    dx_norm = np.linalg.norm(dx - directions, axis=-1)
    dy_norm = np.linalg.norm(dy - directions, axis=-1)

    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see the original mipnerf paper).
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12)
    return origins, directions, viewdirs, radii#, imageplane
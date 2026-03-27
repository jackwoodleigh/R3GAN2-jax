import jax
from jax import lax
import numpy as np
import jax.numpy as jnp
from flax import nnx
import scipy.signal

#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.

def matrix(*rows):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, jax.Array)]
    if len(ref) == 0:
        return jnp.asarray(np.asarray(rows))
    elems = [x if isinstance(x, jax.Array) else jnp.full(ref[0].shape, x) for x in elems]
    return jnp.stack(elems, axis=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    )

def translate3d(tx, ty, tz):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    )

def scale2d(sx, sy):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
    )

def scale3d(sx, sy, sz):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
    )
    
def rotate2d(theta):
    return matrix(
        [jnp.cos(theta), jnp.sin(-theta), 0],
        [jnp.sin(theta), jnp.cos(theta),  0],
        [0,              0,               1],
    )
    
def translate2d_inv(tx, ty):
    return translate2d(-tx, -ty)

def scale2d_inv(sx, sy):
    return scale2d(1 / sx, 1 / sy)

def rotate2d_inv(theta):
    return rotate2d(-theta)

def translate_channels(num_channels, t):
    C = jnp.tile(jnp.eye(num_channels + 1)[None], (t.shape[0], 1, 1))
    idx = jnp.arange(num_channels)
    return C.at[:, idx, -1].set(t[:, None])  

def scale_channels(num_channels, s):
    C = jnp.tile(jnp.eye(num_channels + 1)[None], (s.shape[0], 1, 1))
    idx = jnp.arange(num_channels)
    return C.at[:, idx, idx].set(s[:, None])

def generate_orthogonal_span(v):
    v = v[:-1]

    sorted_idx = jnp.argsort(jnp.abs(v))
    i = sorted_idx[0]
    j = sorted_idx[1]

    e_i = jnp.zeros_like(v).at[i].set(1.0)
    e_j = jnp.zeros_like(v).at[j].set(1.0)

    u = e_i - (v @ e_i) * v
    u = u / jnp.linalg.norm(u)

    w = e_j - (v @ e_j) * v - (u @ e_j) / (u @ u) * u
    w = w / jnp.linalg.norm(w)

    S = jnp.outer(u, w) - jnp.outer(w, u)
    return S

def rotate_channels(S, theta):
    C = jnp.tile(jnp.eye(S.shape[0] + 1)[None], (theta.shape[0], 1, 1))
    S = jnp.tile(S[None], (theta.shape[0], 1, 1))
    theta = -theta.reshape(-1, 1, 1)
    R = jax.scipy.linalg.expm(S * theta)
    C = C.at[:, :R.shape[1], :R.shape[2]].set(R)
    return C


def upsample2d(images, conv_weight, conv_pad):
    N, C, H, W = images.shape
    images = jnp.stack([images, jnp.zeros_like(images)], axis=4).reshape(N, C, H, -1)[:, :, :, :-1]
    images = jax.lax.conv_general_dilated(images, conv_weight[:, :, None, :], window_strides=(1,1), padding=[(0,0),(conv_pad,conv_pad)], feature_group_count=C)
    images = jnp.stack([images, jnp.zeros_like(images)], axis=3).reshape(N, C, -1, images.shape[3])[:, :, :-1, :]
    images = jax.lax.conv_general_dilated(images, conv_weight[:, :, :, None], window_strides=(1,1), padding=[(conv_pad,conv_pad),(0,0)], feature_group_count=C)
    return images

def downsample2d(images, conv_weight, conv_pad, Hz_pad):
    N, C, H, W = images.shape
    images = jax.lax.conv_general_dilated(images, conv_weight[:, :, None, :], window_strides=(1,2), padding=[(0,0),(conv_pad,conv_pad)], feature_group_count=C)
    images = images[:, :, :, Hz_pad:-Hz_pad]
    images = jax.lax.conv_general_dilated(images, conv_weight[:, :, :, None], window_strides=(2,1), padding=[(conv_pad,conv_pad),(0,0)], feature_group_count=C)
    images = images[:, :, Hz_pad:-Hz_pad, :]
    return images
    

def affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    if align_corners:
        xs = jnp.linspace(-1, 1, W)
        ys = jnp.linspace(-1, 1, H)
    else:
        xs = jnp.linspace(-1 + 1/W, 1 - 1/W, W)
        ys = jnp.linspace(-1 + 1/H, 1 - 1/H, H)
    grid_x, grid_y = jnp.meshgrid(xs, ys, indexing='xy')
    ones = jnp.ones_like(grid_x)
    grid = jnp.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)  # [H*W, 3]
    coords = (theta @ grid.T).transpose(0, 2, 1)  # [batch, H*W, 2]
    return coords.reshape(N, H, W, 2)


def grid_sample(images, grid, align_corners=False):
    N, C, H_in, W_in = images.shape

    x = grid[..., 0]
    y = grid[..., 1]

    if align_corners:
        x = (x + 1) / 2 * (W_in - 1)
        y = (y + 1) / 2 * (H_in - 1)
    else:
        x = (x + 1) / 2 * W_in - 0.5
        y = (y + 1) / 2 * H_in - 0.5

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y))[..., None]
    wb = ((x1 - x) * (y - y0))[..., None]
    wc = ((x - x0) * (y1 - y))[..., None]
    wd = ((x - x0) * (y - y0))[..., None]

    n_idx = jnp.arange(N)[:, None, None]

    def safe_gather(iy, ix):
        valid = ((ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in))[..., None].astype(images.dtype)
        iy = jnp.clip(iy, 0, H_in - 1)
        ix = jnp.clip(ix, 0, W_in - 1)
        return images[n_idx, :, iy, ix] * valid  # [N, H_out, W_out, C]

    out = (safe_gather(y0, x0) * wa +
           safe_gather(y1, x0) * wb +
           safe_gather(y0, x1) * wc +
           safe_gather(y1, x1) * wd)

    return out.transpose(0, 3, 1, 2)  # [N, C, H_out, W_out]



class AugmentPipe:
    def __init__(self, p=1,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        p                = float(p)
            
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.lumaflip         = float(lumaflip)         # Probability multiplier for luma flip.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)

        Hz_lo = np.asarray(wavelets['sym2'])
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2
        Hz_fbank = np.eye(4, 1)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.Hz_fbank = jnp.array(Hz_fbank, dtype=jnp.float32)
    
    def __call__(self, images_list, p, key, debug_percentile=None):
        batch_size, num_channels, height, width = images_list[0].shape
        key = jax.random.fold_in(key, ord('A'))
        if debug_percentile is not None:
            debug_percentile = jnp.asarray(debug_percentile, dtype=jnp.float32)
        
        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = jnp.eye(3)
        G_inv = I_3
        
        
        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------
        
        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            key, key1, key2 = jax.random.split(key, 3)
            i = jnp.floor(jax.random.uniform(key1, [batch_size]) * 2)
            i = jnp.where(jax.random.uniform(key2, [batch_size]) < self.xflip * p, i, jnp.zeros_like(i))
            if debug_percentile is not None:
                i = jnp.full_like(i, jnp.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)
            
         # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            key, key1, key2 = jax.random.split(key, 3)
            i = jnp.floor(jax.random.uniform(key1, [batch_size]) * 4)
            i = jnp.where(jax.random.uniform(key2, [batch_size])  < self.rotate90 * p, i, jnp.zeros_like(i))
            if debug_percentile is not None:
                i = jnp.full_like(i, jnp.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)
            
        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            key, key1, key2 = jax.random.split(key, 3)
            t = (jax.random.uniform(key1, [batch_size, 2]) * 2 - 1) * self.xint_max
            t = jnp.where(jax.random.uniform(key2, [batch_size, 1]) < self.xint * p, t, jnp.zeros_like(t))
            if debug_percentile is not None:
                t = jnp.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            G_inv = G_inv @ translate2d_inv(jnp.round(t[:,0] * width), jnp.round(t[:,1] * height))
            
            
        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------            
            
        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            key, key1, key2 = jax.random.split(key, 3)
            s = jnp.exp2(jax.random.normal(key1, [batch_size]) * self.scale_std)
            s = jnp.where(jax.random.uniform(key2, [batch_size]) < self.scale * p, s, jnp.ones_like(s))
            if debug_percentile is not None:
                s = jnp.full_like(s, jnp.exp2(jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.scale_std))
            G_inv = G_inv @ scale2d_inv(s, s)
            
        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - jnp.sqrt(jnp.clip(1 - self.rotate * p, 0, 1)) # P(pre OR post) = p
        if self.rotate > 0:
            key, key1, key2 = jax.random.split(key, 3)
            theta = (jax.random.uniform(key1, [batch_size])  * 2 - 1) * np.pi * self.rotate_max
            theta = jnp.where(jax.random.uniform(key2, [batch_size]) < p_rot, theta, jnp.zeros_like(theta))
            if debug_percentile is not None:
                theta = jnp.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            G_inv = G_inv @ rotate2d_inv(-theta)
           
        # Apply anisotropic scaling with probability (aniso * strength). 
        if self.aniso > 0:
            key, key1, key2 = jax.random.split(key, 3)
            s = jnp.exp2(jax.random.normal(key1, [batch_size]) * self.aniso_std)
            s = jnp.where(jax.random.uniform(key2, [batch_size])  < self.aniso * p, s, jnp.ones_like(s))
            if debug_percentile is not None:
                s = jnp.full_like(s, jnp.exp2(jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)
        
        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            key, key1, key2 = jax.random.split(key, 3)
            theta = (jax.random.uniform(key1, [batch_size]) * 2 - 1) * np.pi * self.rotate_max
            theta = jnp.where(jax.random.uniform(key2, [batch_size]) < p_rot, theta, jnp.zeros_like(theta))
            if debug_percentile is not None:
                theta = jnp.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.
            
        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            key, key1, key2 = jax.random.split(key, 3)
            t = jax.random.normal(key1, [batch_size,2 ]) * self.xfrac_std
            t = jnp.where(jax.random.uniform(key2, [batch_size, 1]) < self.xfrac * p, t, jnp.zeros_like(t))
            if debug_percentile is not None:
                t = jnp.full_like(t, jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)
            G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)
            
        
        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------
            
        # Execute if the transform is not identity.
        if G_inv is not I_3:
            
            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]) # [idx, xyz]
            cp = G_inv @ cp.T # [batch, xyz, idx]
            Hz = np.asarray(wavelets['sym6'], dtype=np.float32)
            Hz_pad = len(Hz) // 4
            margin = cp[:, :2, :].transpose(1, 0, 2).reshape(2, -1) # [xy, batch * idx]
            margin = jnp.max(jnp.concatenate([-margin, margin]), axis=1) # [x0, y0, x1, y1]
            margin = margin + jnp.array([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2)
            margin = jnp.maximum(margin, jnp.array([0, 0] * 2))
            margin = jnp.minimum(margin, jnp.array([width-1, height-1] * 2))
            mx0, my0, mx1, my1 = jnp.ceil(margin).astype(jnp.int32)

            # Pad image and adjust origin.
            images_list = [jnp.pad(images, ((0,0), (0,0), (my0,my1), (mx0,mx1)), mode='reflect') for images in images_list]
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            
            conv_weight = jnp.tile(jnp.array(Hz[None, None, ::-1]), (num_channels, 1, 1))
            conv_pad = (len(Hz) + 1) // 2
            images_list = [upsample2d(images, conv_weight, conv_pad) for images in images_list]
            G_inv = scale2d(2, 2) @ G_inv @ scale2d_inv(2, 2)
            G_inv = translate2d(-0.5, -0.5) @ G_inv @ translate2d_inv(-0.5, -0.5)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images_list[0].shape[3], 2 / images_list[0].shape[2]) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2])
            grid = affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images_list = [grid_sample(images, grid) for images in images_list]

            # Downsample and crop.
            conv_weight = jnp.tile(jnp.array(Hz[None, None, :]), (num_channels, 1, 1))  # not flipped for downsample
            
            conv_pad = (len(Hz) - 1) // 2
            images_list = [downsample2d(images, conv_weight, conv_pad, Hz_pad) for images in images_list]
            
        
        
        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_C = jnp.eye(num_channels + 1)
        C = I_C
        
        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            key, key1, key2 = jax.random.split(key, 3)
            b = jax.random.normal(key1, [batch_size]) * self.brightness_std
            b = jnp.where(jax.random.uniform(key2, [batch_size]) < self.brightness * p, b, jnp.zeros_like(b))
            if debug_percentile is not None:
                b = jnp.full_like(b, jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.brightness_std)
            C = translate_channels(num_channels, b) @ C
        
        # Apply contrast with probability (contrast * strength).
        if self.contrast > 0:
            key, key1, key2 = jax.random.split(key, 3)
            c = jnp.exp2(jax.random.normal(key1, [batch_size])  * self.contrast_std)
            c = jnp.where(jax.random.uniform(key2, [batch_size]) < self.contrast * p, c, jnp.ones_like(c))
            if debug_percentile is not None:
                c = jnp.full_like(c, jnp.exp2(jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.contrast_std))
            C = scale_channels(num_channels, c) @ C
        
        # Apply luma flip with probability (lumaflip * strength).
        v = jnp.asarray([1 for _ in range(num_channels)] + [0]) / jnp.sqrt(num_channels) # Luma axis.
        if self.lumaflip > 0:
            key, key1, key2 = jax.random.split(key, 3)
            i = jnp.floor(jax.random.uniform(key1, [batch_size, 1, 1]) * 2)
            i = jnp.where(jax.random.uniform(key2, [batch_size, 1, 1]) < self.lumaflip * p, i, jnp.zeros_like(i))
            if debug_percentile is not None:
                i = jnp.full_like(i, jnp.floor(debug_percentile * 2))
            C = (I_C - 2 * jnp.outer(v, v) * i) @ C # Householder reflection.
            
        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0:
            key, key1, key2 = jax.random.split(key, 3)
            theta = (jax.random.uniform(key1, [batch_size]) * 2 - 1) * np.pi * self.hue_max
            theta = jnp.where(jax.random.uniform(key2, [batch_size])< self.hue * p, theta, jnp.zeros_like(theta))
            if debug_percentile is not None:
                theta = jnp.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.hue_max)
            C = rotate_channels(generate_orthogonal_span(v), theta) @ C # Rotate around v.
            
        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0:
            key, key1, key2 = jax.random.split(key, 3)
            s = jnp.exp2(jax.random.normal(key1, [batch_size, 1, 1])* self.saturation_std)
            s = jnp.where(jax.random.uniform(key2, [batch_size, 1, 1]) < self.saturation * p, s, jnp.ones_like(s))
            if debug_percentile is not None:
                s = jnp.full_like(s, jnp.exp2(jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
            C = (jnp.outer(v, v) + (I_C - jnp.outer(v, v)) * s) @ C
            
            
        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # Execute if the transform is not identity.
        if C is not I_C:
            images_list = [images.reshape([batch_size, num_channels, height * width]) for images in images_list]
            images_list = [(C[:, :num_channels, :num_channels] @ images + C[:, :num_channels, num_channels:]) for images in images_list]
            images_list = [images.reshape([batch_size, num_channels, height, width]) for images in images_list]

        # ----------------------
        # Image-space filtering.
        # ----------------------
        
        if self.imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands
            expected_power = jnp.array([10, 1, 1, 1], dtype=jnp.float32) / 13

            g = jnp.ones([batch_size, num_bands])
            for i, band_strength in enumerate(self.imgfilter_bands):
                key, key1, key2 = jax.random.split(key, 3)
                t_i = jnp.exp2(jax.random.normal(key1, [batch_size]) * self.imgfilter_std)
                t_i = jnp.where(jax.random.uniform(key2, [batch_size]) < self.imgfilter * p * band_strength, t_i, jnp.ones_like(t_i))
                if debug_percentile is not None:
                    t_i = jnp.full_like(t_i, jnp.exp2(jax.scipy.special.erfinv(debug_percentile * 2 - 1) * self.imgfilter_std)) if band_strength > 0 else jnp.ones_like(t_i)
                t = jnp.ones([batch_size, num_bands])
                t = t.at[:, i].set(t_i)
                t = t / jnp.sqrt((expected_power * t ** 2).sum(axis=-1, keepdims=True))
                g = g * t 
            
            Hz_prime = g @ self.Hz_fbank                                          # [batch, tap]
            Hz_prime = jnp.tile(Hz_prime[:, None, :], (1, num_channels, 1))       # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1])       # [batch * channels, 1, tap]
            
            p = self.Hz_fbank.shape[1] // 2
            images_list = [images.reshape([1, batch_size * num_channels, height, width]) for images in images_list]
            images_list = [jnp.pad(images, ((0,0), (0,0), (p,p), (p,p)), mode='reflect') for images in images_list]
            images_list = [jax.lax.conv_general_dilated(images, Hz_prime[:, :, None, :], window_strides=(1,1), padding='VALID', feature_group_count=batch_size*num_channels) for images in images_list]
            images_list = [jax.lax.conv_general_dilated(images, Hz_prime[:, :, :, None], window_strides=(1,1), padding='VALID', feature_group_count=batch_size*num_channels) for images in images_list]
            images_list = [images.reshape([batch_size, num_channels, height, width]) for images in images_list]
        
        
        # ------------------------
        # Image-space corruptions.
        # ------------------------

        # Apply additive RGB noise with probability (noise * strength).
        if self.noise > 0:
            key, key1, key2, key3 = jax.random.split(key, 4)
            sigma = jnp.abs(jax.random.normal(key1, [batch_size, 1, 1, 1])) * self.noise_std
            sigma = jnp.where(jax.random.uniform(key2, [batch_size, 1, 1, 1]) < self.noise * p, sigma, jnp.zeros_like(sigma))
            if debug_percentile is not None:
                sigma = jnp.full_like(sigma, jax.scipy.special.erfinv(debug_percentile) * self.noise_std)
            images_list = [(images + jax.random.normal(key3, [batch_size, num_channels, height, width]) * sigma) for images in images_list]
            
        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            key, key1, key2, key3 = jax.random.split(key, 4)
            size = jnp.full([batch_size, 2, 1, 1, 1], self.cutout_size)
            size = jnp.where(jax.random.uniform(key1, [batch_size, 1, 1, 1, 1]) < self.cutout * p, size, jnp.zeros_like(size))
            center = jax.random.uniform(key2, [batch_size, 2, 1, 1, 1])
            if debug_percentile is not None:
                size = jnp.full_like(size, self.cutout_size)
                center = jnp.full_like(center, debug_percentile)
            coord_x = jnp.arange(width).reshape([1, 1, 1, -1])
            coord_y = jnp.arange(height).reshape([1, 1, -1, 1])
            mask_x = (jnp.abs((coord_x + 0.5) / width - center[:, 0]) >= size[:, 0] / 2)
            mask_y = (jnp.abs((coord_y + 0.5) / height - center[:, 1]) >= size[:, 1] / 2)
            mask_x, mask_y = jnp.broadcast_arrays(mask_x, mask_y)
            mask = jnp.logical_or(mask_x, mask_y).astype(jnp.float32)

            results = []
            for images in images_list:
                mean = jnp.mean(images, axis=(2, 3), keepdims=True)
                std = jnp.std(images, axis=(2, 3), keepdims=True)
                noise = std * jax.random.normal(key3, [batch_size, num_channels, height, width]) + mean
                results += [images * mask + noise * (1 - mask)]
            images_list = results

        return images_list

            


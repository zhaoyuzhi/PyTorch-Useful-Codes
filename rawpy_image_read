import rawpy
import rawpy.enhance
import imageio

# Load a RAW file and save the postprocessed image using default parameters:
path = 'C:\\Users\\yzzha\\Desktop\\DCIM\\101D5200\\sorted\\DSC_2937\\DSC_2940.NEF'
with rawpy.imread(path) as raw:
    rgb = raw.postprocess()
print(rgb.dtype, rgb.shape)
imageio.imsave('default.tiff', rgb)

# Save as 16-bit linear image:
with rawpy.imread(path) as raw:
    rgb = raw.postprocess(gamma = (1, 1), no_auto_bright = True, output_bps = 16)
print(rgb.dtype, rgb.shape)
imageio.imsave('linear.tiff', rgb)

# Find bad pixels using multiple RAW files and repair them:
bad_pixels = rawpy.enhance.find_bad_pixels(path)
with rawpy.imread(path) as raw:
    rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method = 'median')
    rgb = raw.postprocess()
print(rgb.dtype, rgb.shape)
imageio.imsave('repaired.tiff', rgb)

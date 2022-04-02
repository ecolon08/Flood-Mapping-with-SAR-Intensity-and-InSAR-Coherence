import rasterio
from shapely.geometry import box, mapping
import fiona
import rasterio.mask
from tqdm import tqdm
from osgeo import gdal

def create_shp_from_geotiff(geotiff_pth, shp_outdir, chip_fname):
    """
    Function to generate a shapefile from a geotiff input

    :param geotiff_pth: path to geotiff file
    :param shp_outdir: path to output directory
    :param chip_fname: filename to save shapefiles as
    :return: None, function saves to disk
    """
    # Open the geotiff chip using rasterio
    chip_raster = rasterio.open(geotiff_pth)

    # extract the chip bounds
    chip_bounds = chip_raster.bounds

    # Create a polygon from the raster bounds
    chip_bbox_poly = box(*chip_bounds)

    # Create a schema
    schema = {'geometry': 'Polygon', 'properties': {'id': 'str'}}

    # Create shapefile
    fname = f"{shp_outdir}/{chip_fname}_shp"

    try:
        with fiona.open(fname, 'w', driver='ESRI Shapefile', crs=chip_raster.crs.to_dict(), schema=schema) as c:
            c.write({'geometry': mapping(chip_bbox_poly), 'properties': {}})

            print("shapefile written to disk")
    except:
        print("Could not write shape file")


def cut_geotiff_from_shp(shp_pth, geotiff_pth, cut_width, cut_height, out_dir, fname):
    """
    Function to cut/mask a geotiff file given an input shapefile with a smaller bounding box
    :param shp_pth: path to shapefile for cutting
    :param geotiff_pth: path to geotiff to cut
    :param cut_width: integer, masked raster width
    :param cut_height: integer, masked raster height
    :param out_dir: output directory path
    :param fname: output filename
    :return: None, function saves to disk
    """

    # Open the shapefile as read-only for cutting and extract geometry
    with fiona.open(shp_pth, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Open geotiff raster to cut
    try:
        with rasterio.open(geotiff_pth) as src:
            # apply mask based off the shapefile
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)

            # extract metadata from original geotiff
            out_meta = src.meta

    except:
        print("Geotiff raster could not be opened")

    # write output file
    out_meta.update({"driver": "GTiff",
                     "height": cut_height,
                     "width": cut_width,
                     "transform": out_transform})

    out_pth = f"{out_dir}/{fname}.tif"

    with rasterio.open(out_pth, "w", **out_meta) as dest:
        dest.write(out_image)


def cut_lrg_rstr(geotiff_pth, shp_pth, out_fpth, cut_height, cut_width):
    """
    Function to cut geotiff given an input shapefile with a smaller bounding box. This function uses GDAL
    and nearest neighbor interpolation to fill in any border / overlap gaps

    :param geotiff_pth: path to geotiff to cut
    :param shp_pth: path to shapefile to use for cutting
    :param out_fpth: output directory path
    :param cut_height: integer, masked raster height
    :param cut_width: integer, masked raster width
    :return: None, function saves to disk
    """

    clip = gdal.Warp(out_fpth,
                     geotiff_pth,
                     format = 'GTiff',
                     cutlineDSName = shp_pth,
                     cropToCutline = True,
                     resampleAlg = gdal.GRA_NearestNeighbour,
                     width=cut_width,
                     height=cut_height)


def batch_clip_w_gdal(shp_fpaths, geotiff_pth, out_dir, fname, cut_height, cut_width):
    """
    Function to batch clip a large geotiff using GDAL
    :param shp_fpaths: array containing paths to shapefiles
    :param geotiff_pth: path to geotidd to cut
    :param out_dir: output directory path
    :param fname: output file name
    :param cut_height: integer, masked raster height
    :param cut_width: integer, masked raster width
    :return: None, function saves to disk
    """

    for shp_pth in tqdm(shp_fpaths):

        out_fpth = f"{out_dir}\\{fname}.tif"

        try:
            cut_lrg_rstr(geotiff_pth, shp_pth, out_fpth, cut_height, cut_width)
        except:
            continue

import rasterio
from shapely.geometry import box, mapping
import fiona
import rasterio.mask


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
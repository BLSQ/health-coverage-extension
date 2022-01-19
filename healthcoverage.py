import os
import shutil
from typing import Sequence

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.merge
import requests
from fiona import transform
from gooey import Gooey, GooeyParser
from rasterio.crs import CRS
from rasterstats import zonal_stats
from shapely.geometry import Polygon, shape
from tqdm import tqdm


def coverage(
    districts: str,
    csi: str,
    cs: str,
    population: str,
    output_dir: str,
    min_population: int,
    min_distance_from_csi: int,
    max_distance_served: int,
    country: str,
    epsg: int,
    un_adj: bool = True,
    constrained: bool = True,
    show_progress: bool = False,
):
    """Génère les tables et cartes d'extension de la couverture sanitaire."""
    districts = gpd.read_file(districts)
    if "geometry" not in districts or np.count_nonzero(districts.is_valid) == 0:
        raise ValueError("Le fichier de districts ne contient aucune géométrie.")
    if not districts.crs:
        districts.crs = CRS.from_epsg(4326)

    csi = gpd.read_file(csi)
    if "geometry" not in csi or np.count_nonzero(csi.is_valid) == 0:
        raise ValueError("Le fichier de CSI ne contient aucune géométrie.")
    if not csi.crs:
        csi.crs = CRS.from_epsg(4326)

    cs = gpd.read_file(cs)
    if "geometry" not in cs or np.count_nonzero(cs.is_valid) == 0:
        raise ValueError("Le fichier de CS ne contient aucune géométrie.")
    if not cs.crs:
        cs.crs = CRS.from_epsg(4326)

    # Automatically download Worldpop data if needed
    # TODO: What about year > 2020 ? Not sure if URL is going to be the same.
    if not population:
        print("La carte de population n'a pas été renseignée.")
        print("Télécharge les données WorldPop...")
        dst_dir = os.path.join(output_dir, "worldpop")
        os.makedirs(dst_dir, exist_ok=True)
        population = download_worldpop(
            country=country,
            output_dir=dst_dir,
            year=2020,
            un_adj=un_adj,
            constrained=constrained,
            show_progress=show_progress,
            overwrite=False,
        )

    print("Génère les tiles de population pour chaque district...")
    dst_dir = os.path.join(output_dir, "population_tiles")
    os.makedirs(dst_dir, exist_ok=True)
    split_population_raster(
        population, districts, output_dir=dst_dir, show_progress=show_progress
    )

    print("Calcule la population desservie...")
    dst_file = os.path.join(dst_dir, "population_served.tif")
    served = generate_population_served(
        districts,
        population_dir=dst_dir,
        dst_file=dst_file,
        epsg=epsg,
        area_served=max_distance_served,
        show_progress=show_progress,
    )

    print("Génère les zones d'extension potentielles...")
    dst_file = os.path.join(output_dir, "priority_areas.tif")
    priority_areas = generate_priority_areas(
        population_served=served,
        csi=csi,
        raster_template=served,
        dst_file=dst_file,
        epsg=epsg,
        min_dist_from_csi=min_distance_from_csi,
    )

    print("Calcule la population desservie par chaque CSI...")
    column = f"population_{int(max_distance_served / 1000)}km"
    csi[column] = population_served_per_fosa(csi, served)
    csi.to_file(os.path.join(output_dir, "csi_population.gpkg"), driver="GPKG")

    print("Calcule la population desservie par chaque CS...")
    cs[column] = population_served_per_fosa(cs, served)
    cs.to_file(os.path.join(output_dir, "cs_population.gpkg"), driver="GPKG")

    print("Analyse les zones potentielles d'extension...")
    potential_areas = analyse_potential_areas(priority_areas, csi, epsg, min_population)
    potential_areas.to_file(
        os.path.join(output_dir, "potential_areas.gpkg"), driver="GPKG"
    )

    print("Analyse le potentiel d'extension des CS...")
    potential_cs = analyse_cs(cs, csi, epsg)
    potential_cs.to_file(os.path.join(output_dir, "potential_cs.gpkg"), driver="GPKG")

    # shutil.rmtree(os.path.join(output_dir, "population_tiles"))
    # shutil.rmtree(os.path.join(output_dir, "worldpop"))

    print("Modélisation terminée !")
    return


def _build_worldpop_url(country, year, un_adj=True, constrained=True):
    """Build download URL.

    Parameters
    ----------
    country : str
        Country ISO A3 code.
    year : int, optional
        Year of interest (2000--2020).
    un_adj : bool, optional
        Use UN adjusted population counts.
    constrained : bool, optional
        Constrained or unconstrained dataset ?

    Returns
    -------
    url : str
        Download URL.
    """
    if constrained:
        base_url = (
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained"
        )
        return (
            f"{base_url}/{year}/maxar_v1/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}_constrained.tif"
        )
    else:
        base_url = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
        return (
            f"{base_url}/{year}/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}.tif"
        )


def download_worldpop(
    country,
    output_dir,
    year=2020,
    un_adj=True,
    constrained=True,
    show_progress=True,
    overwrite=False,
):
    """Download a WorldPop population dataset.

    Four types of datasets are supported:
      - Unconstrained (100 m)
      - Unconstrained and UN adjusted (100 m)
      - Constrained (100 m)
      - Constrained and UN adjusted (100 m)

    See Worldpop website for more details:
    <https://www.worldpop.org/project/categories?id=3>

    Parameters
    ----------
    country : str
        Country ISO A3 code.
    output_dir : str
        Path to output directory.
    year : int, optional
        Year of interest (2000--2020). Default=2020.
    un_adj : bool, optional
        Use UN adjusted population counts. Default=False.
    constrained : bool, optional
        Constrained or unconstrained dataset ?
    show_progress : bool, optional
        Show progress bar. Default=False.
    overwrite : bool, optional
        Overwrite existing files. Default=True.

    Return
    ------
    str
        Path to output GeoTIFF file.
    """
    url = _build_worldpop_url(
        country, year=year, un_adj=un_adj, constrained=constrained
    )

    fp = os.path.join(output_dir, url.split("/")[-1])
    if os.path.isfile(fp) and not overwrite:
        print("Données de population déjà téléchargées.")
        return fp

    print(f"Téléchargement des données de population depuis {url}.")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        if show_progress:
            size = int(r.headers.get("Content-Length"))
            bar_format = "{desc} | {percentage:3.0f}% | {rate_fmt}"
            pbar = tqdm(
                desc=os.path.basename(fp),
                bar_format=bar_format,
                total=size,
                unit_scale=True,
                unit="B",
            )

        with open(fp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    if show_progress:
                        pbar.update(1024)

        if show_progress:
            pbar.close()

    print(f"Données de population téléchargées dans {output_dir}.")
    return fp


def split_population_raster(
    population_raster: str,
    districts: gpd.GeoDataFrame,
    output_dir: str,
    show_progress: bool = True,
    overwrite: bool = False
):
    """Split population raster per district.

    The function split the input population raster into
    multiple tiles (one per district). This is to avoid
    taking into account population from other districts
    in the following computations.

    Parameters
    ----------
    population_raster : str
        Path to population raster.
    districts : geodataframe
        Health districts.
    output_dir : str
        Path to output directory.
    show_progress : bool
        Show progress bar.
    overwrite : bool
        Overwrite existing files (default=False).
    """
    if not os.path.isfile(population_raster):
        raise ValueError(f"Population raster not found at {population_raster}.")
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(population_raster) as src:

        # Reproject districts geodataframe if needed
        districts_ = districts.copy()
        if districts_.crs != src.crs:
            districts_ = districts_.to_crs(src.crs)

        dst_profile = src.profile
        dst_profile["compress"] = "zstd"
        dst_profile["predictor"] = 3

        if show_progress:
            pbar = tqdm(total=len(districts_))

        for index, district in districts_.iterrows():

            fp = os.path.join(output_dir, f"{index}.tif")
            if os.path.isfile(fp) and not overwrite:
                if show_progress:
                    pbar.update(1)
                continue

            # Read a window of the population raster based on
            # the district geometry.
            window = rasterio.mask.geometry_window(
                src, shapes=[district.geometry.__geo_interface__]
            )
            transform = src.window_transform(window)
            population = src.read(1, window=window)

            # Make sure that pixels outside the district are assigned
            # the nodata value.
            mask_ = rasterio.mask.geometry_mask(
                geometries=[district.geometry.__geo_interface__],
                out_shape=population.shape,
                transform=transform,
                all_touched=False,
                invert=True,
            )
            population[~mask_] = src.nodata

            # Write raster tile to disk with district index as name
            dst_profile["transform"] = transform
            dst_profile["width"] = population.shape[1]
            dst_profile["height"] = population.shape[0]
            dst_profile["dtype"] = "float32"
            with rasterio.open(fp,
                "w",
                **dst_profile,
            ) as dst:
                dst.write(population.astype(np.float32), 1)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()


def _geom_from_bounds(bounds: Sequence[float]) -> Polygon:
    """Transform bounds tuple into a polygon."""
    xmin, ymin, xmax, ymax = bounds
    return Polygon(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    )


def get_kernel(population_raster: str, epsg: int, buffer_size: int) -> np.ndarray:
    """Get an array kernel corresponding to a buffer of a given size.

    Generates a flat, disk-shaped footprint as 2d array which
    corresponds to a buffer area around each pixel.

    Parameters
    ----------
    population_raster : str
        Path to population GeoTIFF.
    epsg : int
        EPSG code of the projected coordinate system.
    buffer_size : int
        Buffer size in meters.

    Return
    ------
    kernel : ndarray
        Buffer kernel as a numpy array.
    """
    with rasterio.open(population_raster) as src:

        extent = _geom_from_bounds(src.bounds)
        src_crs = f"EPSG:{src.crs.to_epsg()}"
        dst_crs = f"EPSG:{epsg}"

        # Temporarily reproject the geometry in order to set the buffer radius
        # in meters instead of decimal degrees.
        geom = transform.transform_geom(
            src_crs, dst_crs, extent.centroid.__geo_interface__
        )
        geom = shape(geom).buffer(buffer_size)
        geom = transform.transform_geom(dst_crs, src_crs, geom.__geo_interface__)

        # Rasterize the buffer and use it as a 2d array kernel
        crop, crop_transform = rasterio.mask.mask(
            src, shapes=[geom], crop=True, indexes=1
        )
        kernel = rasterio.mask.geometry_mask(
            [geom], out_shape=crop.shape, transform=crop_transform, invert=True
        )

    return kernel.astype("uint8")


def compute_population_served(
    population_raster: str, epsg: int, area_served: int, geom: Polygon
) -> np.ndarray:
    """Create a raster with population served per pixel.

    In the population served raster, each pixel is assigned
    the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding
    to the sum of all its neighboring pixels -- constrained by a
    disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    population_raster : str
        Path to population GeoTIFF.
    epsg : int
        EPSG code of the projected coordinate system.
    area_served : int
        Area served radius in meters.
    geom : shapely geometry
        Area of interest. Pixel outside will be masked.

    Return
    ------
    ndarray
        Population served per pixel.
    """
    kernel = get_kernel(population_raster, epsg=epsg, buffer_size=area_served)
    with rasterio.open(population_raster) as src:
        pop = src.read(1)
        pop[pop < 0] = 0
        pop[pop == src.nodata] = 0
        district = rasterio.mask.geometry_mask(
            geometries=[geom.__geo_interface__],
            out_shape=pop.shape,
            transform=src.transform,
            all_touched=True,
            invert=True,
        )
        pop[~district] = 0
    pop_sum = cv2.filter2D(src=pop, ddepth=-1, kernel=kernel).astype("int32")
    pop_sum[~district] = -1
    return pop_sum


def generate_population_served(
    districts: gpd.GeoDataFrame,
    population_dir: str,
    dst_file: str,
    epsg: int,
    area_served: int,
    show_progress: bool = True,
    overwrite: bool = False
) -> str:
    """Compute population served per pixel.

    In the population served raster, each pixel is assigned
    the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding
    to the sum of all its neighboring pixels -- constrained by a
    disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    districts : geodataframe
        Geodataframe with districts (EPSG:4326).
    population_dir : str
        Path to directory with splitted population rasters.
    dst_file : str
        Path to output file.
    epsg : int
        EPSG code of the projected coordinate system.
    area_served : int
        Area served radius in meters.
    show_progress : bool, optional
        Show a progress bar.
    overwrite : bool, optional
        Overwrite existing files (default=False).

    Return
    ------
    str
        Path to output file.

    Notes
    -----
    The functions uses GDAL, more specifically gdal_merge.py:
    <https://gdal.org/programs/gdal_merge.html>
    """
    if show_progress:
        pbar = tqdm(total=len(districts))

    # Compute population served for each population raster tile,
    # i.e. once per district
    for index, district in districts.iterrows():

        # skip if file has already been processed
        fp = os.path.join(population_dir, f"{index}_served.tif")
        if os.path.isfile(fp) and not overwrite:
            if show_progress:
                pbar.update(1)
            continue

        population_raster = os.path.join(population_dir, f"{index}.tif")
        with rasterio.open(population_raster) as src:
            dst_profile = src.profile.copy()
        pop_sum = compute_population_served(
            population_raster, epsg, area_served, district.geometry
        )

        dst_profile["dtype"] = "int32"
        dst_profile["nodata"] = -1
        with rasterio.open(fp, "w", **dst_profile) as dst:
            dst.write(pop_sum, 1)

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    # Merge all population served tiles into a mosaic raster
    if not os.path.isfile(dst_file) or overwrite:
        
        print("Assemble les tiles de population...")
        tiles = [
            os.path.join(population_dir, f)
            for f in os.listdir(population_dir)
            if f.endswith("_served.tif")
        ]

        with rasterio.open(tiles[0]) as src:
            meta = src.meta.copy()
        
        data, dst_transform = rasterio.merge.merge(tiles)
        meta.update({
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": dst_transform
        })

        with rasterio.open(dst_file, "w", **meta) as dst:
            dst.write(data)

    return dst_file


def already_served(
    fosa: gpd.GeoDataFrame, crs: str, min_distance: int, raster_template: str
):
    """Create a mask with areas already served by an existing CSI.

    Parameters
    ----------
    fosa : geodataframe
        A geodataframe with all the CSI.
    crs : str
        A projected CRS string used to compute the buffers in meters.
    min_distance : int
        Min. distance from existing CSI (i.e. buffer radius) in meters.
    raster_template : str
        Path to a template raster. Output raster will have equal width, height,
        crs and affine transform.

    Return
    ------
    ndarray
        Mask with positive values for areas served.
    """
    with rasterio.open(raster_template) as src:
        raster_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height

    return rasterio.features.geometry_mask(
        geometries=[
            geom.__geo_interface__
            for geom in fosa.to_crs(crs).buffer(min_distance).to_crs(raster_crs)
        ],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )


def population_served_per_fosa(
    fosa: gpd.GeoDataFrame,
    population_served_raster: str,
) -> pd.Series:
    """Get population served for each FOSA per district.

    Parameters
    ----------
    fosa : geodataframe
        Geodataframe with FOSAs.
    population_served_raster : str
        Path to population served raster.

    Return
    ------
    serie
        Population served for each CSI.
    """
    fosa_pop_served = pd.Series(index=fosa.index, dtype="int32")

    with rasterio.open(population_served_raster) as src:
        src_transform = src.transform
        pop_served = src.read(1)
        src_crs = src.crs

    if fosa.crs != src_crs:
        fosa_ = fosa.to_crs(src_crs)
    else:
        fosa_ = fosa.copy()

    for index, fs in fosa_.iterrows():
        if fs.geometry:
            row, col = rasterio.transform.rowcol(
                src_transform, fs.geometry.x, fs.geometry.y
            )
            try:
                fosa_pop_served.at[index] = int(pop_served[row, col])
            except IndexError:
                fosa_pop_served.at[index] = None
        else:
            fosa_pop_served.at[index] = None

    return fosa_pop_served


def generate_priority_areas(
    population_served: str,
    csi: gpd.GeoDataFrame,
    raster_template: str,
    dst_file: str,
    epsg: int = 32632,
    min_dist_from_csi: int = 15000,
) -> str:
    """Compute a raster showing priority areas.

    Priority areas are locations at more than <buffer_size> of
    an existing CSI and with more than <population_threshold>
    people in a buffer of 5km (as defined in the population_served
    raster).

    Parameters
    ----------
    population_served : str
        Path to population served raster.
    csi : geodataframe
        Geodataframe with CSI geometries.
    raster_template : str
        Path to a template raster. CRS, transform and shape
        will be used to compute the output raster.
    dst_file : str
        Path to output file.
    epsg : int
        EPSG code of the CRS used when computing buffers.
    min_dist_from_csi : int
        Buffer radius in meters. Minimum distance from existing
        CSI for priority areas.

    Return
    ------
    str
        Output file.
    """
    with rasterio.open(population_served) as src:
        priority_areas = src.read(1)
        dst_profile = src.profile.copy()
    served_by_csi = already_served(
        fosa=csi,
        crs=f"EPSG:{epsg}",
        min_distance=min_dist_from_csi,
        raster_template=raster_template,
    )
    priority_areas[served_by_csi == 1] = -1
    with rasterio.open(dst_file, "w", **dst_profile) as dst:
        dst.write(priority_areas, 1)
    return dst_file


def analyse_potential_areas(
    priority_areas: str,
    csi: gpd.GeoDataFrame,
    epsg: int = 32632,
    min_population: int = 5000,
) -> gpd.GeoDataFrame:
    """Analyse potential areas for extension.

    Parameters
    ----------
    priority_areas : str
        Path to raster of priority areas.
    csi : geodataframe
        Centres de santé.
    epsg : int
        EPSG used to compute distances in meters.
    min_population : int
        Min. population served for a potential area.

    Return
    ------
    geodataframe
        Potential areas with added metrics such as population served
        and distance to nearest CSI.
    """
    with rasterio.open(priority_areas) as src:
        data = src.read(1)
        data[data < min_population] = 0
        data[data >= min_population] = 1
        dst_transform = src.transform

    # Vectorize continuous hot spots in priority areas raster
    # i.e. areas with pixels > min_population
    geoms = []
    for polygon, value in rasterio.features.shapes(
        data.astype("uint8"), transform=dst_transform
    ):
        if value:
            geoms.append(shape(polygon).simplify(dst_transform.a))

    potential_areas = gpd.GeoDataFrame(index=[i for i in range(0, len(geoms))])
    potential_areas["geometry"] = geoms
    potential_areas.crs = "EPSG:4326"

    # Compute area in km2
    potential_areas["area"] = round(
        potential_areas.to_crs(f"EPSG:{epsg}").area * 1e-6, 2
    )
    potential_areas = potential_areas[potential_areas["area"] >= 10]

    # Get maximum population served in polygon
    stats = zonal_stats(
        [geom for geom in potential_areas.geometry],
        raster=priority_areas,
        stats=["max"],
    )
    potential_areas["max_population_served"] = [int(s["max"]) for s in stats]
    distance_csi = []
    csi_ = csi.to_crs(f"EPSG:{epsg}")
    csi_ = csi_[csi_.geometry.is_valid]
    for area in potential_areas.to_crs(f"EPSG:{epsg}").geometry:
        distance_csi.append(round(csi_.distance(area.centroid).min() / 1000, 2))
    potential_areas["distance_nearest_csi"] = distance_csi

    return potential_areas


def analyse_cs(
    cs: gpd.GeoDataFrame, csi: gpd.GeoDataFrame, epsg: int = 32632
) -> gpd.GeoDataFrame:
    """Analyse cases de santé for extension.

    Parameters
    ----------
    cs : geodataframe
        Cases de santé.
    csi : geodataframe
        Centres de santé.
    epsg : int, optional
        EPSG code of the metric CRS.

    Return
    ------
    geodataframe
        Potential CS with metrics such as population served and distance to
        nearest CSI.
    """
    distance_csi = []
    csi_ = csi.to_crs(f"EPSG:{epsg}")
    csi_ = csi_[csi_.geometry.is_valid]
    for fosa in cs.to_crs(f"EPSG:{epsg}").geometry:
        if fosa:
            distance_csi.append(round(csi_.distance(fosa).min() / 1000, 2))
        else:
            distance_csi.append(None)
    cs["distance_nearest_csi"] = distance_csi
    return cs[cs.distance_nearest_csi >= 15]


@Gooey(
    program_name="Module d'extension de la couverture santé",
    default_size=(800, 600),
    required_cols=1,
    optional_cols=1,
    navigation="tabbed",
    progress_regex=r"^progress: (\d+)%$"
)
def app():

    parser = GooeyParser(description="Module d'extension de la couverture santé")
    general = parser.add_argument_group("Général")
    fosa = parser.add_argument_group("Formations sanitaires")
    dhis2 = parser.add_argument_group("DHIS2")
    modeling = parser.add_argument_group("Modélisation")
    worldpop = parser.add_argument_group("WorldPop")

    general.add_argument(
        "--districts",
        metavar="Districts",
        help="Fichier des districts (Shapefile, Geopackage, ou GeoJSON)",
        required=True,
        widget="FileChooser",
    )

    fosa.add_argument(
        "--csi",
        metavar="Centres de santé",
        help="Fichier des centres de santé (Shapefile, Geopackage, ou GeoJSON)",
        required=True,
        widget="FileChooser",
    )

    fosa.add_argument(
        "--cs",
        metavar="Cases de santé",
        help="Fichier des cases de santé (Shapefile, Geopackage, ou GeoJSON)",
        required=True,
        widget="FileChooser",
    )

    dhis2.add_argument(
        "--dhis2-instance", metavar="Instance DHIS2", help="URL de l'instance DHIS2"
    )

    dhis2.add_argument(
        "--dhis2-username", metavar="Utilisateur DHIS2", help="Nom d'utilisateur DHIS2"
    )

    dhis2.add_argument(
        "--dhis2-password",
        metavar="Mot de passe DHIS2",
        help="Mot de passe DHIS2",
        widget="PasswordField",
    )

    general.add_argument(
        "--output-dir",
        metavar="Dossier de sortie",
        help="Dossier où enregistrer les résultats",
        required=True,
        widget="DirChooser",
    )

    modeling.add_argument(
        "--min-distance-csi",
        metavar="Distance minimum",
        type=int,
        help="Distance minimum entre un nouveau CSI et un CSI existant (en mètres)",
        default=15000,
    )

    modeling.add_argument(
        "--max-distance-served",
        metavar="Distance desservie",
        type=int,
        help="Rayon autour d'un CSI autour duquel la population est desservie (en mètres)",
        default=5000,
    )

    modeling.add_argument(
        "--min-population",
        metavar="Population desservie minimum",
        type=int,
        help="Population desservie minimum pour qu'une CS soit considérée pour conversion vers un CSI",
        default=5000,
    )

    general.add_argument("--country", metavar="Pays", help="Code pays", type=str, default="NER")

    general.add_argument("--epsg", metavar="EPSG", help="EPSG code", type=int, default=32632)

    worldpop.add_argument(
        "--un-adj",
        metavar="UN ajustement",
        action="store_true",
        help="Utiliser les données WorldPop ajustées aux prédictions des Nations Unies",
        default=True,
    )

    worldpop.add_argument(
        "--unconstrained",
        metavar="Non-contraint",
        action="store_true",
        help="Utiliser les données WorldPop non-contraintes",
        default=False,
    )

    args = parser.parse_args()

    coverage(
        districts=args.districts,
        csi=args.csi,
        cs=args.cs,
        population=None,
        output_dir=args.output_dir,
        min_population=args.min_population,
        min_distance_from_csi=args.min_distance_csi,
        max_distance_served=args.max_distance_served,
        country=args.country,
        epsg=args.epsg,
        un_adj=args.un_adj,
        constrained=False if args.unconstrained else True,
    )


if __name__ == "__main__":
    app()

# Extension de la couverture sanitaire au Niger

Le module d'extension de la couverture sanitaire au Niger permet le calcul automatique d'un ensemble de métriques et d'analyses. L'objectif est triple:

* Identifier les zones du territoires candidates à l'implantation d'un CSI
* Identifier les conversions possibles des cases de santé vers un CSI
* Identifier les CSI surchargés

## Méthodologie

Le module repose sur quatre sources de données:

1. Une carte de **population** (Worldpop est utilisé automatiquement à défaut)
2. Un fichier de **districts** (Shapefile, GeoPackage, GeoJSON)
3. Un fichier de **centres de santé**, ou CSI (Shapefile, GeoPackage, GeoJSON)
4. Un fichier de **cases de santé** (Shapefile, GeoPackage, GeoJSON)

Le module calcule l'ensemble des métriques de manière indépendante pour chaque district: la population de l'autre côté d'une frontière d'un district n'est pas prise en compte lors du calcul de la population desservie par un CS.

![Zones d'extension potentielles](documentation/images/identify_areas.png)  
*Image: Zones d'extension potentielles et population desservie*

Les zones d'extension potentielles sont des espaces caractérisés par une importante population mais une absence de CSI, i.e. une population desservie supérieure à `MIN_POPULATION_SERVED` (5,000 par défaut) et à plus de `MIN_DISTANCE_FROM_CSI` (15 km par défaut) d'un CSI existant.

![Conversions possibles](documentation/images/identify_cs.png)  
*Image: Conversions possibles de cases de santé et population desservie*

Les cases de santé à conversion potentielle sont identifiées par une forte population desservie (5,000 par défaut) et une localisation à plus de `MIN_DISTANCE_FROM_CSI` (15 km par défaut) d'un CSI.

## Installation

Cloner le dépôt Github et installer les dépendences Python :

``` bash
git clone https://github.com/BLSQ/health-coverage-extension
cd health-coverage-extension
pip install -r requirements.txt
```

NB: GDAL doit également être installé.

## Utilisation

Le module est un script python utilisable en ligne de commande.

```
python -m healthcoverage --help
```

```
usage:
  python -m healthcoverage <arguments>

description:
  Module d'extension de la couverture sanitaire.

arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Dossier de sortie.
  --districts DISTRICTS
                        Districts de santé (GPKG, SHP, GeoJSON).
  --csi CSI             Centres de santé (GPKG, SHP, GeoJSON).
  --cs CS               Cases de santé (GPKG, SHP, GeoJSON).
  --population POPULATION
                        Carte de population (GeoTIFF).
  --min-distance-csi MIN_DISTANCE_CSI
                        Min. distance d'un CSI existant (default: 15000 m).
  --max-distance-served MAX_DISTANCE_SERVED
                        Distance de population desservie (default: 5000 m).
  --min-population MIN_POPULATION
                        Min. population desservie (default: 5000 m).
  --country COUNTRY     Code pays (default: NER).
  --epsg EPSG           CRS utilisé pour calculer les distances (default: 32632).
  --no-un-adj           [Worldpop] Ne pas utiliser l'ajustement UN.
  --unconstrained       [Worldpop] Utiliser le jeu de données non-contraint
```

### Arguments requis

* `--output-dir`  
Dossier de sortie où l'ensemble des fichiers produits seront sauvegardés.
* `--districts`  
Fichier vectoriel (shapefile, geopackage, geojson) contenant les géométries des districts du pays
* `--csi`  
Fichier vectoriel (shapefile, geopackage, geojson) contenant les géométries des CSI
* `--cs`  
Fichier vectoriel (shapefile, geopackage, geojson) contenant les géométries des cases de santé

### Arguments optionnels

* `--population`  
Carte de population (GeoTIFF). Si non-renseignée, alors la carte sera automatiquement téléchargée depuis Worldpop.
* `--min-distance-csi` (default: 15,000)  
La distance minimum requise à n'importe quel CSI existant pour qu'une zone ou case de santé puisse être considérée pour une extension (en mètres).
* `--max-distance-served` (default: 5,000)  
Rayon autour duquel la population est considérée comme desservie (en mètres).
* `--min-population` (default: 5,000)  
Population desservie minimum pour qu'une zone ou case de santé soit considérée pour une extension.
* `--country` (default: NER)  
Code pays en 3 lettres. Utilisé uniquement pour le téléchargement automatique des données Worldpop.
* `--epsg` (default: 32632)  
Code EPSG d'un CRS métrique.
* `--no-un-adj`  
[Worldpop] Cette option permet de télécharger la version des données sans ajustement de la population.
* `--unconstrained`  
[Worldpop] Cette option permet de télécharger la version non-contraintes des données de population.

## Image Docker

Une [image Docker](https://hub.docker.com/r/blsq/health-coverage-extension) est également disponible. Pour l'utiliser:

1. Installer [Docker Desktop](https://docs.docker.com/desktop/windows/install/)
2. Démarrer Docker Desktop
3. Lancer l'invite de commandes

### Exemples

```
docker run -v "C:\data\extension:/data" blsq/health-extension-coverage \
    --districts /data/input/districts.gpkg \
    --csi /data/input/csi.gpkg \
    --cs /data/input/cs.gpkg
    --output-dir /data/output
```

Cette commande va produire les fichiers suivants dans `C:\data\extension\output`:
* Résultats de l'analyse :
    * `potential_areas.gpkg`  
    Zones d'extension potentielles (géométries, population desservie, distance au CSI le plus proche).
    * `potential_cs.gpkg`  
    CS à conversion potentielle (géométries, population desservie, distance au CSI le plus proche).
* Données brutes :
    * `csi_population.gpkg`  
    Fichier CSI original avec population desservie.
    * `cs_population.gpkg`  
    Fichier CS original avec population desservie.
    * `priority_areas.gpkg`  
    Raster avec population desservie pour chaque pixel de 100 m.

NB: L'option `-v "C:\data\extension:/data"` permet de mapper le répertoire `C:\data\extension` sur la machine hôte au répertoire `/data` du container Docker.

import os
import shutil
import subprocess
import tempfile

from gooey import Gooey, GooeyParser


@Gooey(
    program_name="Module d'extension de la couverture santé",
    default_size=(800, 600),
    required_cols=1,
    optional_cols=1,
    navigation="tabbed",
)
def main():

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
        help="Distance minimum entre un nouveau CSI et un CSI existant (en mètres)",
        default=15000,
    )

    modeling.add_argument(
        "--max-distance-served",
        metavar="Distance desservie",
        help="Rayon autour d'un CSI autour duquel la population est desservie (en mètres)",
        default=5000,
    )

    modeling.add_argument(
        "--min-population",
        metavar="Population desservie minimum",
        help="Population desservie minimum pour qu'une CS soit considérée pour conversion vers un CSI",
        default=5000,
    )

    general.add_argument("--country", metavar="Pays", help="Code pays", default="NER")

    general.add_argument("--epsg", metavar="EPSG", help="EPSG code", default=32632)

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

    _docker()

    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="blsq_") as tmp_dir:

        # copy districts, csi, and cs files
        for arg in (args.districts, args.csi, args.cs):

            basedir = os.path.dirname(arg)
            basename = os.path.basename(arg)

            # if shapefile, copy all files
            if arg.lower().endswith(".shp"):
                no_extension = basename.replace(".shp", "").replace(".SHP", "")
                for f in os.listdir(basedir):
                    if f.startswith(no_extension):
                        shutil.copyfile(
                            os.path.join(basedir, f), os.path.join(tmp_dir, f)
                        )

            else:
                shutil.copyfile(arg, os.path.join(tmp_dir, basename))

        os.makedirs(os.path.join(tmp_dir, "output"), exist_ok=True)

        cmd = [
            "docker",
            "run",
            "-v",
            f"{tmp_dir}:/data",
            "blsq/health-coverage-extension",
            "--districts",
            f"/data/{os.path.basename(args.districts)}",
            "--csi",
            f"/data/{os.path.basename(args.csi)}",
            "--cs",
            f"/data/{os.path.basename(args.cs)}",
            "--output-dir",
            "/data/output",
            "--no-progress",
        ]

        print(" ".join(cmd) + "\n")

        if args.min_distance_csi:
            cmd += ["--min-distance-csi", args.min_distance_csi]

        if args.max_distance_served:
            cmd += ["--max-distance-served", args.max_distance_served]

        if args.min_population:
            cmd += ["--min-population", args.min_population]

        if args.country:
            cmd += ["--country", args.country]

        if args.epsg:
            cmd += ["--epsg", args.epsg]

        if not args.un_adj:
            cmd += ["--no-un-adj"]

        if args.unconstrained:
            cmd += ["--unconstrained"]

        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise Exception("Error")

        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copytree(
            os.path.join(tmp_dir, "output"), args.output_dir, dirs_exist_ok=True
        )


def _docker():
    """Check if the docker daemon is available."""
    p = subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL)
    if p.returncode != 0:
        raise FileNotFoundError("Docker n'est pas disponible.")


if __name__ == "__main__":
    main()

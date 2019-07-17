import glob                                             # Read files from folder
from bs4 import BeautifulSoup as bs                     # Parse XML files

DATA = "../dataset/MEDDOCAN-hard"

if __name__ == '__main__':
    # progressbar.streams.wrap_stderr()

    labels = {
        "Datos del paciente",
        "Nombre:",
        "Apellidos:",
        "CIPA:",
        "NHC:",
        "NASS:",
        "Domicilio:",
        "Localidad/ Provincia:",
        "CP:",
        "NHC:",
        "Fecha de nacimiento:",
        "País:",
        "Edad:",
        "Sexo:",
        "Fecha de Ingreso:",
        "Médico:",
        "Remitido por:"}

    # Get all XML files from a folder
    files = glob.glob("{}/**/*.xml".format(DATA), recursive=True)

    # Process all files in the folder
    for file in files:
        print('Preparing file {}'.format(file))

        # Read data file and parse the XML
        with open(file, mode='r', encoding='utf-8-sig') as infile:
            soup = bs(infile, "html.parser")

        # Get the pacient note text
        record = soup.find("text").text

        for label in labels:
            record = record.replace(label, ''.ljust(len(label)))

        soup.find("text").string.replace_with(record)

        with open(file, mode='w', encoding='utf-8-sig') as infile:
            infile.write(soup.prettify())

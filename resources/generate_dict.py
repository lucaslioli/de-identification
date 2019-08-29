import csv

FILEPATH = "resources/gazetteer/dados.csv"
COLNAME = "NOME"

if __name__ == '__main__':

    allNames = {}

    prep={"da", "de", "dos", "e"}

    with open(FILEPATH) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            row[COLNAME] = row[COLNAME].replace("\"", "").split()
            row[COLNAME] = [w for w in row[COLNAME] if w not in prep]
            
            for name in row[COLNAME]:
                if(name not in allNames.keys()):
                    allNames[name] = 1
                else:
                    allNames[name] = allNames[name]+1

    for key, value in allNames.items():
        print ("{}, {}".format(key, value))

    exit()
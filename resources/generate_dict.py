import csv

FILEPATH = "resources/gazetteer/dados.csv"
COLNAME = "NOME"

if __name__ == '__main__':

    stNames = []
    ndNames = []

    prep={"de", "da"}

    with open(FILEPATH) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            row[COLNAME] = row[COLNAME].replace("\"", "").split()
            row[COLNAME] = [w for w in row[COLNAME] if w not in prep]

            first = row[COLNAME][0]
            last = row[COLNAME][1:]
            
            if(first not in stNames):
                stNames.append(first)
            
            for name in last:
                if(name not in ndNames):
                    ndNames.append(name)

    print(stNames)
    print(ndNames)

    exit()
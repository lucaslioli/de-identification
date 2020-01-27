import csv
import json

from statistics import mean

FILEPATH = "dataset/DADOS_0.csv"
COLNAME = "EVOLUCOES"

if __name__ == '__main__':

    data = []
    avg_size = []
    avg_words = []

    with open(FILEPATH, mode="r", encoding="ISO-8859-1") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row[COLNAME] is not None:
                data.append({
                    'text': row[COLNAME],
                    'meta': {
                        'size': len(row[COLNAME]),
                        'words': len(row[COLNAME].split())
                    }
                })

                avg_size.append(len(row[COLNAME]))
                avg_words.append(len(row[COLNAME].split()))

    json_data = json.dumps(data, sort_keys=True, indent=4)

    print(json_data)

    # print("AVG Size: ", mean(avg_size))
    # print("AVG Words: ", mean(avg_words))

    exit()
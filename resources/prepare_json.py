import csv
import json

FILEPATH = "dataset/DADOS_0-evolucoes"
COLNAME = "EVOLUCOES"

if __name__ == '__main__':

    notes = []

    with open(FILEPATH+".csv", mode="r", encoding="ISO-8859-1") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row[COLNAME] is not None:
                notes.append({
                    'text': row[COLNAME],
                    'meta': {
                        'code': row['CODIGO_PACIENTE'],
                        'length': len(row[COLNAME]),
                        'words': len(row[COLNAME].split())
                    }
                })

        csvfile.close()

    patients = {}
    avg_length = 0
    avg_words = 0

    for note in notes:
        if note['meta']['code'] not in patients.keys():
            patients[note['meta']['code']] = 0
        patients[note['meta']['code']] += 1

        del(note['meta'])

        avg_length += len(note['text'])
        avg_words += len(note['text'].split())

    # print("Evolutions by patient:")
    # print(json.dumps(patients, indent=2))

    print("\nStatistics:\n")

    print("Number of Evolutions: {}".format(len(notes)))
    print("Number of Patients: {}".format(len(patients)))
    print("Maximum evolution per patient: {}".format(max(patients.values())))

    print("Average of Words: {}".format(int(avg_words/len(notes))))
    print("Average length: {}".format(int(avg_length/len(notes))))

    json_notes = json.dumps(notes, separators=('\n', ':'))

    with open(FILEPATH+".json", mode='w', encoding="utf-8") as jsonfile:
        jsonfile.write(json_notes[1:-1])

        print("\nJSON file saved as: {}.json".format(FILEPATH))

        jsonfile.close()

    exit()
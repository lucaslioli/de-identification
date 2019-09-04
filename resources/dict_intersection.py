
DICT_MED = "resources/gazetteer/BRHIM/medicamentos.txt"
DICT_NAME = "resources/gazetteer/BRHIM/pacientes.txt"

def intersection(list1, list2): 
    list3 = [value for value in list1 if value in list2] 
    return list3 

if __name__ == '__main__':

    file = open(DICT_MED, "r")
    list1 = file.read().splitlines()
    list1 = [elem.lower() for elem in list1]

    file = open(DICT_NAME, "r")
    list2 = file.read().splitlines()
    list2 = [elem.lower() for elem in list2]
    
    print(intersection(list1, list2))

    exit()
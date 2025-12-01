import re

def turnListToString(li):
    """joins attributenames of rhs to string with comma seperation"""
    s = ", "
    return s.join([ l.replace(",", " " ) for l in li])

def listToLabel(head_labels):
    """builds lookuptable to transfrom columnid to headlabel"""
    ct = {}
    for i in range(len(head_labels)):
        ct['column' + str(i+1)] = head_labels[i]
    return ct

def lineToDataStruct(lineElements, columnNameToLabel):
    """transforms csv line from alorithm output to list of attributes and the attribute they predict"""
    index_dependant = 0
    for element in lineElements:
        if element.startswith('"dependant":'):
            break
        index_dependant += 1

    rhs = lineElements[index_dependant + 1].split(':')[1]
    for removeChar in ['"','[',']','{','}']:
        rhs = rhs.replace(removeChar, '')
    determinant_index = 0
    rhs = columnNameToLabel[rhs]

    for element in lineElements:
        if element.startswith('"determinant'):
            #print('found at ' + str(determinant_index))
            break
        determinant_index += 1

    X = [] # RHS
    for s in lineElements[determinant_index + 1: index_dependant]:
        match = re.search(r'"columnIdentifier":"(column\d+)"', s)
        if match:
            result = match.group(1)
            X.append(columnNameToLabel[result])

    return X,rhs

def readin_raw(path = '', head_string = '', seperator_head_string =',' ):
    """reads in the fd algorithms output and returns a list"""
    labels = head_string.split(seperator_head_string)
    label_lookup = listToLabel(labels)
    print(label_lookup)
    data = []
    with open(path, 'r') as file:
        for line in file:
            lhs, rhs = lineToDataStruct(line.strip().split(","), label_lookup)
            data.append( (lhs, rhs) )
    return data

def find_fds_for_rhs(target_rhs, data,sorted=False, lhs_validator = lambda x: True, lhs_size_min = -1, lhs_size_max = -1):
    """filters based on rhs and allows to sort and ensure min and max on lhs"""
    results = []
    for tuple in data:
        lhs, rhs = tuple
        if rhs == target_rhs and lhs_validator(lhs):
            results.append(tuple)
    if sorted:
        results.sort(key=lambda x: len(x[0]))
    size_filter = lambda lhs: (lhs_size_min < 0 or len(lhs) >= lhs_size_min ) and (lhs_size_max < 0 or len(lhs) <= lhs_size_max)
    return [t for t in results if size_filter(t[0])]

def fuzzy_filter(data, exclude_list = [], include_list = []):
    """method to filter out based on substrings on lhs"""
    excludes_promise = lambda lhs: (not exclude_list) or (not any([any([ex.lower() in label.lower() for ex in exclude_list]) for label in lhs]))
    include_promise = lambda lhs: (not include_list) or (all([any(inc.lower() in label.lower() for label in lhs) for inc in include_list])) 
    filtered = [t for t in data if excludes_promise(t[0]) and include_promise(t[0])]
    return sorted(filtered, key=lambda x: len(x[0]))


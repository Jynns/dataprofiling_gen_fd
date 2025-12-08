import re
import pandas as pd
import os


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
    assert isinstance(exclude_list, list), "exclude_list must be a list"
    assert isinstance(include_list, list), "include_list must be a list"
    excludes_promise = lambda lhs: (not exclude_list) or (not any([any([ex.lower() in label.lower() for ex in exclude_list]) for label in lhs]))
    include_promise = lambda lhs: (not include_list) or (all([any(inc.lower() in label.lower() for label in lhs) for inc in include_list])) 
    filtered = [t for t in data if excludes_promise(t[0]) and include_promise(t[0])]
    return sorted(filtered, key=lambda x: len(x[0]))

def convertToDataFrame(fds, table_name, is_approx, null_eq_null, is_genuine):
    n = len(fds)

    lhs, rhs = zip(*fds)  # unpack pairs

    wb_dataframe = pd.DataFrame({
        "lhs": [turnListToString(x) for x in lhs],
        "rhs": list(rhs),
        "tablename": [table_name + ".csv"] * n,
        "is_approx": [is_approx] * n,
        "null_eq_nul": [null_eq_null] * n,
        "is_genuine": [is_genuine] * n,
    })   
    return wb_dataframe

_FILENAME_TRAININGDATA = 'dataset.csv'

def writeBackTrainingData(df: pd.DataFrame):
    """
    Writes or appends training data to training.csv.

    Behavior:
    - If training.csv does NOT exist:
        * Write df directly to training.csv

    - If training.csv DOES exist:
        * Read it in
        * If existing has 'id' but new df doesn't:
            - Check that all non-id columns match
            - Add an 'id' column to new df, continuing from max existing id
        * Check that columns now match exactly
        * Check if any of the new 'tablename' values already exist in the file:
            - If yes: do NOT write anything, just return
        * Otherwise:
            - Concatenate old + new data
            - Overwrite training.csv with the concatenated dataframe
    """

    file_path = os.path.join('.', _FILENAME_TRAININGDATA)

    # No existing file: just write the new df
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
        return

    # Existing file case
    try:
        existing_df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read existing {_FILENAME_TRAININGDATA}: {e}")

    new_df = df.copy()

    # ---- SPECIAL CASE: existing has 'id', new df doesn't ----
    if 'id' in existing_df.columns and 'id' not in new_df.columns:
        existing_without_id = [c for c in existing_df.columns if c != 'id']

        # Check that all non-id columns match exactly
        if existing_without_id == list(new_df.columns):
            # Determine starting id: continue from max existing id if numeric
            if pd.api.types.is_integer_dtype(existing_df['id']):
                start_id = int(existing_df['id'].max()) + 1
            else:
                # Fallback: just use length if id is non-integer type
                start_id = len(existing_df)

            # Add 'id' to new_df
            new_df = new_df.reset_index(drop=True)
            new_df.insert(0, 'id', range(start_id, start_id + len(new_df)))
        else:
            # Columns don't match aside from id → real mismatch
            raise ValueError(
                f"Column mismatch (ignoring 'id'): existing file has {existing_without_id}, "
                f"but new data has {list(new_df.columns)}."
            )

    # ---- FINAL SCHEMA CHECK ----
    if list(existing_df.columns) != list(new_df.columns):
        raise ValueError(
            f"Column mismatch: existing file has {list(existing_df.columns)}, "
            f"but new data has {list(new_df.columns)}."
        )

    # ---- CHECK FOR EXISTING TABLENAMES ----
    if 'tablename' not in existing_df.columns or 'tablename' not in new_df.columns:
        raise ValueError("Both existing and new data must have a 'tablename' column.")

    existing_tablenames = set(existing_df['tablename'].unique())
    new_tablenames = set(new_df['tablename'].unique())

    # If any new tablenames already exist, do nothing and quit
    overlapping = existing_tablenames.intersection(new_tablenames)
    if overlapping:
        print(f"Tablenames {overlapping} already exist. No data written.")
        return

    # ---- CONCATENATE AND WRITE BACK ----
    try:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate dataframes: {e}")

    combined_df.to_csv(file_path, index=False)


# Open the file containing polynomial groupings and store results
def extract_poly_mappings():
    f = open("FacePolyMappings.data", "r")
    field_name = ""
    mappings = dict()
    for line in f:
        if line.startswith("#") or len(line.strip()) == 0:
            continue
            
        field_start_loc = line.find("--|")
        field_end_loc = line.find("|--")
        if field_start_loc != -1:
            field_name = line[field_start_loc+3:field_end_loc].strip()
            mappings[field_name] = []
        else:
            mappings[field_name].append(line.strip())
    
    print(mappings)
    return mappings


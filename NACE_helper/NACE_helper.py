def get_level_1_nace(x):
    x = float(x)
    
    if 1 <= x <= 3.22:
        return 'A'
    elif 5 <= x <= 9.9:
        return 'B'
    elif 10 <= x <= 33.20:
        return 'C'
    elif 35 <= x <= 35.30:
        return 'D'
    elif 36 <= x <= 39:
        return 'E' 
    elif 41 <= x <= 43.99:
        return 'F'
    elif 45 <= x <= 47.99:
        return 'G'
    elif 49 <= x <= 53.2:
        return 'H'
    elif 55 <= x <= 56.3:
        return 'I'
    elif 58 <= x <= 63.99:
        return 'J'
    elif 64 <= x <= 66.3:
        return 'K'
    elif 68 <= x <= 68.32:
        return 'L'
    elif 69 <= x <= 75:
        return 'M' 
    elif 77 <= x <= 82.99:
        return 'N'
    elif 84 <= x <= 84.3:
        return 'O'
    elif 85 <= x <= 85.6:
        return 'P'
    elif 86 <= x <= 88.99:
        return 'Q'
    elif 90 <= x <= 93.29:
        return 'R'
    elif 94 <= x <= 96.09:
        return 'S' 
    elif 97 <= x <= 98.2:
        return 'T'
    elif x == 99:
        return 'U'
    else:
        return None


def get_nace_level(nace_code):

    try: 
        float(nace_code)
    except ValueError: 
        return 1

    if float(nace_code) < 10:
        nace_code = "0" + str(nace_code)
    original_level = len(nace_code.replace(".", ""))
    return original_level


def get_all_level(nace_code, df_nace_codes_descriptions= None): 

    if isinstance(nace_code, int) or isinstance(nace_code, float):
        if nace_code < 10:
            nace_code = "0" + str(nace_code)
        
    nace_code = str(nace_code)

    init_level = None 

    if nace_code.isalpha(): 
        init_level = 1
    elif "." not in nace_code: 
        init_level = 2
    else: 
        init_level = len(nace_code.split(".")[1])+2

    levels = {init_level: nace_code}

    for level in range(1,init_level):     
        if level == 3: 
            levels[level] = str(nace_code)[:-1]
        if level == 2: 
            levels[level] = str(int(nace_code[:2]))
        if level == 1: 
            levels[level] = get_level_1_nace(float(nace_code))

    return levels



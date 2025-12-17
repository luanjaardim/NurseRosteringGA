import xml.etree.ElementTree as ET

def xml_to_dict(element):
    print(element.tag, element.attrib, element.text)
    if len(element) == 0:
        return element.text
    else:
        return {element.tag: {child.tag: xml_to_dict(child) for child in element}}

def parse(file_path):
    import os
    import xml.etree.ElementTree as ET
    import json
    if not os.path.exists(file_path):
        raise Exception("Instance file does not exists")

    return json.dumps(xml_to_dict(ET.parse(file_path).getroot()), indent=4)

def parse_txt(file_path):
    import os
    if not os.path.exists(file_path):
        raise Exception("Instance file does not exists")

    data = {
        'len_day': 0,
        'shifts': [],
        'staff': [],
        'days_off': [],
        'requests': [],
        'requests_off': [],
        'cover' : []
    }
    lines = [ line.strip() for line in open(file_path).readlines() if (line.strip() and (not line.strip().startswith('#'))) ]
    i = 0
    section = ""
    while i < len(lines):
        e = lines[i]
        if "SECTION" in e:
            section=e[len("SECTION_"):]
            i+=1
            continue

        columns = e.split(',')
        match section:
            case "HORIZON":
                data['len_day'] = int(e)
            case "SHIFTS":
                s_id, minutes_len, cannot_follow = columns
                cannot_follow = [ s for s in cannot_follow.split('|') if s ]
                data['shifts'].append({'id': s_id, 'len': int(minutes_len), 'cannot_follow': cannot_follow})
            case "STAFF":
                st_id, max_shifts, max_minutes, min_minutes, max_consec_shifts, min_consec_shifts, min_consec_off, max_weekends = columns
                limits = list(map(lambda limit: tuple(limit.split('=')), max_shifts.split('|')))
                data['staff'].append({
                    'id': st_id,
                    'shift_limits': limits,
                    'max_minu': int(max_minutes),
                    'min_minu': int(min_minutes),
                    'max_cons_shifts': int(max_consec_shifts),
                    'min_cons_shifts': int(min_consec_shifts),
                    'min_cons_off': int(min_consec_off),
                    'max_weekends': int(max_weekends)
                })
            case "DAYS_OFF":
                data['days_off'].append({'staff_id': columns[0], 'days_off': columns[1:]})
            case "SHIFT_ON_REQUESTS":
                st_id, day, sh_id, weight = columns
                data['requests'].append({'staff_id': st_id, 'day': int(day), 'shift_id': sh_id, 'weight': int(weight)})
            case "SHIFT_OFF_REQUESTS":
                st_id, day, sh_id, weight = columns
                data['requests_off'].append({'staff_id': st_id, 'day': int(day), 'shift_id': sh_id, 'weight': int(weight)})
            case "COVER":
                day, sh_id, requirement, weight_under, weight_over = columns
                data['cover'].append({'day': int(day), 'id': sh_id, 'requirement': int(requirement), 'weight_under': int(weight_under), 'weight_over': int(weight_over)})
        i+=1
    return data


def parse_roster(file_path): #[ (c.tag, c.text) for c in ass ]
    tree = ET.parse(file_path)
    root = tree.getroot()

    assignments = []

    for emp in root.findall('Employee'):
        staff_id = emp.attrib['ID']

        for ass in emp.findall('Assign'):
            day = int(ass.find('Day').text)
            shift = ass.find('Shift').text

            assignments.append({
                'staff_id': staff_id,
                'day': day,
                'shift_id': shift
            })

    #     [
    #   {'staff_id': 'E1', 'day': 0, 'shift_id': 'D'},
    #   {'staff_id': 'E2', 'day': 0, 'shift_id': 'N'},
    #   ...
    #     ]
    return assignments


def roster_to_flat_individual(roster, data):
    """
    Converte a solção do .roster para os vetores dia por dia usado pelo GA
    """
    staff_index  = { st['id']: i for i, st in enumerate(data['staff']) }

    n_days = data['len_day']

    #Verificar quando tiver mais de um turno
    schedule = [[] for _ in range(n_days)]

    for staf_a in roster:

        day = staf_a['day']
        st_id = staf_a['staff_id']

        schedule[day].append(staff_index[st_id])

    return schedule

if __name__ == "__main__":
    d = parse_txt('Instance1.txt')
    import json
    print(json.dumps(d, indent=4))

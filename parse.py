from textwrap import indent


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
                    'max_cons_off': int(min_consec_off),
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

if __name__ == "__main__":
    d = parse_txt('Instance1.txt')
    import json
    print(json.dumps(d, indent=4))

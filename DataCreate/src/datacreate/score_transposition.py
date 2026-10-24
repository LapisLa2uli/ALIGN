"""Read a uniform MusicXML transposition without loading any ML runtime."""
import xml.etree.ElementTree as ET


def score_transposition(score_path, override=None):
    if override is not None:
        return int(override)
    tree = ET.parse(score_path)
    values = set()
    for element in tree.iter():
        if element.tag.split('}')[-1] != 'transpose':
            continue
        fields = {c.tag.split('}')[-1]: c.text for c in element}
        values.add(int(fields.get('chromatic', 0)) + 12 * int(fields.get('octave-change', 0)))
    if len(values) > 1:
        raise ValueError('Changing instrument transpositions require separate solo takes')
    return next(iter(values), 0)

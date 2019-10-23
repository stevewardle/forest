from datetime import datetime
import collections

import numpy as np
import mule

from forest import geo


def empty_image():
    return {
        "x": [],
        "y": [],
        "dw": [],
        "dh": [],
        "image": [],
        "name": [],
        "units": [],
        "valid": [],
        "initial": [],
        "length": [],
        "level": []
    }


def _to_datetime(d):
    if isinstance(d, datetime):
        return d
    elif isinstance(d, str):
        try:
            return datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
    elif isinstance(d, np.datetime64):
        return d.astype(datetime)
    else:
        raise Exception("Unknown value: {}".format(d))


def coordinates(valid_time, initial_time, pressures, pressure):
    valid = _to_datetime(valid_time)
    initial = _to_datetime(initial_time)
    hours = (valid - initial).total_seconds() / (60*60)
    length = "T{:+}".format(int(hours))
    if (len(pressures) > 0) and (pressure is not None):
        level = "{} hPa".format(int(pressure))
    else:
        level = "Surface"
    return {
        'valid': [valid],
        'initial': [initial],
        'length': [length],
        'level': [level]
    }

# TODO: This logic should move to a "Group" concept.
def _load(pattern):
    """Return all the valid field objects that can be loaded
    from the given filename pattern."""
    # Load the file object (this could return one of several things
    # depending on what the exact file is, but we'll worry about
    # that later if we need to)
    umf = mule.load_umfile(pattern)

    # Get the list of Field objects
    fields = umf.fields

    # Filter the list to remove padding fields and other types
    # of special field we might not expect to display correctly
    for field in list(fields):
        if field.lbrel not in (2, 3):
            fields.remove(field)
    assert len(fields) > 0

    # Create names for the fields; for a UM file
    def _fieldname(field):
        return (f"{field.stash.name} ({field.lbuser4}) "
                f"- T+{field.lbft} - lvl{field.lblev} "
                f"- proc{field.lbproc}")
        
    name_counts = collections.Counter(_fieldname(field) for field in fields)
    duplicate_names = {name for name, count in name_counts.items()
                       if count > 1}

    # Map names (with numeric suffixes for duplicates) to field
    duplicate_counts = collections.defaultdict(int)
    field_mapping = {}
    for field in fields:
        name = _fieldname(field)
        if name in duplicate_names:
            duplicate_counts[name] += 1
            name += f' ({duplicate_counts[name]})'
        field_mapping[name] = field
    return field_mapping, umf


class ImageLoader:
    def __init__(self, label, pattern):
        self._fields, self._file = _load(pattern) 
        self._label = label

    def image(self, state):
        field = self._fields[state.variable]

        def _create_coord(start, step, n_points):
            return np.arange(1, n_points + 1)*step + start

        lats = _create_coord(field.bzy, field.bdy, field.lbrow)
        lons = _create_coord(field.bzx, field.bdx, field.lbnpt)        

        data = field.get_data()

        shift_by = np.sum(lons > 180.0)

        lons[lons > 180.0] -= 360.0
        lons = np.roll(lons, shift_by)
        data = np.roll(data, shift_by, axis=1)
     
        data = geo.stretch_image(lons, lats, data)

        data.update(coordinates(state.valid_time, state.initial_time,
                                state.pressures, state.pressure))
        data.update({
                'name': [self._label],
                'units': ["UM"],
        })

        return data


class Navigator:
    def __init__(self, paths):
        self._fields, self._file = _load(paths[0])

    def variables(self, pattern):
        return list(self._fields.keys())

    def initial_times(self, pattern, variable=None):
        flh = self._file.fixed_length_header
        return [datetime(*flh.raw[21:27])]

    def valid_times(self, pattern, variable, initial_time):
        return [datetime(*self._fields[variable].raw[1:7])]

    def pressures(self, pattern, variable, initial_time):
        return [self._fields[variable].lblev]

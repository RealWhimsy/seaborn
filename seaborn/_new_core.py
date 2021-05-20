from __future__ import annotations
from typing import Union, Optional, List, Dict, Tuple
from collections.abc import Hashable, Sequence, Mapping, Sized
from numbers import Number

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series, Index


# TODO ndarray can be the numpy ArrayLike on 1.20+ (?)
# TODO pandas typing (from data-science-types?) doesn't like Number
Vector = Union[Series, Index, ndarray, Sequence, Number]

# TODO Should we define a DataFrame-like type that is Union[DataFrame, Mapping]?


class Plot:

    data: PlotData  # TODO possibly should be private?
    layers: List[Layer]  # TODO probably should be private?

    def __init__(
        self,
        data: Optional[Union[DataFrame, Mapping]] = None,
        **variables: Optional[Union[Hashable, Vector]],
    ):

        # Note that we can't assume wide-form here if variables does not contain x or y
        # because those might get assigned in long-form fashion per layer.

        self.data = PlotData(data, variables)
        self.layers = []

    def add(
        self,
        mark: Mark,
        stat: Stat = None,
        data: Optional[Union[DataFrame, Mapping]] = None,
        variables: Optional[Dict[str, Optional[Union[Hashable, Vector]]]] = None,
    ) -> Plot:

        # TODO what if in wide-form mode, we convert to long-form
        # based on the transform that mark defines?
        layer_data = self.data.update(data, variables)

        if stat is None:
            stat = mark.default_stat

        self.layers.append(Layer(layer_data, mark, stat))

        return self

    # TODO problem with "draw" meaning something specific in mpl?
    def draw(self) -> Plot:

        # TODO one option is to loop over the layers here and use them to
        # initialize and scaling/mapping we need to do (using parameters)
        # possibly previously set and stored through calls to map_hue etc.

        # TODO or something like this
        for layer in self.layers:
            self._draw_layer(layer)

        return self

    def _draw_layer(self, layer):

        # Roughly ...

        # TODO where does this method come from?
        data = self.as_numeric(layer.data)

        # TODO who owns the groupby logic?
        data = self.stat(data)

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        data = self.invert_scale(data)

        # Something like this?
        layer.mark._draw(data)  # do we pass ax (and/or facets?! here?)

    def show(self) -> Plot:

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise
        import matplotlib.pyplot as plt
        self.draw()
        plt.show()

        return self

    def save(self) -> Plot:  # or to_file or similar to match pandas?

        return self

    def _repr_html_(self) -> str:

        html = ""

        return html


# Do we want some sort of generator that yields a tuple of (semantics, data,
# axes), or similar?  I guess this is basically the existing iter_data, although
# currently the logic of getting the relevant axes lives externally (but makes
# more sense within the generator logic). Where does this iteration happen? I
# think we pass the generator into the Mark.plot method? Currently the plot_*
# methods define their own grouping variables. So I guess we need to delegate to
# them. But maybe that could be an attribute on the mark?  (Same deal for the
# stat?)


class PlotData:  # TODO better name?

    # How to handle wide-form data here, when the dimensional semantics are defined by
    # the mark? (I guess? that will be most consistent with how it currently works.)
    # I think we want to avoid too much deferred execution or else tracebacks are going
    # to be confusing to follow...

    # With wide-form data, should we allow marks with distinct wide_form semantics?
    # I think in most cases that will not make sense? When to check?

    # I guess more generally, what to do when different variables are assigned in
    # different calls to Plot.add()? This has to be possible (otherwise why allow it)?
    # ggplot allows you to do this but only uses the first layer for labels, and only
    # if the scales are compatible.

    # Who owns the existing VectorPlotter.variables, VectorPlotter.var_levels, etc.?

    frame: DataFrame
    names: Dict[str, Optional[str]]
    _source: Optional[Union[DataFrame, Mapping]]

    def __init__(
        self,
        data: Optional[Union[DataFrame, Mapping]],
        variables: Optional[Dict[str, Union[Hashable, Vector]]],
        # TODO pass in wide semantics?
    ):

        if variables is None:
            variables = {}

        # TODO only specing out with long-form data for now...
        frame, names = self._assign_variables_longform(data, variables)

        self.frame = frame
        self.names = names
        self._source = data

    def update(
        self,
        data: Optional[Union[DataFrame, Mapping]],
        variables: Optional[Dict[str, Union[Hashable, Vector]]],
    ) -> PlotData:

        # TODO name-wise, does update imply an inplace operation in a confusing way?

        # TODO Note a tricky thing here which is that often x/y will be inherited
        # meaning that the variable specification here will look like "wide-form"

        # Inherit the original source of the upsteam data by default
        if data is None:
            data = self._source

        if variables is None:
            variables = {}

        # Passing var=None implies that we do not want that variable in this layer
        disinherit = [k for k, v in variables.items() if v is None]

        # Create a new dataset with just the info passed here
        new = PlotData(data, variables)

        # -- Update the inherited DataFrame and names with this new information

        drop_cols = [k for k in self.frame if k in new.frame or k in disinherit]
        frame = (
            self.frame
            .drop(columns=drop_cols)
            .join(new.frame)  # type: ignore  # thinks frame.join is a Series??
        )

        names = {k: v for k, v in self.names.items() if k not in disinherit}
        names.update(new.names)

        new.frame = frame
        new.names = names

        return new

    def _assign_variables_longform(
        self,
        data: Optional[Union[DataFrame, Mapping]],
        variables: Dict[str, Optional[Union[Hashable, Vector]]]
    ) -> Tuple[DataFrame, Dict[str, Optional[str]]]:
        """Define plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data
            Input data where variable names map to vector values.
        variables
            Keys are seaborn variables (x, y, hue, ...) and values are vectors
            in any format that can construct a :class:`pandas.DataFrame` or
            names of columns or index levels in ``data``.

        Returns
        -------
        frame
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        names
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        Raises
        ------
        ValueError
            When variables are strings that don't appear in ``data``.

        """
        plot_data: Dict[str, Vector] = {}
        var_names: Dict[str, Optional[str]] = {}

        # Data is optional; all variables can be defined as vectors
        if data is None:
            data = {}

        # TODO should we try a data.to_dict() or similar here to more
        # generally accept objects with that interface?
        # Note that dict(df) also works for pandas, and gives us what we
        # want, whereas DataFrame.to_dict() gives a nested dict instead of
        # a dict of series.

        # Variables can also be extracted from the index attribute
        # TODO is this the most general way to enable it?
        # There is no index.to_dict on multiindex, unfortunately
        if hasattr(data, "index"):
            index = data.index.to_frame()  # type: ignore # mypy/1424
        else:
            index = {}

        for key, val in variables.items():

            # Simply ignore variables with no specification
            if val is None:
                continue

            # Try to treat the argument as a key for the data collection.
            # But be flexible about what can be used as a key.
            # Usually it will be a string, but allow numbers or tuples too when
            # taking from the main data object. Only allow strings to reference
            # fields in the index, because otherwise there is too much ambiguity.
            try:
                val_as_data_key = (
                    val in data
                    or (isinstance(val, str) and val in index)
                )
            except (KeyError, TypeError):
                val_as_data_key = False

            if val_as_data_key:

                # We know that __getitem__ will work

                if val in data:
                    plot_data[key] = data[val]  # type: ignore # fails on key: Hashable
                elif val in index:
                    plot_data[key] = index[val]  # type: ignore # fails on key: Hashable
                var_names[key] = str(val)

            elif isinstance(val, str):

                # This looks like a column name but we don't know what it means!

                err = f"Could not interpret value `{val}` for parameter `{key}`"
                raise ValueError(err)

            else:

                # Otherwise, assume the value is itself data

                # Raise when data object is present and a vector can't matched
                if isinstance(data, pd.DataFrame) and not isinstance(val, pd.Series):
                    if isinstance(val, Sized) and len(data) != len(val):
                        val_cls = val.__class__.__name__
                        err = (
                            f"Length of {val_cls} vectors must match length of `data`"
                            f" when both are used, but `data` has length {len(data)}"
                            f" and the vector passed to `{key}` has length {len(val)}."
                        )
                        raise ValueError(err)

                plot_data[key] = val  # type: ignore # fails on key: Hashable

                # Try to infer the name of the variable
                var_names[key] = getattr(val, "name", None)

        # Construct a tidy plot DataFrame. This will convert a number of
        # types automatically, aligning on index in case of pandas objects
        frame = pd.DataFrame(plot_data)  # type: ignore # should allow dict[str, Number]

        # Reduce the variables dictionary to fields with valid data
        names: Dict[str, Optional[str]] = {
            var: name
            for var, name in var_names.items()
            # TODO I am not sure that this is necessary any more
            if frame[var].notnull().any()
        }

        return frame, names


class Stat:

    pass


class Mark:

    # TODO will subclasses overwrite this? Should this be defined elsewhere?
    group_vars: List[str] = ["col", "row", "group"]

    default_stat: Optional[Stat] = None  # TODO or identity?

    pass


class Point(Mark):

    def __init__(self, **kwargs):

        self.kwargs = kwargs

    def _draw(self, plot):  # TODO data_gen is maybe too restrictive a name?

        kws = self.kwargs.copy()

        for keys, data, ax in plot.data_gen(self.group_vars):

            # Define the vectors of x and y positions
            empty = np.full(len(data), np.nan)
            x = data.get("x", empty)
            y = data.get("y", empty)

            # Set defaults for other visual attributes
            kws.setdefault("edgecolor", "w")

            if "style" in data:
                # Use a representative marker so scatter sets the edgecolor
                # properly for line art markers. We currently enforce either
                # all or none line art so this works.
                example_level = self._style_map.levels[0]
                example_marker = self._style_map(example_level, "marker")
                kws.setdefault("marker", example_marker)

            # TODO this makes it impossible to vary alpha with hue which might
            # otherwise be useful? Should we just pass None?
            kws["alpha"] = 1 if self.alpha == "auto" else self.alpha

            # Draw the scatter plot
            points = ax.scatter(x=x, y=y, **kws)

            # Apply the mapping from semantic variables to artist attributes

            if "hue" in self.variables:
                points.set_facecolors(self._hue_map(data["hue"]))

            if "size" in self.variables:
                points.set_sizes(self._size_map(data["size"]))

            if "style" in self.variables:
                p = [self._style_map(val, "path") for val in data["style"]]
                points.set_paths(p)

            # Apply dependant default attributes


class Layer:

    # Does this need to be anything other than a simple container for these attributes?
    # Could use a Dataclass I guess?

    def __init__(self, data: PlotData, mark: Mark, stat: Stat = None):

        self.data = data
        self.mark = mark
        self.stat = stat

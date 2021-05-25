from __future__ import annotations
from typing import Any, Union, Optional, Literal
from collections.abc import Hashable, Sequence, Mapping, Sized
from numbers import Number
from collections import UserString
from datetime import datetime
import warnings

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_datetime64_dtype
import matplotlib as mpl
from matplotlib.colors import Colormap, Normalize

from .palettes import (
    QUAL_PALETTES,
    color_palette,
)
from .utils import (
    get_color_cycle,
    remove_na,
)


# TODO ndarray can be the numpy ArrayLike on 1.20+, which will subsume sequence (?)
Vector = Union[Series, Index, ndarray, Sequence]

PaletteSpec = Optional[Union[str, list, dict, Colormap]]

# TODO Should we define a DataFrame-like type that is DataFrame | Mapping?
# TODO same for variables ... these are repeated a lot.


class Plot:

    _data: PlotData
    _layers: list[Layer]
    _mappings: dict[str, SemanticMapping]  # TODO keys could be Literal

    def __init__(
        self,
        data: Optional[DataFrame | Mapping] = None,
        **variables: Optional[Hashable | Vector],
    ):

        # Note that we can't assume wide-form here if variables does not contain x or y
        # because those might get assigned in long-form fashion per layer.
        # TODO I am thinking about just not supporting wide-form data in this interface
        # and handling the reshaping in the functional interface externally

        self._data = PlotData(data, variables)
        self._layers = []
        self._mappings = {}  # TODO initialize with defaults?

    def on(self) -> Plot:

        # TODO  Provisional name for a method that accepts an existing Axes object,
        # and possibly one that does all of the figure/subplot configuration
        raise NotImplementedError()
        return self

    def add(
        self,
        mark: Mark,
        stat: Stat = None,
        data: Optional[DataFrame | Mapping] = None,
        variables: Optional[dict[str, Optional[Hashable | Vector]]] = None,
    ) -> Plot:

        # TODO what if in wide-form mode, we convert to long-form
        # based on the transform that mark defines?
        layer_data = self._data.update(data, variables)

        if stat is None:
            stat = mark.default_stat

        self._layers.append(Layer(layer_data, mark, stat))

        return self

    def map_hue(
        self,
        palette: Optional[PaletteSpec] = None,
        order: Optional[list] = None,
        norm: Optional[Normalize] = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # ALSO TODO should these be initialized with defaults?
        self._mappings["hue"] = HueMapping(palette, order, norm)
        return self

    def theme(self) -> Plot:

        # TODO We want to be able to use the existing seaborn themeing system
        # to do plot-specific theming
        raise NotImplementedError()
        return self

    def plot(self) -> Plot:

        # TODO a rough sketch ...

        # TODO one option is to loop over the layers here and use them to
        # initialize and scaling/mapping we need to do (using parameters)
        # possibly previously set and stored through calls to map_hue etc.
        # Alternately (and probably a better idea), we could concatenate
        # the layer data and then pass that to the Mapping objects to
        # set them up. Note that if strings are passed in one layer and
        # floats in another, this will turn the whole variable into a
        # categorical. That might make sense but it's different from if you
        # plot twice once with strings and then once with numbers.
        # Another option would be to raise if layers have different variable
        # types (this is basically what ggplot does), but that adds complexity.

        all_data = pd.concat([layer.data.frame for layer in self._layers])

        # TODO This feels messy here. Should be in a separate method. But called here?
        default_mappings = {  # TODO central source for this!
            "hue": HueMapping,
        }
        for var, mapping in default_mappings.items():
            if var in all_data and var not in self._mappings:
                self._mappings[var] = mapping()  # TODO refactor w/above

        mappings = {}
        for var, mapping in self._mappings.items():
            if var in all_data:
                mapping.train(all_data[var])  # TODO return self?
                mappings[var] = mapping

        # TODO or something like this
        for layer in self._layers:

            # TODO alt. assign as attribute on Layer?
            layer_mappings = {k: v for k, v in mappings.items() if k in layer.data}

            self._plot_layer(layer, layer_mappings)

        return self

    def _plot_layer(self, layer, mappings):

        df = layer.data.frame

        # TODO where does this method come from?
        # data = self.as_numeric(layer.data)

        # TODO who owns the groupby logic?
        # data = self.stat(data)

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        # data = self.invert_scale(data)

        # Something like this?
        layer.mark._plot(df, mappings)  # do we pass ax (and/or facets?! here?)

    def show(self) -> Plot:

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise. In this vision, it would
        # make sense to specify whether or not to use pyplot at the initial Plot().
        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        import matplotlib.pyplot as plt  # type: ignore
        self.plot()
        plt.show()

        return self

    def save(self) -> Plot:  # or to_file or similar to match pandas?

        return self

    def _repr_html_(self) -> str:

        html = ""

        return html


# TODO
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
    names: dict[str, Optional[str]]
    _source: Optional[DataFrame | Mapping]

    def __init__(
        self,
        data: Optional[DataFrame | Mapping],
        variables: Optional[dict[str, Hashable | Vector]],
        # TODO pass in wide semantics?
    ):

        if variables is None:
            variables = {}

        # TODO only specing out with long-form data for now...
        frame, names = self._assign_variables_longform(data, variables)

        self.frame = frame
        self.names = names
        self._source = data

    def __contains__(self, key: Hashable) -> bool:
        return key in self.frame

    def update(
        # TODO name-wise, does update imply an inplace operation in a confusing way?
        # I think yes ... maybe "augment"?
        self,
        data: Optional[DataFrame | Mapping],
        variables: Optional[dict[str, Optional[Hashable | Vector]]],
    ) -> PlotData:

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

        # -- Update the inherited DataSource with this new information

        drop_cols = [k for k in self.frame if k in new.frame or k in disinherit]
        frame = (
            self.frame
            .drop(columns=drop_cols)
            .join(new.frame, how="outer")
        )

        names = {k: v for k, v in self.names.items() if k not in disinherit}
        names.update(new.names)

        new.frame = frame
        new.names = names

        return new

    def _assign_variables_longform(
        self,
        data: Optional[DataFrame | Mapping],
        variables: dict[str, Optional[Hashable | Vector]]
    ) -> tuple[DataFrame, dict[str, Optional[str]]]:
        """
        Define plot variables given long-form data and/or vector inputs.

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
        plot_data: dict[str, Vector] = {}
        var_names: dict[str, Optional[str]] = {}

        # Data is optional; all variables can be defined as vectors
        if data is None:
            data = {}

        # TODO Generally interested in accepting a generic DataFrame interface
        # Track https://data-apis.org/ for development

        # Variables can also be extracted from the index of a DataFrame
        index: dict[str, Any]
        if isinstance(data, pd.DataFrame):
            index = data.index.to_frame().to_dict(
                "series")  # type: ignore  # data-sci-types wrong about to_dict return
        else:
            index = {}

        for key, val in variables.items():

            # Simply ignore variables with no specification
            if val is None:
                continue

            # Try to treat the argument as a key for the data collection.
            # But be flexible about what can be used as a key.
            # Usually it will be a string, but allow other hashables when
            # taking from the main data object. Allow only strings to reference
            # fields in the index, because otherwise there is too much ambiguity.
            try:
                val_as_data_key = (
                    val in data
                    or (isinstance(val, str) and val in index)
                )
            except (KeyError, TypeError):
                val_as_data_key = False

            if val_as_data_key:

                if val in data:
                    plot_data[key] = data[val]  # type: ignore # fails on key: Hashable
                elif val in index:
                    plot_data[key] = index[val]  # type: ignore # fails on key: Hashable
                var_names[key] = str(val)

            elif isinstance(val, str):

                # This looks like a column name but we don't know what it means!
                # TODO improve this feedback to distinguish between
                # - "you passed a string, but did not pass data"
                # - "you passed a string, it was not found in data"

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
        frame = pd.DataFrame(plot_data)

        # Reduce the variables dictionary to fields with valid data
        names: dict[str, Optional[str]] = {
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
    group_vars: list[str] = ["col", "row", "group"]

    default_stat: Optional[Stat] = None  # TODO or identity?

    pass


class Point(Mark):

    def __init__(self, **kwargs):

        self.kwargs = kwargs

    def _plot(self, data, mappings):

        kws = self.kwargs.copy()

        # TODO! Get this from plotter or generator ===
        import matplotlib.pyplot as plt
        ax = plt.gca()
        # ===

        # TODO since names match, can probably be automated!
        if "hue" in data:
            c = mappings["hue"](data["hue"])
        else:
            c = None

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        ax.scatter(x=data["x"], y=data["y"], c=c, **kws)


class Layer:

    # Does this need to be anything other than a simple container for these attributes?
    # Could use a Dataclass I guess?

    def __init__(self, data: PlotData, mark: Mark, stat: Stat = None):

        self.data = data
        self.mark = mark
        self.stat = stat


class SemanticMapping:

    def __call__(self, x):  # TODO types; will need to overload (wheee)
        # TODO this is a hack to get things working
        # We are missing numeric maps and lots of other things
        if isinstance(x, pd.Series):
            if x.dtype.name == "category":  # TODO! possible pandas bug
                x = x.astype(object)
            return x.map(self.lookup_table)
        else:
            return self.lookup_table[x]


# TODO Currently, the SemanticMapping objects are also the source of the information
# about the levels/order of the semantic variables. Do we want to decouple that?

# In favor:
# Sometimes (i.e. categorical plots) we need to know x/y order, and we also need
# to know facet variable orders, so having a consistent way of defining order
# across all of the variables would be nice.

# Against:
# Our current external interface consumes both mapping parameterization like the
# color palette to use and the order information. I think this makes a fair amount
# of sense. But we could also break those, e.g. have `scale_fixed("hue", order=...)`
# similar to what we are currently developing for the x/y. Is is another method call
# which may be annoying. But then alternately it is maybe more consistent (and would
# consistently hook into whatever internal representation we'll use for variable order).
# Also, the parameters of the semantic mapping often implies a particular scale
# (i.e., providing a palette list forces categorical treatment) so it's not clear
# that it makes sense to determine that information at different points in time.


class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""

    # TODO type the important class attributes here

    def __init__(
        self,
        palette: Optional[PaletteSpec] = None,
        order: Optional[list] = None,
        norm: Optional[Normalize] = None,
    ):

        self._input_palette = palette
        self._input_order = order
        self._input_norm = norm

    def train(  # TODO ggplot name; let's come up with something better
        self,
        data: Series,
    ) -> None:

        palette: Optional[PaletteSpec] = self._input_palette
        order: Optional[list] = self._input_order
        norm: Optional[Normalize] = self._input_norm
        cmap: Optional[Colormap] = None

        # TODO these are currently extracted from a passed in plotter instance
        # can we avoid doing that now that we are deferring the mapping?
        input_format: Literal["long", "wide"] = "long"
        var_type = variable_type(data)

        if data.notna().any():

            map_type = self.infer_map_type(palette, norm, input_format, var_type)

            # Our goal is to end up with a dictionary mapping every unique
            # value in `data` to a color. We will also keep track of the
            # metadata about this mapping we will need for, e.g., a legend

            # --- Option 1: numeric mapping with a matplotlib colormap

            if map_type == "numeric":

                data = pd.to_numeric(data)
                levels, lookup_table, norm, cmap = self.numeric_mapping(
                    data, palette, norm,
                )

            # --- Option 2: categorical mapping using seaborn palette

            elif map_type == "categorical":

                levels, lookup_table = self.categorical_mapping(
                    data, palette, order,
                )

            # --- Option 3: datetime mapping

            elif map_type == "datetime":
                # TODO this needs actual implementation
                cmap = norm = None
                levels, lookup_table = self.categorical_mapping(
                    # Casting data to list to handle differences in the way
                    # pandas and numpy represent datetime64 data
                    list(data), palette, order,
                )

            else:
                raise RuntimeError()  # TODO should never get here ...

            # TODO do we need to return and assign out here or can the
            # type-specific methods do the assignment internally

            # TODO I don't love how this is kind of a mish-mash of attributes
            # Can we be more consistent across SemanticMapping subclasses?
            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.norm = norm
            self.cmap = cmap

    def infer_map_type(
        self,
        palette: Optional[PaletteSpec],
        norm: Optional[Normalize],
        input_format: Literal["long", "wide"],
        var_type: Optional[Literal["numeric", "categorical", "datetime"]],
    ) -> Optional[Literal["numeric", "categorical", "datetime"]]:
        """Determine how to implement the mapping."""
        map_type: Optional[Literal["numeric", "categorical", "datetime"]]
        if palette in QUAL_PALETTES:
            map_type = "categorical"
        elif norm is not None:
            map_type = "numeric"
        elif isinstance(palette, (dict, list)):
            map_type = "categorical"
        elif input_format == "wide":
            map_type = "categorical"
        else:
            map_type = var_type

        return map_type

    def categorical_mapping(
        self,
        data: Series,
        palette: Optional[PaletteSpec],
        order: Optional[list],
    ) -> tuple[list, dict]:
        """Determine colors when the hue mapping is categorical."""
        # -- Identify the order and name of the levels

        levels = categorical_order(data, order)
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            lookup_table = palette

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table

    def numeric_mapping(
        self,
        data: Series,
        palette: Optional[PaletteSpec],
        norm: Optional[Normalize],
    ) -> tuple[list, dict, Optional[Normalize], Colormap]:
        """Determine colors when the hue variable is quantitative."""
        cmap: Colormap
        if isinstance(palette, dict):

            # The presence of a norm object overrides a dictionary of hues
            # in specifying a numeric mapping, so we need to process it here.
            levels = list(sorted(palette))
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            lookup_table = palette.copy()

        else:

            # The levels are the sorted unique values in the data
            levels = list(np.sort(remove_na(data.unique())))

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # TODO do we want to do something complicated to ensure contrast?
            palette = "ch:" if palette is None else palette

            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)

            # Now sort out the data normalization
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "``hue_norm`` must be None, tuple, or Normalize object."
                raise ValueError(err)

            if not norm.scaled():
                norm(np.asarray(data.dropna()))

            lookup_table = dict(zip(levels, cmap(norm(levels))))

        return levels, lookup_table, norm, cmap


# TODO do modern functions ever pass a type other than Series into this?
def categorical_order(vector: Vector, order: Optional[Vector] = None) -> list:
    """
    Return a list of unique data values using seaborn's ordering rules.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    vector : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is None:

        # TODO We don't have Categorical as part of our Vector type
        # Do we really accept it? Is there a situation where we want to?

        # if isinstance(vector, pd.Categorical):
        #     order = vector.categories

        if isinstance(vector, pd.Series):
            if vector.dtype == "category":
                order = vector.cat.categories
            else:
                order = vector.unique()
        else:
            order = pd.unique(vector)

        if variable_type(vector) == "numeric":
            order = np.sort(order)

        order = filter(pd.notnull, order)
    return list(order)


class VarType(UserString):
    """
    Prevent comparisons elsewhere in the library from using the wrong name.

    Errors are simple assertions because users should not be able to trigger
    them. If that changes, they should be more verbose.

    """
    # TODO VarType is an awfully overloaded name, but so is DataType ...
    allowed = "numeric", "datetime", "categorical"

    def __init__(self, data):
        assert data in self.allowed, data
        super().__init__(data)

    def __eq__(self, other):
        assert other in self.allowed, other
        return self.data == other


def variable_type(
    vector: Vector,
    boolean_type: Literal["numeric", "categorical"] = "numeric",
) -> VarType:
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """

    # If a categorical dtype is set, infer categorical
    if is_categorical_dtype(vector):
        return VarType("categorical")

    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return VarType("numeric")

    # Special-case binary/boolean data, allow caller to determine
    # This triggers a numpy warning when vector has strings/objects
    # https://github.com/numpy/numpy/issues/6784
    # Because we reduce with .all(), we are agnostic about whether the
    # comparison returns a scalar or vector, so we will ignore the warning.
    # It triggers a separate DeprecationWarning when the vector has datetimes:
    # https://github.com/numpy/numpy/issues/13548
    # This is considered a bug by numpy and will likely go away.
    with warnings.catch_warnings():
        warnings.simplefilter(
            action='ignore',
            category=(FutureWarning, DeprecationWarning)  # type: ignore  # mypy bug?
        )
        if np.isin(vector, [0, 1, np.nan]).all():
            return VarType(boolean_type)

    # Defer to positive pandas tests
    if is_numeric_dtype(vector):
        return VarType("numeric")

    if is_datetime64_dtype(vector):
        return VarType("datetime")

    # --- If we get to here, we need to check the entries

    # Check for a collection where everything is a number

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    if all_numeric(vector):
        return VarType("numeric")

    # Check for a collection where everything is a datetime

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    if all_datetime(vector):
        return VarType("datetime")

    # Otherwise, our final fallback is to consider things categorical

    return VarType("categorical")

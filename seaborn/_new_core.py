from __future__ import annotations
from typing import Union, Optional
from collections.abc import Hashable, Sequence, Mapping
from numbers import Number

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series


Vector = Union[Series, ndarray, Sequence, Number]

# TODO Should we define a DataFrame-like type that is Union[DataFrame, Mapping]?


class Plot:

    def __init__(
        self,
        data: Optional[Union[DataFrame, Mapping]] = None,
        **variables: Union[Hashable, Vector],
    ):

        # Note that we can't assume wide-form here if variables does not contain x or y
        # because those might get assigned in long-form fashion per layer.

        self.data = DataSource(data, variables)
        self.layers: list[Layer] = []

    def add(
        self,
        mark: Mark,
        stat: Stat = None,
        data: Union[DataFrame, Mapping, None] = None,
        variables: dict[str, Union[Hashable, Vector, None]] = None,
    ) -> Plot:

        # TODO what if in wide-form mode, we convert to long-form
        # based on the transform that mark defines?
        data = DataSource(data, variables).join(self.data)

        if stat is None:
            stat = mark.default_stat

        self.layers.append(Layer(data, mark, stat))

        return self

    def plot(self) -> Plot:

        # TODO or something like this
        for layer in self.layers:
            self._plot_layer(layer)

        return self

    def _plot_layer(self, layer):

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
        layer.mark.plot(data)  # do we pass ax (and/or facets?! here?)

    def show(self) -> Plot:

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise
        import matplotlib.pyplot as plt
        self.plot()
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


class DataSource:

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

    def __init__(self, data, variables):

        pass

    def join(self, other: DataSource):

        pass


class Stat:

    pass


class Mark:

    # TODO will subclasses overwrite this? Should this be defined elsewhere?
    group_vars: list[str] = ["col", "row", "group"]

    default_stat: Optional[Stat] = None  # TODO or identity?

    pass


class Point(Mark):

    def _plot(self, data_gen):  # TODO data_gen is maybe too restrictive a name?

        for keys, data, ax in data_gen(self.group_vars):

            pass


class Layer:

    # Does this need to be anything other than a simple container for these attributes?
    # Could use a Dataclass I guess?

    def __init__(self, data: DataSource, mark: Mark, stat: Stat = None):

        self.data = data
        self.mark = mark
        self.stat = stat

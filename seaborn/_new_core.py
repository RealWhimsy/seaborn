

class Plot:

    def __init__(self, data=None, **variables):

        self.data = DataSource(data, variables)
        self.layers = []

    def add(self, mark, stat=None, data=None, variables=None):

        # TODO what if in wide-form mode, we convert to long-form
        # based on the transform that mark defines?
        data = DataSource(data, variables).join(self.data)

        if stat is None:
            stat = mark.default_stat

        self.layers.append(Layer(data, mark, stat))

    def plot(self):

        # TODO or something like this
        for layer in self.layers:
            layer.plot()

    def show(self):

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise
        import matplotlib.pyplot as plt
        self.plot()
        plt.show()

    def save(self):  # or to_file or similar to match pandas?

        pass

    def _repr_html_(self):

        pass


class Layer:

    def __init__(self, data, mark, stat):

        self.data = data
        self.mark = mark
        self.stat = stat

    def plot(self):

        # Do we need this method, or can Plot just have a plot_layer method that
        # each entry in the layers list gets passed to? (Meaning that Layer would just
        # be a simple Dataclass or similar?

        # Roughly ...

        # TODO where does this method come from?
        data = self.as_numeric(self.data)

        # TODO who owns the groupby logic?
        data = self.stat(data)

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        data = self.invert_scale(data)

        # Something like this?
        self.mark.plot(data)  # do we pass ax (and/or facets?! here?)


class DataSource:

    def __init__(self, data, variables):

        pass

    def join(self, other):

        pass


class Mark:

    pass


class Point(Mark):

    def plot(self):

        # TODO how do we end up with ax, data, etc?
        pass

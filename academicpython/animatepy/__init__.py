import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


class animate():
    """Tools to create animation and display it in Jupyter Notebook

    """

    def __init__(self, **kwargs):
        global fig
        fig = plt.figure(**kwargs)
        self.img = []

    def display(self, **kwargs):
        """Display a list of figures as HTML movie

        """

        plt.close()
        # _kwargs = {'interval': 50, 'repeat_delay': 10000}
        _kwargs = {'interval': 50}
        for kwarg in kwargs:
            _kwargs[kwarg] = kwargs[kwarg]
        ani = animation.ArtistAnimation(fig, self.img, **_kwargs)
        return HTML(ani.to_html5_video())


def animateit(f, ims, **kwargs):
    _kwargs = {'interval': 50}
    for kwarg in kwargs:
        _kwargs[kwarg] = kwargs[kwarg]
    ani = animation.ArtistAnimation(f, ims, **_kwargs)
    return HTML(ani.to_html5_video())

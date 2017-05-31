import numpy as np
from cmath import sqrt, tanh
from math import log
import scipy
from scipy.integrate import quad
from scipy.special import iv as I
from scipy.stats import lognorm
from scipy.constants import e, N_A
import matplotlib.pyplot as plt
import functools
from lmfit import minimize, Parameters, fit_report

# Constants
j = 1j  # for complex number use
test = True # test do/no go
v = False # Test save
w = True # Test show


class Memoized2(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        fn = functools.partial(self.__call__, obj)
        fn.reset = self._reset
        return fn

    def _reset(self):
        """Reset the cache."""
        self.cache = {}


class Bazant_Model():
    """Model developed by Song and Bazant."""

    def __init__(self, dimension, L_avg, Atot=1):
        """Initialize model properties"""

        self.n = dimension  # 1 = planar, 2 = cyclindrical, 3 = spherical

        # Fitting Parameters
        self.L_avg = L_avg  # average length
        self.Atot = Atot
        self.rhoct = None  # charge transfer resisitivity
        self.Dch = None  # diffusivity
        self.Cdl = None  # double layer capaitance
        self.Rextd = None  # resistance of exernal connections and electrolyte
        self.dphidc = None  # potential/concentration derivative
        self.sigma = -1  # standard deviation

        # Things which will be calculated
        self.dist = None  # will be a lognormal dist object
        self.integral_scale = 1  # integral scale, default 1 for n < 3

        # This will return the dimensionless impedance and will be a function
        # of w only
        self.Zdim = None

        # This will return the dimensional impedance, rather than the
        # dimensionless
        self.Z = None

    def set_parameters(self, params):
        """Set the model properties to desired values"""
        sigma = params['sigma'].value

        if sigma != self.sigma:
            self.PrL.reset()

        self.rhoct = params['rhoct'].value
        self.Dch = params['Dch'].value
        self.Cdl = params['Cdl'].value
        self.Rextd = params['Rextd'].value
        self.dphidc = params['dphidc'].value
        self.sigma = sigma
        if sigma > 0:
            self.probL()  # calculate the lognomral pdf at the given sigma
            if self.n == 3:
                self.scale_set()
        self.Z_overall()  # calculate the whole model into a function

    def rhoD(self, l):
        """Fuction given a length returns the diffusion resistivity using
        previously defined properties"""
        # Note this is defined with dim'less l to use in integration later
        res = -(self.dphidc * l) / (e * self.Dch)
        return res

    def rhoDct(self, l):
        """Returns rhoD/ct given a diffusion length using previously set
        properties"""
        # This is a dimensionless quantity
        return self.rhoD(l) / self.rhoct

    def wd(self, l):
        """Returns omega D, the variable used to nondimensionalize the frequency
        as a function of length"""
        return self.Dch / l ** 2

    def apdim(self, l):
        """Returns the dimensionless area of the particle as a function of
        diffusion length"""
        return l ** (self.n - 1)

    @Memoized2
    def zddim(self, w):
        """Given a dimensionless frequency, returns the bulk diffusion
        impedance depending on the dimensionality of the particles"""
        # Threshold to use for high freq approximations
        w_high = 5000

        # parameters used in following equations
        a = sqrt(j * w)
        b = sqrt(2 * j * w)

        if self.n == 1:
            if w < w_high:
                return (a * tanh(a)) ** -1
            else:
                # This is a high frequency approximation
                return b ** -1 * (1 - j)
        elif self.n == 2:
            if w < w_high:
                return I(0, a) / (a * I(1, a))
            else:
                # This is a high frequency approximation
                return b ** -1 * (1 - j) - j * b ** -1
        elif self.n == 3:
            if w < w_high:
                return tanh(a) / (a - tanh(a))
            else:
                # This is a high frequency approximation
                return b ** -1 * (1 - j) - j * w ** -1

    def probL(self, plot=False):
        """Sets the probability distribution function for the model. Uses the
        given standard deviation to create a lognormal probability
        distribution"""

        # Converts from normal distribution (avg, std) to lognormal mean and
        # standard deviation (mu, sig)
        avg = 1
        std = self.sigma
        # mu formula typically has log(fn) but lognorm requires the exp of the
        # mean, so the log is cancelled
        mu = avg ** 2 / (std ** 2 + avg ** 2) ** 0.5
        sig = (log(std ** 2 / avg ** 2 + 1)) ** 0.5

        # Call lognorm from scipy.stats
        dist = lognorm(sig, scale=mu)
        self.dist = dist

        # Check that the resulting pdf fits the necessary constraints
        assert np.absolute(1 - dist.mean()) < 1e-5
        #assert np.absolute(1 - dist.cdf(500)) < 1e-5

        # Used for debugging to view the probability distribution directly
        if plot is True:
            plt.figure()
            plt.plot(np.linspace(0, 5, 250), dist.pdf(np.linspace(0, 5, 250)))
            plt.plot(np.linspace(0, 5, 250), dist.cdf(np.linspace(0, 5, 250)))
            plt.xlabel('L')
            plt.ylabel('probability density')
            plt.show()

    @Memoized2
    def PrL(self, l):
        return self.dist.pdf(l)

    def scale_set(self):
        """Sets the necessary scaling for spherical particles"""
        assert self.n == 3, 'Scaling improperly for non-spherical particles'

        # Define integrand for scaling factor
        def scaling_integrand(ldim):
            return self.PrL(ldim) * ldim ** 2

        # Calculate scaling factor and set as class property
        self.integral_scale = quad(scaling_integrand, 0, np.inf)[0]

    def Yc(self, w):
        """Return the admittance of a capacitor given dimensional frequency
        and previously defined parameters, result is unitless"""
        return j * self.rhoct * self.Cdl * w

    def complex_quadrature(self, func, a, b, **kwargs):
        """
        A function that will evaluate an integral for complex values, takes as input
        a function, the starting value, and the final value of the integral
        Taken from:
        http://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
        """

        def real_func(x):
            return scipy.real(func(x))

        def imag_func(x):
            return scipy.imag(func(x))

        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])

    def Ybd(self, wx):
        """The bulk diffusion addmitance. Takes in dimensional frequency
        and uses many other properties to calculate the diffusion impedance
        of the film based on the size of the nanoparticles."""

        # Perfect size distribution, evaluate with analytial solution
        if self.sigma == 0:
            # nondimensionalize frequency and calculate diffusion impedance
            w = wx / self.wd(self.L_avg)
            zd = self.zddim(w)

            # Perfect distribution, no need to integrate
            res = (1 + self.rhoDct(self.L_avg) * zd) ** -1
            return res

        # Calculate using integration over variety of particle size
        elif self.sigma > 0:
            # Set up the integrand as a f(ldim) ldim is dimensionless dif length
            def integrand(ldim):

                # Nondimensionalize as a function of dimless l and cal impedance
                w = wx / self.wd(self.L_avg * ldim)
                zd = self.zddim(w)

                # calculate rhoD/ct as a function of dimless l
                pDct = self.rhoDct(self.L_avg * ldim)

                # return the resultant calculation to the integral
                return self.PrL(ldim) * self.apdim(ldim) * (1 + pDct * zd) ** -1

            # Evaluate the integral for the diffusion admittance
            Ybddim = self.complex_quadrature(integrand, 0, np.inf)
            # return only the value, not the error, scaled correctly
            return Ybddim[0] / self.integral_scale
        else:
            # This only happens if there is a negative stdev
            raise ValueError('Negative std deviation entered')

    def nyquist(self, Z, param=None):
        """Plots the result on a nyquist plot"""
        plt.plot(Z.real, -Z.imag, label=param, lw=2)
        plt.xlabel('Z')
        plt.ylabel("-Z''")
        plt.xlim(0, 3)
        plt.ylim(0, 3)

    def Z_overall(self):
        """Creates the functions for pulling impedance as a fn of frequency
        off a class object"""

        # Define and set dimensionless function
        def Z_func_d(w):
            ybd = self.Ybd(w)
            yc = self.Yc(w)
            return self.Rextd + (yc + ybd) ** -1

        self.Zdim = Z_func_d

        # Define and set dimensional version
        def Z_func(w):
            return Z_func_d(w) * self.rhoct / self.Atot

        self.Z = Z_func

    def residual(self, params, ws, Z, eps_data=None):
        self.set_parameters(params)
        vZ = np.vectorize(self.Z)

        diff = vZ(ws) - Z

        z1d = np.zeros(Z.size * 2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real
        z1d[1:z1d.size:2] = diff.imag
        return z1d

################################################################################

def read_impedance_file(filename):
    res = np.genfromtxt(filename, delimiter=',', skip_header=5,
                        usecols=(3, 13, 14))
    freq = res[:, 0]
    z_real = res[:, 1]
    z_imag = res[:, 2]
    z = z_real + j * z_imag

    return freq, z


def bazant_fit(filename):
    w, Z = read_impedance_file(filename)

    # Sample Measured Constants
    n = 2
    Atot = 50  # area total
    L_avg = 200e-9  # 400 nm in m

    # Parameters and initial guesses, along with fixing or bounds
    params = Parameters()
    params.add('rhoct', value=700 * 1e-4)  # Ohms*m2, charge transfer resistance
    params.add('Dch', value=1.29e-11 * 1e-4)  # m2/s
    params.add('Cdl', value=6.22e-7 * 1e4, min = 0)  # F/m2
    params.add('Rextd', value=0.5)  # dim'less
    params.add('dphidc', value=-301 * 1e-4 / N_A)  # V*m^3
    params.add('sigma', value=0.5, min = 0) # std deviation

    model = Bazant_Model(n, L_avg, Atot)
    model.set_parameters(params)

    out = minimize(model.residual, params, args=(w, Z))

    print(fit_report(out))

# bazant_fit('data/TH27_Second_Dis.csv')


################################################################################

def model_test(save=False, show=False):
    # Define a frequency vector to use for testing
    fake_frequency = np.logspace(-1, 5)

    # Sample Measured Constants
    Atot = 3  # area total
    L_avg = 40e-9  # 400 nm in m

    params = Parameters()
    params.add('rhoct', value=700 * 1e-4)  # Ohms*m2, charge transfer resistance
    params.add('Dch', value=1.29e-11 * 1e-4)  # m2/s
    params.add('Cdl', value=6.22e-7 * 1e4)  # F/m2
    params.add('Rextd', value=0.5)  # dim'less
    params.add('dphidc', value=-301 * 1e-6 / N_A)  # V*m^3

    ns = [1, 2, 3]  # dimensionality, manually change to adjust dimensionality
    sigmas = [0, 0.25, 0.5, 1]  # stdevs to be plotted

    for n in ns:
        plt.figure(n)

        model = Bazant_Model(n, L_avg, Atot)
        for sig in sigmas:
            Z = np.zeros(len(fake_frequency), dtype=np.complex)
            params.add('sigma', value=sig)

            model.set_parameters(params)
            for i, w in enumerate(fake_frequency):
                Z[i] = model.Zdim(w)

            model.nyquist(Z, sig)
            print('n = {0}, Sigma = {1} Finished'.format(n, sig))

        # add some things to the plot
        plt.legend()
        plt.title('n = {0}'.format(n))

        # Save or show the plot
        if save is True:
            plt.savefig('Plots/nanowire_imp_n{0}'.format(n))
            print('Plot Saved\n')
        else:
            print('\n')
    if show is True:
        print('Plot Showed\n')
        plt.show()

if test is True:
    model_test(save = v, show = w)

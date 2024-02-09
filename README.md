# retina-therm

A Green's function based model for calculating the temperature rise caused by laser exposure to the retina.
The model supports both flat top and Gaussian beam profiles and multiple absorbing
layers. It is written in Python, so can be installed with `pip` and run anywhere Python
runs.

It is currently a work in progress, but is already capable of simulating CW, pulsed, and
multi-pulsed exposures to a multi-layer retina.

# Derivation of Method

The program uses a Green's Function solution to the heat equation with a source term due
to Beer's Law absorption in one or more layers of material that have the same thermal
properties. Unlike a finite-difference heat solver, this method can quickly calculate
the temperature rise at a single point in space and time. That means it can be much
faster that an FD solver if all you want/need is the temperature at a single point.

However, it is more limited than an FD solver, the solution assumes that the media is
infinite (no boundaries), and homogeneous with respect to thermal properties. When
simulating retina exposures, the thermal properties of all retinal layers and
surrounding tissue are typically assumed to be the same as water, so this method
can still be applied to retinal exposures. It cannot be applied to skin exposures
where the different layers have different thermal properties and there is an air-skin
surface boundary.

See the [write-up](./doc//writeups/derivation/2023-GreensFunctionSolutionForRetinaLaserExposure.pdf) for a detailed derivation of the method.

# Installing

Install with pip

```bash
$ pip install retina-therm
```

# Usage

`retina-therm` is a library (module) that implements the various models with a CLI that
can configure and run models from a YAML config files. If you want to write your own
CLI, or embed a model in your own module, see the CLI source or unit tests for examples
of how to set up and run the models.

If you just want to run a simulation, you can do so by running the CLI with an argument
specifying the configuration file.

```bash
$ retina-therm temperature-rise config.yml
```

Here is an example configuration that simulates one of the exposures simulated by
[Mainster in 1970](https://pubmed.ncbi.nlm.nih.gov/5416049/) (This is a great paper by
the way).

```yaml
thermal:
    k: 1.5e-3 cal / K / s / cm
    rho: 1 g/cm^3
    c: 1 cal / K /g
layers:
  - mua: 310 1/cm
    d: 10 um
    z0: 0 um
  - mua: 53 1/cm
    d: 100 um
    z0: 10 um
laser:
  wavelength: 700 nm
  duration: 10 s
  E0: 1 cal/s/cm^2
  R: 10 um

simulation:
  sensor:
    z: 1 um
  time:
    max: 1 s
    dt: 10 us
  datout: retina-therm-{c[cmd]}-{c[laser/R]}.txt
  with_units: False
  use_multi_precision: False
  use_aoproximate: True
```

Note how physical quantities are given with units. `retina-therm` uses
[Pint](https://pint.readthedocs.io/en/stable/) internally
to do unit conversions, so you can specify configuration parameters in whatever unit you
have, no need to convert beforehand.
